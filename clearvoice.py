import math
import re
import struct
import pickle

import numpy as np

from tinygrad.helpers import fetch, partition
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict, safe_save
from tinygrad import Tensor, dtypes, nn, TinyJit, Variable
import json, urllib.request, typing, unicodedata, sys

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]:
    return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)
  
# ios needs to be like 500 max_len and 15s ref len for RAM
MAX_LEN = 1000
REF_AUDIO_LEN = 15
FRAME_RATE = 25
AUDIO_CHUNK_DURATION = 15.0
POSITION_TEMP = 5.0
LAYER_PENTALTY_FACTOR = 5.0
GUIDANCE_SCALE = 2.0
T_SHIFT = 0.1
AUDIO_CHUNKED_THRESHOLD = 30.0

HIDDEN_SIZE = 1024
NUM_AUDIO_CODEBOOK = 8
AUDIO_VOCAB_SIZE = 1025
AUDIO_CODEBOOK_WEIGHTS = [8, 8, 6, 6, 4, 4, 2, 2]
AUDIO_MASK_ID = 1024
SAMPLING_RATE = 24000
# saved from getting all chars with https://github.com/k2-fsa/OmniVoice/blob/9948396864cb713b0c2f92495cf4449bd8717127/omnivoice/utils/duration.py#L204
with open("char_weights.json", "r") as f: CHAR_WEIGHTS = json.load(f)

data = json.load(urllib.request.urlopen("https://huggingface.co/k2-fsa/OmniVoice/resolve/main/tokenizer.json"))
special_tokens = data["added_tokens"]
special_tokens = {item['content']: item['id'] for item in special_tokens}
tok = SimpleTokenizer(normal_tokens=data["model"]["vocab"], special_tokens=special_tokens)

import io
def waveform_to_wav_bytes(audio, sample_rate: int):
    channels = 1
    audio_clipped = [max(-1.0, min(1.0, x)) for x in audio]
    audio_int16 = [int(x * 32767.0) for x in audio_clipped]

    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    data_size = len(audio_int16) * 2
    chunk_size = 36 + data_size

    buf = io.BytesIO()

    buf.write(b'RIFF')
    buf.write(struct.pack('<I', chunk_size))
    buf.write(b'WAVE')

    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<H', channels))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', byte_rate))
    buf.write(struct.pack('<H', block_align))
    buf.write(struct.pack('<H', 16))

    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(struct.pack(f'<{len(audio_int16)}h', *audio_int16))

    return buf.getvalue()

def load_waveform(data):
  sample_rate = struct.unpack_from('<I', data, 24)[0]
  channels = struct.unpack_from('<H', data, 22)[0]
  data_offset = data.find(b'data') + 8
  raw_samples = data[data_offset:]
  n_samples = len(raw_samples) // 2  # int16 = 2 bytes
  samples = struct.unpack(f'<{n_samples}h', raw_samples)
  audio = [[] for _ in range(channels)]
  for i in range(0, len(samples), channels):
    frame = samples[i:i + channels]
    for ch in range(channels):
      audio[ch].append(frame[ch] / 32768.0)

  return audio, sample_rate

import bisect

def interp_1d(x, xp, fp):
  out = []
  n = len(xp)

  for xi in x:
    if xi <= xp[0]:
      out.append(fp[0])
    elif xi >= xp[-1]:
      out.append(fp[-1])
    else:
      i = bisect.bisect_left(xp, xi)
      x0, x1 = xp[i - 1], xp[i]
      y0, y1 = fp[i - 1], fp[i]
      t = (xi - x0) / (x1 - x0)
      out.append(y0 + t * (y1 - y0))

  return out

def resample(data, orig_sr, target_sr):
  # data is always multi-channel, shape (channels, samples)
  duration = len(data[0]) / orig_sr

  orig_times = [i * duration / len(data[0]) for i in range(len(data[0]))]
  new_length = int(duration * target_sr)
  new_times = [i * duration / new_length for i in range(new_length)]
  
  resampled_channels = [interp_1d(new_times, orig_times, channel) for channel in data]
  return resampled_channels

def load_audio(audio, sampling_rate: int):
  data, sr = load_waveform(audio)
  data = [sum(samples) / len(samples) for samples in zip(*data)]
  data = resample([data], sr, sampling_rate)[0]
  rms = math.sqrt(sum(x * x for x in data) / len(data))
  if 0 < rms < 0.1:
    scale = 0.1 / rms
    data = [x * scale for x in data]

  return data

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

class Qwen3RMSNorm:
  def __init__(self, sz=1024):
    self.variance_epsilon = 1e-6
    self.weight = Tensor.empty(sz)
  
  def __call__(self, hidden_states):
    hidden_states = hidden_states.cast(dtypes.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states * Tensor.rsqrt(variance + self.variance_epsilon))
    return self.weight * hidden_states
  
def repeat_kv(hidden_states, n_rep: int):
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed

def rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return Tensor.cat(-x2, x1, dim=-1)

class Qwen3Attention:
  def __init__(self):
    self.head_dim = 128
    self.q_norm = Qwen3RMSNorm(sz=128)
    self.k_norm = Qwen3RMSNorm(sz=128)
    self.q_proj = nn.Linear(in_features=1024, out_features=2048, bias=False)
    self.k_proj = nn.Linear(in_features=1024, out_features=1024, bias=False)
    self.v_proj = nn.Linear(in_features=1024, out_features=1024, bias=False)
    self.o_proj = nn.Linear(in_features=2048, out_features=1024, bias=False)
    self.scaling = 0.08838834764831845
    self.num_key_value_groups = 2

  def __call__(self, hidden_states, position_embeddings, attention_mask):
      position_embeddings = position_embeddings
      input_shape = hidden_states.shape[:-1]
      hidden_shape = (*input_shape, -1, self.head_dim)

      x = self.q_proj(hidden_states).view(hidden_shape)
      query_states = self.q_norm(x).transpose(1, 2)

      x = self.k_proj(hidden_states).view(hidden_shape)
      key_states = self.k_norm(x).transpose(1, 2)

      value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

      cos, sin = position_embeddings
      query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

      key_states = repeat_kv(key_states, self.num_key_value_groups)
      value_states = repeat_kv(value_states, self.num_key_value_groups)

      attn_output = Tensor.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask
        ).transpose(1, 2).contiguous()
      
      attn_output = attn_output.reshape(*input_shape, -1).contiguous()
      attn_output = self.o_proj(attn_output)
      return attn_output

class Qwen3RotaryEmbedding:
  def __init__(self):
    self.attention_scaling = 1.0

  def __call__(self, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].cast(dtypes.float).expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].cast(dtypes.float)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    cos = (emb.cos() * self.attention_scaling).cast(dtypes.float16)
    sin = (emb.sin() * self.attention_scaling).cast(dtypes.float16)
    return cos, sin

class Qwen3MLP():
  def __init__(self):
    self.down_proj = nn.Linear(in_features=3072, out_features=1024, bias=False)
    self.gate_proj = nn.Linear(in_features=1024, out_features=3072, bias=False)
    self.up_proj = nn.Linear(in_features=1024, out_features=3072, bias=False)
    self.act_fn = Tensor.silu
  
  def __call__(self, x): return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen3DecoderLayer:
  def __init__(self):
    self.input_layernorm = Qwen3RMSNorm()
    self.self_attn = Qwen3Attention()
    self.post_attention_layernorm = Qwen3RMSNorm()
    self.mlp = Qwen3MLP()
  
  def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
      residual = hidden_states
      hidden_states = self.input_layernorm(hidden_states)
      hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
      hidden_states = residual + hidden_states

      # Fully Connected
      residual = hidden_states
      hidden_states = self.post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states)
      hidden_states = residual + hidden_states
      return hidden_states


class llm:
  def __init__(self):
    self.embed_tokens = nn.Embedding(151676, 1024)
    self.norm = Qwen3RMSNorm()
    self.rotary_emb = Qwen3RotaryEmbedding()
    self.layers = []
    for i in range(28):
      self.layers.append(Qwen3DecoderLayer())

  def __call__(self, attention_mask=None, position_ids=None, inputs_embeds=None):
      position_ids = Tensor.arange(inputs_embeds.shape[1])
      position_ids = position_ids.unsqueeze(0)

      hidden_states = inputs_embeds
      position_embeddings = self.rotary_emb(position_ids)
      
      for decoder_layer in self.layers:
        hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings,)

      return self.norm(hidden_states)

class HubertModel:
  def __init__(self):
    self.feature_extractor = HubertFeatureEncoder()
    self.feature_projection = HubertFeatureProjection()
    self.encoder = HubertEncoder()

  def __call__(self, input_values):
      extract_features = self.feature_extractor(input_values)
      extract_features = extract_features.transpose(1, 2)
      hidden_states = self.feature_projection(extract_features)
      encoder_outputs = self.encoder(hidden_states)
      return encoder_outputs

class HubertPositionalConvEmbedding:
  def __init__(self):
    self.conv = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=128, stride=1, padding=64, groups=16)
    self.activation = Tensor.gelu
  
  def __call__(self, hidden_states):
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = self.conv(hidden_states)    
    #https://github.com/huggingface/transformers/blob/c5deba28c83d853a1f63a0ab589a4531346fbcb0/src/transformers/models/hubert/modeling_hubert.py#L102
    hidden_states = hidden_states[:, :, : -1]
    hidden_states = self.activation(hidden_states).transpose(1, 2)
    return hidden_states

class HubertFeedForward:
  def __init__(self):
    self.intermediate_dense = nn.Linear(in_features=768, out_features=3072, bias=True)
    self.intermediate_act_fn = Tensor.gelu
    self.output_dense = nn.Linear(in_features=3072, out_features=768, bias=True)

  def __call__(self, hidden_states):
    hidden_states = self.intermediate_dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    hidden_states = self.output_dense(hidden_states)
    return hidden_states


class HubertAttention:
  def __init__(self):
    self.head_dim = 64
    self.q_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.k_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.v_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.out_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.scaling = 0.125
    self.is_causal = False
  
  def __call__(self, hidden_states):
      input_shape = hidden_states.shape[:-1]

      hidden_shape = (*input_shape, -1, self.head_dim)
      query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
      kv_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
      key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
      value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
      
      attn_output = Tensor.scaled_dot_product_attention(
              query_states,
              key_states,
              value_states,
          ).transpose(1, 2).contiguous()

      attn_output = attn_output.reshape(*input_shape, -1).contiguous()
      attn_output = self.out_proj(attn_output)

      return attn_output, None, None

class HubertEncoderLayer:
  def __init__(self):
    self.attention = HubertAttention()
    self.feed_forward = HubertFeedForward()
    self.layer_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    self.final_layer_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  def __call__(self, hidden_states):
    attn_residual = hidden_states
    hidden_states, _, _ = self.attention(hidden_states)
    hidden_states = attn_residual + hidden_states
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = hidden_states + self.feed_forward(hidden_states)
    hidden_states = self.final_layer_norm(hidden_states)
    outputs = (hidden_states,)
    return outputs

class HubertEncoder:
  def __init__(self):
    self.pos_conv_embed = HubertPositionalConvEmbedding()
    self.layer_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    self.layers = []
    for i in range(12): self.layers.append(HubertEncoderLayer())
  
  def __call__(self, hidden_states):
    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.layer_norm(hidden_states)

    all_hidden_states = ()
    for layer in self.layers:
        all_hidden_states = all_hidden_states + (hidden_states,)
        layer_outputs = layer(hidden_states)
        hidden_states = layer_outputs[0]
    all_hidden_states = all_hidden_states + (hidden_states,)

    return all_hidden_states

class HubertFeatureProjection:
  def __init__(self):
    self.layer_norm = nn.LayerNorm(512, eps=1e-05, elementwise_affine=True)
    self.projection = nn.Linear(in_features=512, out_features=768, bias=True)

  def __call__(self, hidden_states):
      hidden_states = self.layer_norm(hidden_states)
      hidden_states = self.projection(hidden_states)
      return hidden_states

class HubertGroupNormConvLayer:
  def __init__(self):
    self.conv = nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False)
    self.layer_norm = nn.GroupNorm(512, 512)
  
  def __call__(self, hidden_states):
    hidden_states = self.conv(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = Tensor.gelu(hidden_states)
    return hidden_states

class HubertNoLayerNormConvLayer:
  def __init__(self, k=3): self.conv = nn.Conv1d(512, 512, kernel_size=k, stride=2, bias=False)
  
  def __call__(self, hidden_states):
    hidden_states = self.conv(hidden_states)
    hidden_states = Tensor.gelu(hidden_states)
    return hidden_states

class HubertFeatureEncoder:
  def __init__(self):
    self.conv_layers = [HubertGroupNormConvLayer()]
    for i in range(1, 7):
      self.conv_layers.append(HubertNoLayerNormConvLayer(k=3 if i < 5 else 2))
  
  def __call__(self, input_values):
    hidden_states = input_values[:, None]
    for conv_layer in self.conv_layers:
      hidden_states = conv_layer(hidden_states)
    return hidden_states

class HiggsAudioV2TokenizerResidualUnit:
  def __init__(self):
    self.conv1 = nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv2 = nn.Conv1d(768, 768, kernel_size=1, stride=1, bias=False)

  def __call__(self, hidden_state):
    output_tensor = Tensor.elu(hidden_state)
    output_tensor = self.conv1(output_tensor)
    output_tensor = Tensor.elu(output_tensor)
    output_tensor = self.conv2(output_tensor)
    hidden_state = hidden_state + output_tensor
    return hidden_state

class HiggsAudioV2TokenizerSemanticEncoderBlock:
  def __init__(self):
    self.res_units = [HiggsAudioV2TokenizerResidualUnit(), HiggsAudioV2TokenizerResidualUnit()]
    self.conv = nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1)
  
  def __call__(self, hidden_state):
    for unit in self.res_units:
        hidden_state = unit(hidden_state)
    hidden_state = self.conv(hidden_state)
    return hidden_state

class SemanticEncoder:
  def __init__(self):
     self.conv = nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
     self.conv_blocks = [HiggsAudioV2TokenizerSemanticEncoderBlock(), HiggsAudioV2TokenizerSemanticEncoderBlock()]
   
  def __call__(self, hidden_state):
    hidden_state = self.conv(hidden_state)
    for block in self.conv_blocks:
        hidden_state = block(hidden_state)
    return hidden_state

class Snake1d:
  def __init__(self, sz): self.alpha = Tensor.zeros(1, sz, 1)

  def __call__(self, hidden_states):
    shape = hidden_states.shape
    hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
    hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * Tensor.sin(self.alpha * hidden_states).pow(2)
    hidden_states = hidden_states.reshape(shape)
    return hidden_states

class DacEncoderBlock:
  def __init__(self, in_ch, out_ch, k, s, p):
    self.res_unit1 = DacResidualUnit(in_ch=in_ch, out_ch=in_ch, p1=3, d1=1)
    self.res_unit2 = DacResidualUnit(in_ch=in_ch, out_ch=in_ch, p1=9, d1=3)
    self.res_unit3 = DacResidualUnit(in_ch=in_ch, out_ch=in_ch, p1=27, d1=9)
    self.snake1 = Snake1d(in_ch)
    self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)

  def __call__(self, hidden_state):
    hidden_state = self.res_unit1(hidden_state)
    hidden_state = self.res_unit2(hidden_state)
    hidden_state = self.snake1(self.res_unit3(hidden_state))
    hidden_state = self.conv1(hidden_state)
    return hidden_state

class DacEncoder:
  def __init__(self):
    self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
    self.conv2 = nn.Conv1d(2048, 256, kernel_size=3, stride=1, padding=1)
    self.block = [DacEncoderBlock(64, 128, 16, 8, 4,),
                  DacEncoderBlock(128, 256, 10, 5, 3),
                  DacEncoderBlock(256, 512, 8, 4, 2),
                  DacEncoderBlock(512, 1024, 4, 2, 1),
                  DacEncoderBlock(1024, 2048, 6, 3, 2)]
    self.snake1 = Snake1d(2048)
  
  def __call__(self, hidden_state):
    hidden_state = self.conv1(hidden_state)
    for module in self.block:
        hidden_state = module(hidden_state)
    hidden_state = self.snake1(hidden_state)
    hidden_state = self.conv2(hidden_state)
    return hidden_state

class ConvTranspose1d:
  def __init__(self, in_ch, n, s, p, op):
    self.weight = Tensor.zeros([in_ch*2, in_ch, n], dtype=dtypes.float16)
    self.bias = Tensor.zeros([in_ch], dtype=dtypes.float16)
    self.stride = s
    self.padding = p
    self.kernel_size = self.padding*2
    self.output_padding = op
  
  def __call__(self, input):
    size, in_channels, in_width = input.shape
    _, _, kernel_size = self.weight.shape

    upsampled = Tensor.zeros(size, in_channels, in_width * self.stride - (self.stride - 1), dtype=dtypes.float)
    upsampled[:, :, ::self.stride] = input

    pad = 1 * (kernel_size - 1) - self.padding
    weight_flipped = self.weight.flip(-1)
    weight_conv = weight_flipped.permute(1, 0, 2)


    out = Tensor.conv2d(
        upsampled.unsqueeze(2),
        weight_conv.unsqueeze(2),
        bias=None,
        stride=(1, 1),
        padding=(0, pad),
        dilation=(1, 1),
        groups=1,
    ).squeeze(2)

    if self.output_padding > 0: out = Tensor.pad(out, (0, self.output_padding))
    out += self.bias.view(1, -1, 1)
    return out

class DacResidualUnit:
  def __init__(self, in_ch, out_ch, p1, d1):
    self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=1, padding=p1, dilation=d1)
    self.conv2 = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1)
    self.snake1 = Snake1d(in_ch)
    self.snake2 = Snake1d(in_ch)

  def __call__(self, hidden_state):
    output_tensor = hidden_state
    output_tensor = self.snake1(output_tensor)
    output_tensor = self.conv1(output_tensor)
    output_tensor = self.snake2(output_tensor)
    output_tensor = self.conv2(output_tensor)
    padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
    output_tensor = hidden_state + output_tensor
    return output_tensor    

class DacDecoderBlock:
  def __init__(self, in_ch, n, s, p, op):
    self.snake1 = Snake1d(in_ch*2)
    self.conv_t1 = ConvTranspose1d(in_ch=in_ch, n=n, s=s, p=p, op=op) # todo
    self.res_unit1 = DacResidualUnit(out_ch=in_ch, in_ch=in_ch, p1=3, d1=1)
    self.res_unit2 = DacResidualUnit(out_ch=in_ch, in_ch=in_ch, p1=9, d1=3)
    self.res_unit3 = DacResidualUnit(out_ch=in_ch, in_ch=in_ch, p1=27, d1=9)
   
  def __call__(self, hidden_state):
    hidden_state = self.snake1(hidden_state)
    hidden_state = self.conv_t1(hidden_state)
    hidden_state = self.res_unit1(hidden_state)
    hidden_state = self.res_unit2(hidden_state)
    hidden_state = self.res_unit3(hidden_state)
    return hidden_state

class DacDecoder:
  def __init__(self):
    self.conv1 = nn.Conv1d(256, 1024, kernel_size=7, stride=1, padding=3)
    self.conv2 = nn.Conv1d(32, 1, kernel_size=7, stride=1, padding=3)
    self.block = [DacDecoderBlock(512, 16, s=8, p=4, op=0),
                  DacDecoderBlock(256, 10, s=5, p=3, op=1),
                  DacDecoderBlock(128, 8, s=4, p=2, op=0),
                  DacDecoderBlock(64, 4, s=2, p=1, op=0),
                  DacDecoderBlock(32, 6, s=3, p=2, op=1)]
    self.snake1 = Snake1d(32)
  
  def __call__(self, hidden_state):
      hidden_state = self.conv1(hidden_state)
      for layer in self.block:
          hidden_state = layer(hidden_state)
      hidden_state = self.snake1(hidden_state)
      hidden_state = self.conv2(hidden_state)
      return hidden_state

class HiggsAudioV2TokenizerEuclideanCodebook:
  def __init__(self): self.embed = nn.Embedding(1024, 64)

class HiggsAudioV2TokenizerVectorQuantization:
  def __init__(self):
    self.codebook = HiggsAudioV2TokenizerEuclideanCodebook()
    self.project_in = nn.Linear(in_features=1024, out_features=64, bias=True)
    self.project_out = nn.Linear(in_features=64, out_features=1024, bias=True)

class HiggsAudioV2TokenizerResidualVectorQuantization:
  def __init__(self):
    self.quantizers = []
    for _ in range(8): self.quantizers.append(HiggsAudioV2TokenizerVectorQuantization())

  def encode(self, embeddings):
    residual = embeddings
    all_indices = []
    for q in self.quantizers:
      hidden_states = residual.permute(0, 2, 1)
      hidden_states = q.project_in(hidden_states)

      shape = hidden_states.shape
      hidden_states = hidden_states.reshape((-1, shape[-1]))

      embed = q.codebook.embed.weight.T
      scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
      dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
      indices = dist.argmax(axis=-1)

      indices = indices.view(*shape[:-1])
      quantized = q.codebook.embed(indices)
      quantized = q.project_out(quantized)
      quantized = quantized.permute(0, 2, 1)
      residual = residual - quantized
      all_indices.append(indices)
    out_indices = Tensor.stack(all_indices)
    return out_indices

class audio_tokenizer:
  def __init__(self):
    self.semantic_downsample_factor = 3
    self.hop_length = 960
    self.semantic_model = HubertModel()
    self.encoder_semantic = SemanticEncoder()
    self.acoustic_encoder = DacEncoder()
    self.acoustic_decoder = DacDecoder()
    self.quantizer = HiggsAudioV2TokenizerResidualVectorQuantization()
    self.fc = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 256)

  def _extract_semantic_features(self, input_values):
    input_values = input_values[:, 0, :]
    input_values = Tensor.pad(input_values, (160, 160))
    hidden_states = self.semantic_model(input_values)
    stacked = Tensor.stack([h for h in hidden_states], dim=1)
    semantic_features = stacked.mean(axis=1)
    semantic_features = semantic_features[:, :: self.semantic_downsample_factor, :]
    return semantic_features

  # todo variable size
  @TinyJit
  def decode(self, audio_codes):
      audio_codes = audio_codes.unsqueeze(0).transpose(0, 1)
      quantized_out = 0.0
      for i, indices in enumerate(audio_codes):
        quantizer = self.quantizer.quantizers[i]
        quantized = quantizer.codebook.embed(indices)
        quantized = quantizer.project_out(quantized)
        quantized = quantized.permute(0, 2, 1)
        quantized_out = quantized_out + quantized
      quantized = quantized_out
      quantized_acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
      hidden_state = self.acoustic_decoder.conv1(quantized_acoustic)

      for layer in self.acoustic_decoder.block: hidden_state = layer(hidden_state)

      hidden_state = self.acoustic_decoder.snake1(hidden_state)
      hidden_state = self.acoustic_decoder.conv2(hidden_state)
      return hidden_state[0, 0, :]

class omni:
  def __init__(self):
    self.llm = llm()
    self.audio_embeddings = nn.Embedding(NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, HIDDEN_SIZE)
    self.audio_heads = nn.Linear(HIDDEN_SIZE, NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, bias=False)
    weights = safe_load(fetch("https://huggingface.co/roryclear/OmniVoice-F16/resolve/main/model_f16.safetensors"))
    load_state_dict(self, weights)
    #for k,v in get_state_dict(self).items():
    #  if v.dtype == dtypes.float32: v.replace(v.cast(dtypes.float16))
    

    #https://github.com/huggingface/transformers/blob/f73cc1b1fe0477053638fc929546bac8b3697007/src/transformers/models/qwen3/modeling_qwen3.py#L130-L132
    self.llm.rotary_emb.inv_freq = 1.0 / (1000000 ** (Tensor.arange(0, 128, 2).cast(dtypes.float) / 128))

    self.audio_tokenizer = audio_tokenizer()
    weights = safe_load(fetch("https://huggingface.co/k2-fsa/OmniVoice/resolve/main/audio_tokenizer/model.safetensors"))
    weights["semantic_model.encoder.pos_conv_embed.conv.weight"] = weights["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
    for i in range(len(self.audio_tokenizer.quantizer.quantizers)):
      weights[f"quantizer.quantizers.{i}.codebook.embed.weight"] = weights[f"quantizer.quantizers.{i}.codebook.embed"]
    load_state_dict(self.audio_tokenizer, weights)
    #for k,v in get_state_dict(self.audio_tokenizer).items():
    #  if v.dtype == dtypes.float32: v.replace(v.cast(dtypes.float16))
    self.codebook_layer_offsets = (Tensor.arange(NUM_AUDIO_CODEBOOK) * AUDIO_VOCAB_SIZE)
  
  # todo change back to 16
  def generate(self, text, cv_path=None, ref_audio_tokens=None, num_steps=16, language="None"):
    with open(cv_path, "r") as f: vc = json.load(f)
    ref_text = vc["ref_text"]
    ref_audio = base64.b64decode(vc["ref_audio"])

    ref_wav = load_audio(ref_audio, SAMPLING_RATE)
    wav_len = len(ref_wav)
    ref_wav = self.expand_wav(ref_wav=ref_wav)
    ref_audio_tokens = self.encode(Tensor([[ref_wav]])) # todo, use 10s instead of 20s?
    ref_audio_tokens = ref_audio_tokens.numpy()
    ref_audio_tokens = ref_audio_tokens[0, :, :int(wav_len / self.audio_tokenizer.hop_length)]
  
    # so c_len doesn't exceed MAX_LEN
    style_tokens = tok.encode(f"<|denoise|><|lang_start|>{language}<|lang_end|><|instruct_start|>None<|instruct_end|>")
    chunks = self.get_chunks(text, ref_text, wav_len, style_tokens, ref_audio_tokens)
    res = []
    rets = []
    target_lengths = []
    for i in range(len(chunks)):
      target_length = self._estimate_target_tokens(chunks[i], ref_text, int(wav_len / self.audio_tokenizer.hop_length))
      text_tokens = tok.encode(f"<|text_start|>{' '.join(x.strip() for x in (ref_text, chunks[i]) if x.strip())}<|text_end|>")
      ret = self._generate_iterative(text_tokens=text_tokens, target_length=target_length, ref_audio_tokens=ref_audio_tokens, num_steps=num_steps, style_tokens=style_tokens)
      rets.append(ret)
      target_lengths.append(target_length)
    for ret, target_length in zip(rets, target_lengths):
      target_lengths.append(target_length)
      wv = self.audio_tokenizer.decode(Tensor(ret)).numpy().tolist()
      wv = wv[:target_length * self.audio_tokenizer.hop_length]
      res.extend(wv)
    return res

  def get_chunks(self, text, ref_text, wav_len, style_tokens, ref_audio_tokens):
    chunks_small = re.findall(r"[^。，！？；：、.,?]+[。，！？；：、.,?]?", text) # eng and cn gaps
    chunks = [""]
    j = 0
    for i in range(len(chunks_small)):
      if chunks_small[i][0] == " ": chunks_small[i] = chunks_small[i][1:]
      target_length = self._estimate_target_tokens((chunks[j]+chunks_small[i]), ref_text, int(wav_len / self.audio_tokenizer.hop_length))
      text_tokens = tok.encode(f"<|text_start|>{' '.join(x.strip() for x in (ref_text, (chunks[j]+chunks_small[i])) if x.strip())}<|text_end|>")
      if len(style_tokens) + len(text_tokens) + len(ref_audio_tokens[0]) + target_length < MAX_LEN:
        chunks[j] += chunks_small[i]
      else:
        chunks.append(chunks_small[i])
        j+=1
    return chunks

  def expand_wav(self, ref_wav):
    chunk_size = self.audio_tokenizer.hop_length
    clip_size = int(len(ref_wav) % chunk_size)
    ref_wav = ref_wav[:-clip_size] if clip_size > 0 else ref_wav
    wav_len = len(ref_wav)
    if len(ref_wav) <= (SAMPLING_RATE*REF_AUDIO_LEN):
      ref_wav = ref_wav + [0] * ((SAMPLING_RATE*REF_AUDIO_LEN) - wav_len)
    else:
      ref_wav = ref_wav[:REF_AUDIO_LEN*SAMPLING_RATE]
    return ref_wav

  # https://github.com/huggingface/transformers/blob/1c75d06e73bf25d48a4379b9452ca009da9cf0a1/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py#L41
  @TinyJit
  def encode(self, input_values):
    e_semantic_input = self.audio_tokenizer._extract_semantic_features(input_values)
    e_semantic = self.audio_tokenizer.encoder_semantic(e_semantic_input.transpose(1, 2))
    e_acoustic = self.audio_tokenizer.acoustic_encoder(input_values)
    embeddings = Tensor.cat(e_acoustic, e_semantic, dim=1)
    embeddings = self.audio_tokenizer.fc(embeddings.transpose(1, 2)).transpose(1, 2)
    audio_codes = self.audio_tokenizer.quantizer.encode(embeddings)
    return audio_codes.transpose(0, 1)

  def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens):
    ref_weight = sum(CHAR_WEIGHTS[ord(c)] for c in ref_text)
    speed_factor = ref_weight / num_ref_audio_tokens
    target_weight = sum(CHAR_WEIGHTS[ord(c)] for c in text)
    estimated_duration = target_weight / speed_factor
    return int(estimated_duration)

  @TinyJit
  def __call__(self, input_ids, audio_mask, attention_mask, tokens, c_len_var, t_len_var):
    pred_tokens = Tensor.zeros(1, NUM_AUDIO_CODEBOOK, MAX_LEN)
    text_embeds = self.llm.embed_tokens(input_ids[:, 0, :])
    shifted_ids = input_ids * audio_mask.unsqueeze(1) + self.codebook_layer_offsets.view(1, -1, 1)
    audio_embeds = self.audio_embeddings(shifted_ids).sum(axis=1)
    inputs_embeds = Tensor.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)
    hidden_states = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits_flat = self.audio_heads(hidden_states)
    logits = logits_flat.view(2, c_len_var, NUM_AUDIO_CODEBOOK, AUDIO_VOCAB_SIZE).permute(0, 2, 1, 3)
    
    u_logits = logits[1, :, :t_len_var, :]
    c_logits = logits.flip(2)[0, :, :t_len_var, :].flip(1)
    c_log_probs = Tensor.log_softmax(c_logits, axis=-1)
    u_log_probs = Tensor.log_softmax(u_logits, axis=-1)
    log_probs = Tensor.log_softmax(c_log_probs + GUIDANCE_SCALE * (c_log_probs - u_log_probs), axis=-1,)

    pred_tokens[:, :, :t_len_var] += log_probs.argmax(axis=-1)
    scores = log_probs.max(axis=-1)[0] - (Tensor.arange(NUM_AUDIO_CODEBOOK).unsqueeze(-1) * LAYER_PENTALTY_FACTOR)

    scaled_logits = scores / POSITION_TEMP
    u = Tensor.rand(NUM_AUDIO_CODEBOOK, MAX_LEN)
    gumbel_noise = -Tensor.log(-Tensor.log(u + 1e-10) + 1e-10)
    scores = scaled_logits + gumbel_noise[:, :t_len_var]

    scores_out = Tensor.zeros(NUM_AUDIO_CODEBOOK, MAX_LEN)
    scores_out[:, :t_len_var] += Tensor.where(tokens[:, :t_len_var] == AUDIO_MASK_ID, scores[:, :t_len_var], -float("inf"))
    return pred_tokens, scores_out

  def get_inputs(self, text_tokens, target_length, ref_audio_tokens, style_tokens):
    target_audio_tokens = [AUDIO_MASK_ID for _ in range(target_length)]
    c_len = len(style_tokens) + len(text_tokens) + len(ref_audio_tokens[0]) + target_length
    cond_audio_start_idx = c_len - target_length - len(ref_audio_tokens[0])

    cond_input_ids = [[]]
    for i in range(NUM_AUDIO_CODEBOOK): cond_input_ids[0].append(style_tokens + text_tokens + ref_audio_tokens[i].tolist() + target_audio_tokens)
    input_ids = [[[AUDIO_MASK_ID for _ in range(MAX_LEN)] for _ in range(NUM_AUDIO_CODEBOOK)], [[AUDIO_MASK_ID for _ in range(MAX_LEN)] for _ in range(NUM_AUDIO_CODEBOOK)]]

    for i in range(NUM_AUDIO_CODEBOOK): input_ids[0][i][:c_len] = cond_input_ids[0][i][:c_len]
    for i in range(NUM_AUDIO_CODEBOOK): input_ids[1][i][:target_length] = cond_input_ids[0][i][-target_length:]

    cond_audio_mask = ([False] * cond_audio_start_idx + [True] * (c_len - cond_audio_start_idx))
    audio_mask = [[False for _ in range(MAX_LEN)] for _ in range(2)]
    audio_mask[0][:c_len] = cond_audio_mask
    audio_mask[1][:target_length] = cond_audio_mask[-target_length:]

    attention_mask = [[[[False] * MAX_LEN for _ in range(MAX_LEN)]] for _ in range(2)]
    for i in range(c_len): attention_mask[0][0][i][:c_len] = [True] * c_len
    for i in range(target_length): attention_mask[1][0][i][:target_length] = [True] * target_length
    for i in range(target_length, c_len): attention_mask[1][0][i][i] = True

    return c_len, audio_mask, attention_mask, input_ids


  def _generate_iterative(self, text_tokens, target_length, ref_audio_tokens, num_steps=16, style_tokens=None):
    c_len, audio_mask, attention_mask, input_ids = self.get_inputs(text_tokens, target_length, ref_audio_tokens, style_tokens)
    sched, num_steps = get_sched(num_steps=num_steps, target_length=target_length)

    c_len_var = Variable("c_len",1,MAX_LEN).bind(c_len)
    t_len_var = Variable("t_len",1,MAX_LEN).bind(target_length)

    audio_mask = Tensor(audio_mask)
    attention_mask = Tensor(attention_mask)
    input_ids = Tensor(input_ids)

    #print("audio_mask", audio_mask._buffer()._buf.num)
    #print("attention_mask", attention_mask._buffer()._buf.num)
    #print("input_ids", input_ids._buffer()._buf.num)

    tokens = Tensor([[AUDIO_MASK_ID for _ in range(MAX_LEN)] for _ in range(NUM_AUDIO_CODEBOOK)])
    for step in range(num_steps):
      print("STEP",step,"of",num_steps)
      pred_tokens, scores = self(input_ids=input_ids[:, :, :c_len_var], audio_mask=audio_mask[:, :c_len_var], 
                          attention_mask=attention_mask[:, :, :c_len_var, :c_len_var], tokens=tokens,
                          c_len_var=c_len_var, t_len_var=t_len_var)
      #print("tokens", tokens._buffer()._buf.num)
      #print("pred_tokens", pred_tokens._buffer()._buf.num)
      scores = scores.numpy()
      pred_tokens = pred_tokens.numpy()
      
      input_ids = input_ids.numpy()
      pred_tokens = pred_tokens[:, :, :target_length]
      scores = scores[:, :target_length]

      sorted_idx = np.argsort(scores.flatten())[::-1]
      topk_idx = sorted_idx[:sched[step]]
      tokens = tokens.numpy()
      sample_tokens = tokens[:, :target_length]
      sample_tokens = sample_tokens.flatten()
      sample_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
      sample_tokens = sample_tokens.astype(int)
      sample_tokens = sample_tokens.reshape(NUM_AUDIO_CODEBOOK, target_length)

      tokens[:, :target_length] = sample_tokens
      input_ids[0: 1, :,  c_len-target_length:c_len] = sample_tokens
      input_ids[1: 2, :, :target_length] = sample_tokens

      tokens = Tensor(tokens)
      input_ids = Tensor(input_ids)

    return tokens.numpy().tolist()

def get_sched(num_steps, target_length):
  timesteps = [i / num_steps for i in range(num_steps + 1)]
  timesteps = [(T_SHIFT * t) / (1 + (T_SHIFT - 1) * t) for t in timesteps]
  total_mask = target_length * NUM_AUDIO_CODEBOOK
  rem = total_mask
  sched = []
  for step in range(num_steps):
    num = (rem if step == num_steps - 1 else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem,))
    sched.append(int(num))
    if num >= MAX_LEN: return get_sched(num_steps=num_steps*2, target_length=target_length)
    rem -= int(num)
  return sched, num_steps

import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/":
      with open("main.html", "rb") as f: content = f.read()
      self.send_response(200)
      self.send_header("Content-Type", "text/html")
      self.send_header("Content-Length", str(len(content)))
      self.end_headers()
      self.wfile.write(content)

    elif self.path == "/languages.json":
      with open("languages.json", "rb") as f:
          content = f.read()
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(content)))
      self.end_headers()
      self.wfile.write(content)
    elif self.path == "/voices":
      voices = glob.glob("voices/*.cv")
      voice_names = [os.path.splitext(os.path.basename(v))[0] for v in voices]
      content = json.dumps(voice_names).encode('utf-8')
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(content)))
      self.end_headers()
      self.wfile.write(content)

    else:
      self.send_response(404)
      self.end_headers()
  
  def do_POST(self):
    #try:
    if 1==1:
      content_type = self.headers.get('Content-Type')
      boundary = content_type.split('boundary=')[1].encode().strip(b'"')
      body = self.rfile.read(int(self.headers['Content-Length']))
      parts = body.split(b'--' + boundary)[1:-1]
      data = {}
      for part in parts:
        content = part.split(b'\r\n\r\n', 1)[1]
        content = content.rsplit(b'\r\n', 1)[0]
        if b'name="file"' in part:
          data['file'] = content
        elif b'name="ref_text"' in part:
          data['ref_text'] = content.decode()
        elif b'name="voice_name"' in part:
          data['voice_name'] = content.decode()
        elif b'name="selected_voice"' in part:
          selected_voice = content.decode()
        elif b'name="target_text"' in part:
          data['target_text'] = content.decode()
        elif b'name="language"' in part:
          data['language'] = content.decode()
      
      if "ref_text" in data and len(data["ref_text"]) > 0:
        voice = { "ref_text": data['ref_text'], "ref_audio": base64.b64encode(data['file']).decode("ascii")}
        with open(f"voices/{data['voice_name']}.cv", "w") as f: json.dump(voice, f)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "success"}).encode())
        return

      data['voice_name'] = selected_voice

      audio = model.generate(
        text=data['target_text'],
        cv_path=f"voices/{data['voice_name']}.cv",
        num_steps=32,
        language=data["language"]
      )
      wav_bytes = waveform_to_wav_bytes(audio, SAMPLING_RATE)
      self.send_response(200)
      self.send_header("Content-Type", "audio/wav")
      self.send_header("Content-Length", str(len(wav_bytes)))
      self.end_headers()
      self.wfile.write(wav_bytes)

    #except Exception as e:
    #  print(f"Error: {e}")
    #  self.send_response(500)
    #  self.end_headers()

if __name__ == "__main__":
  import os
  import glob
  import base64
  model = omni()
  
  if "--test" in sys.argv:
    '''
    with open("voices/rory-15s.wav", "rb") as f:
      voice = {"ref_text":"Yeah so I was just on the thirty three there, on my way to astro, and like I'm just reading my book and looking out the window, and I look, and there's a dog getting on the bus, and the thing has a leap card in its mouth, and it jumps up and taps the machine",
        "ref_audio":base64.b64encode(f.read()).decode("ascii")} 
      json.dump(voice, open("voices/Rory.cv", "w"))
    exit()
    '''
    
    os.makedirs("outputs", exist_ok=True)
    Tensor.manual_seed(0)

    audio = model.generate(
        text="That's it, turn the page on the day, walk away ,'Cause there's sense in what I say, I'm forty-fifth generation roman but I don't know them or care when I'm spitting, So return to your sitting position and listen",
        cv_path="voices/Rory.cv",
        num_steps=32,
        language="None"
    )
    with open("outputs/output.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))
  else:
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    server.model = model
    print("Serving on http://localhost:8080")
    server.serve_forever()
