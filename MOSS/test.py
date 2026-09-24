import torch
from transformers import AutoProcessor # remove first?
from typing import Any, Callable
import numpy as np
import copy
from pathlib import Path

from transformers.audio_utils import load_audio
from transformers.utils import ModelOutput
from transformers.cache_utils import Cache
from transformers.models.auto.auto_factory import _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict
import os
from transformers import GenerationMixin, PreTrainedModel
from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.whisper.configuration_whisper import WhisperConfig

class MossTranscribeDiarizeConfig(PretrainedConfig):
    """Configuration for MOSS-Transcribe-Diarize: Qwen3 text backbone + Whisper audio encoder."""

    model_type = "moss_transcribe_diarize"
    sub_configs = {"text_config": Qwen3Config, "audio_config": WhisperConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_token_id: int = 151671,
        audio_merge_size: int = 4,
        adaptor_input_dim: int | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Qwen3Config(
                vocab_size=151936,
                hidden_size=1024,
                intermediate_size=3072,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=40960,
                tie_word_embeddings=tie_word_embeddings,
                rope_theta=1_000_000.0,
                layer_types=["full_attention"] * 28,
            )
        elif isinstance(text_config, dict):
            text_config = self.sub_configs["text_config"](**text_config)

        if audio_config is None:
            audio_config = WhisperConfig(
                num_mel_bins=80,
                d_model=1024,
                encoder_layers=24,
                encoder_attention_heads=16,
                encoder_ffn_dim=4096,
                max_source_positions=1500,
                dropout=0.0,
                attention_dropout=0.0,
                activation_dropout=0.0,
                activation_function="gelu",
                encoder_layerdrop=0.0,
                scale_embedding=False,
            )
        elif isinstance(audio_config, dict):
            audio_config = self.sub_configs["audio_config"](**audio_config)

        text_config.tie_word_embeddings = tie_word_embeddings

        self.text_config = text_config
        self.audio_config = audio_config
        self.audio_token_id = audio_token_id
        self.audio_merge_size = audio_merge_size
        self.adaptor_input_dim = adaptor_input_dim or audio_config.d_model * audio_merge_size
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class MossTranscribeDiarizePreTrainedModel(PreTrainedModel):
    config_class = MossTranscribeDiarizeConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    _no_split_modules = ["Qwen3DecoderLayer", "WhisperEncoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_attention_backend = True

from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from torch import nn
from transformers.utils import torch_compilable_check


class VQAdaptor(nn.Module):
    """Projects merged Whisper features to LM hidden dim.

    ``Linear(in → hidden) → SiLU → Linear(hidden → hidden) → LayerNorm``
    """

    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=norm_eps, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layers(x)

class MossTranscribeDiarizeModel(MossTranscribeDiarizePreTrainedModel):
    base_model_prefix = "model"

    """Single-stream multimodal backbone: Whisper-Medium encoder + Qwen3-0.6B.

    Audio features are injected into text embeddings via ``masked_scatter`` at
    positions marked by ``audio_token_id`` in ``input_ids``.
    """

    def __init__(self, config: MossTranscribeDiarizeConfig):
        super().__init__(config)

        self.language_model: nn.Module = Qwen3Model(config.text_config)
        self.whisper_encoder: nn.Module = WhisperEncoder(config.audio_config)
        self.vq_adaptor: VQAdaptor = VQAdaptor(
            input_dim=config.adaptor_input_dim,
            hidden_size=config.text_config.hidden_size,
            norm_eps=config.text_config.rms_norm_eps,
        )
        self.post_init()

    # ---- 4x time merge ---------------------------------------------------

    def time_merge(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T//M, D*M)`` where M is ``audio_merge_size``."""
        B, T, D = features.shape
        merge_size = int(self.config.audio_merge_size)
        T_trim = (T // merge_size) * merge_size
        return features[:, :T_trim, :].reshape(B, T_trim // merge_size, D * merge_size)

    # ---- audio feature extraction -----------------------------------------

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.LongTensor,
        audio_chunk_mapping: Optional[torch.LongTensor] = None,
    ) -> list[torch.Tensor]:
        device = next(self.whisper_encoder.parameters()).device
        encoder_dtype = next(self.whisper_encoder.parameters()).dtype
        input_features = input_features.to(device=device, dtype=encoder_dtype)
        audio_feature_lengths = audio_feature_lengths.to(device=device)

        whisper_features = self.whisper_encoder(input_features, return_dict=True).last_hidden_state

        chunk_mapping = (
            audio_chunk_mapping.to(device=device)
            if audio_chunk_mapping is not None
            else torch.zeros(input_features.shape[0], dtype=torch.long, device=device)
        )

        num_audios = int(chunk_mapping.max().item()) + 1 if chunk_mapping.numel() else 0
        per_audio_chunks = [[] for _ in range(num_audios)]
        for chunk_idx, token_len in enumerate(audio_feature_lengths.tolist()):
            sample_idx = int(chunk_mapping[chunk_idx].item())
            per_audio_chunks[sample_idx].append(
                whisper_features[chunk_idx : chunk_idx + 1, : int(token_len) * 4]
            )

        adapted = []
        for parts in per_audio_chunks:
            feat = torch.cat(parts, dim=1)
            feat = feat.to(self.dtype)
            merged = self.time_merge(feat)
            adapted.append(self.vq_adaptor(merged))
        return adapted

    # ---- inject audio into text embeddings --------------------------------

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.FloatTensor,
        audio_features: torch.Tensor,
    ) -> torch.BoolTensor:
        special_audio_mask = input_ids.to(device=inputs_embeds.device) == self.config.audio_token_id
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_audio_mask

    def inject_audio_features(
        self,
        input_ids,
        inputs_embeds,
        input_features,
        audio_feature_lengths,
        audio_chunk_mapping,
    ):
        """Replace audio placeholder positions with projected audio features."""
        if input_features is None:
            return inputs_embeds
        audio_features = self.get_audio_features(
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )
        audio_embeds = torch.cat([f.squeeze(0) for f in audio_features], dim=0)
        audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds, audio_embeds)
        return inputs_embeds.masked_scatter(audio_mask, audio_embeds)

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        audio_chunk_mapping: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        inputs_embeds = self.inject_audio_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )
        outputs = self.language_model(
            input_ids=None, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, **kwargs,
        )
        return outputs


class MossTranscribeDiarizeForConditionalGeneration(MossTranscribeDiarizePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: MossTranscribeDiarizeConfig):
        super().__init__(config)
        self.model = MossTranscribeDiarizeModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        result = super().tie_weights(*args, **kwargs)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
        return result

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        audio_chunk_mapping: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return CausalLMOutputWithPast(
            loss=None, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        input_features=None,
        audio_feature_lengths=None,
        audio_chunk_mapping=None,
        is_first_iteration=False,
        use_cache=True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration, use_cache=use_cache, **kwargs,
        )
        if input_features is not None and (is_first_iteration or not use_cache):
            model_inputs["input_features"] = input_features
            model_inputs["audio_feature_lengths"] = audio_feature_lengths
            model_inputs["audio_chunk_mapping"] = audio_chunk_mapping
        return model_inputs

class _BaseAutoModelClass:
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict([]))
    def __init__(self, *args, **kwargs) -> None: pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], *model_args, **kwargs):
        kwargs["_from_auto"] = True
        adapter_kwargs = None

        kwargs["adapter_kwargs"] = adapter_kwargs
        
        return MossTranscribeDiarizeForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *model_args, config=None, **kwargs)


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None

# todo just use needed entry...

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
TokenCallback = Callable[[int], None]
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv"}

@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str

def _is_timestamp_char(ch: str) -> bool: return ("0" <= ch <= "9") or ch == "."

def _is_speaker_char(ch: str) -> bool: return ch == "S" or ("0" <= ch <= "9")

def _parse_timestamp(chars: list[str]) -> float | None:
    if not chars:
        return None

    dot_count = 0
    digit_count = 0
    for ch in chars:
        if "0" <= ch <= "9":
            digit_count += 1
        elif ch == ".":
            dot_count += 1
            if dot_count > 1:
                return None
        else:
            return None

    if digit_count == 0:
        return None
    return float("".join(chars))

def _parse_speaker(chars: list[str]) -> str | None:
    if len(chars) < 2 or chars[0] != "S":
        return None
    for ch in chars[1:]:
        if not ("0" <= ch <= "9"):
            return None
    return "".join(chars)

class TranscriptStreamParser:
    """Streaming parser for compact MOSS transcript output.

    Expected segment format:

        [start][Sxx]text[end]

    The parser deliberately avoids regular expressions. It scans characters
    once, keeps only the active token/text buffers, and emits a segment after
    an end timestamp is confirmed by the next segment start or by ``close()``.
    """

    _SEEK_START = 0
    _READ_START = 1
    _EXPECT_SPEAKER_OPEN = 2
    _READ_SPEAKER = 3
    _READ_TEXT = 4
    _READ_END = 5
    _AFTER_END = 6

    def __init__(self, *, strip_text: bool = True, skip_empty: bool = True):
        self.strip_text = strip_text
        self.skip_empty = skip_empty
        self._state = self._SEEK_START
        self._token: list[str] = []
        self._text: list[str] = []
        self._pending_after_end: list[str] = []
        self._start: float | None = None
        self._end: float | None = None
        self._end_token = ""
        self._speaker: str | None = None

    def reset(self) -> None:
        self._state = self._SEEK_START
        self._token.clear()
        self._text.clear()
        self._pending_after_end.clear()
        self._start = None
        self._end = None
        self._end_token = ""
        self._speaker = None

    def feed(self, chunk: str) -> list[TranscriptSegment]:
        """Consume a text chunk and return any newly completed segments."""
        segments: list[TranscriptSegment] = []
        self.feed_into(chunk, segments.append)
        return segments

    def feed_into(self, chunk: str, emit: Callable[[TranscriptSegment], None]) -> None:
        """Consume a text chunk and send completed segments to ``emit``."""
        if not isinstance(chunk, str):
            raise TranscriptParseError(f"chunk must be str, got {type(chunk).__name__}")

        for ch in chunk:
            state = self._state
            if state == self._SEEK_START:
                self._seek_start(ch)
            elif state == self._READ_START:
                self._read_start(ch)
            elif state == self._EXPECT_SPEAKER_OPEN:
                self._expect_speaker_open(ch)
            elif state == self._READ_SPEAKER:
                self._read_speaker(ch)
            elif state == self._READ_TEXT:
                self._read_text(ch)
            elif state == self._READ_END:
                self._read_end(ch, emit)
            elif state == self._AFTER_END:
                self._after_end(ch, emit)

    def close(self) -> list[TranscriptSegment]:
        """Finish the stream and return a final segment if one is complete."""
        segments: list[TranscriptSegment] = []
        self.close_into(segments.append)
        return segments

    def close_into(self, emit: Callable[[TranscriptSegment], None]) -> None:
        """Finish the stream and send a final complete segment to ``emit``."""
        if self._state == self._AFTER_END:
            self._emit_segment(emit)
        self.reset()

    def _seek_start(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_START

    def _read_start(self, ch: str) -> None:
        if ch == "]":
            start = _parse_timestamp(self._token)
            if start is None:
                self.reset()
                return
            self._start = start
            self._state = self._EXPECT_SPEAKER_OPEN
            self._token.clear()
            return

        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return

        self.reset()
        if ch == "[":
            self._state = self._READ_START

    def _expect_speaker_open(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_SPEAKER
        elif not ch.isspace():
            self.reset()

    def _read_speaker(self, ch: str) -> None:
        if ch == "]":
            speaker = _parse_speaker(self._token)
            if speaker is None:
                self.reset()
                return
            self._speaker = speaker
            self._text.clear()
            self._state = self._READ_TEXT
            self._token.clear()
            return

        if _is_speaker_char(ch):
            self._token.append(ch)
            if len(self._token) <= 16:
                return

        self.reset()
        if ch == "[":
            self._state = self._READ_START

    def _read_text(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_END
        else:
            self._text.append(ch)

    def _read_end(self, ch: str, emit: Callable[[TranscriptSegment], None]) -> None:
        if ch == "]":
            end = _parse_timestamp(self._token)
            if end is not None and self._start is not None and end >= self._start:
                self._end = end
                self._end_token = "".join(self._token)
                self._pending_after_end.clear()
                self._state = self._AFTER_END
            else:
                self._text.append("[")
                self._text.extend(self._token)
                self._text.append("]")
                self._state = self._READ_TEXT
            self._token.clear()
            return

        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return

        self._text.append("[")
        self._text.extend(self._token)
        self._text.append(ch)
        self._token.clear()
        self._state = self._READ_TEXT

    def _after_end(self, ch: str, emit: Callable[[TranscriptSegment], None]) -> None:
        if ch == "[":
            self._emit_segment(emit)
            self._token.clear()
            self._state = self._READ_START
            return

        if ch.isspace():
            self._pending_after_end.append(ch)
            return

        self._text.append("[")
        self._text.append(self._end_token)
        self._text.append("]")
        self._text.extend(self._pending_after_end)
        self._text.append(ch)
        self._pending_after_end.clear()
        self._end = None
        self._end_token = ""
        self._state = self._READ_TEXT

    def _emit_segment(self, emit: Callable[[TranscriptSegment], None]) -> None:
        if self._start is None or self._end is None or self._speaker is None:
            self.reset()
            return

        text = "".join(self._text)
        if self.strip_text:
            text = text.strip()
        if text or not self.skip_empty:
            emit(
                TranscriptSegment(
                    start=self._start,
                    end=self._end,
                    speaker=self._speaker,
                    text=text,
                )
            )

        self._token.clear()
        self._text.clear()
        self._pending_after_end.clear()
        self._start = None
        self._end = None
        self._end_token = ""
        self._speaker = None
        self._state = self._SEEK_START

def build_transcription_messages(audio_path: str | Path, prompt: str = DEFAULT_PROMPT) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
            ],
        }
    ]

def _token_count(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_token_count(item) for item in value)
    return 1

class ProgressStreamer:
    """Count generated tokens from ``generate(streamer=...)`` without decoding text."""

    def __init__(self, callback: TokenCallback):
        self.callback = callback
        self.generated_tokens = 0
        self._seen_prompt = False

    def put(self, value):
        token_count = _token_count(value)
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        self.generated_tokens += token_count
        self.callback(self.generated_tokens)

    def end(self):
        return None

def load_audio_av(audio: str, sampling_rate: int) -> np.ndarray:
    """Decode an audio stream from a media container with PyAV."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("Install `av` to decode audio from video containers.") from exc

    chunks: list[np.ndarray] = []
    with av.open(audio) as container:
        stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {audio!r}.")

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
        for frame in container.decode(stream):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                chunks.append(resampled.to_ndarray().reshape(-1))

        frames = resampler.resample(None)
        if frames is not None:
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError(f"No decodable audio samples found in {audio!r}.")
    return (np.concatenate(chunks).astype(np.float32) / 32768.0).astype(np.float32, copy=False)

def load_audio_item(audio: str | np.ndarray, sampling_rate: int) -> np.ndarray:
    return load_audio(audio, sampling_rate=sampling_rate)
    
def process_audio_info(messages: list[dict[str, Any]], sampling_rate: int):
    """Load audio items from chat messages in the same order as the template."""
    audios = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for item in content:
            if item.get("type") != "audio":
                continue
            audio = item.get("audio") or item.get("audio_url") or item.get("url") or item.get("path")
            if audio is None:
                raise ValueError("Audio content must include audio, audio_url, url, or path.")
            audios.append(load_audio_item(audio, sampling_rate=sampling_rate))
    return audios

def prepare_inputs(processor, messages, *, max_length: int = 131072, device: torch.device | None = None):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)
    audio_kwargs = {"device": str(device)} if device is not None and device.type == "cuda" else {}
    return processor(
        text=text,
        audio=audios,
        max_length=max_length,
        audio_kwargs=audio_kwargs,
        return_tensors="pt",
    )

def generate_transcription(
    model,
    processor,
    messages,
    *,
    max_length: int = 131072,
    max_new_tokens: int | None = None,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    input_callback: Callable[[int], None] | None = None,
    token_callback: TokenCallback | None = None,
) -> dict[str, Any]:
    device = device or next(model.parameters()).device
    dtype = dtype or next(model.parameters()).dtype
    context = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else torch.no_grad()
    )
    with context:
        inputs = prepare_inputs(processor, messages, max_length=max_length, device=device).to(device)

    prompt_len = int(inputs["attention_mask"][0].sum().item())
    if input_callback is not None:
        input_callback(prompt_len)
    generation_config = copy.deepcopy(model.generation_config)
    if max_new_tokens is not None:
        generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample
    if do_sample and temperature is not None:
        generation_config.temperature = temperature
    if do_sample and top_p is not None:
        generation_config.top_p = top_p
    if do_sample and top_k is not None:
        generation_config.top_k = top_k
    streamer = ProgressStreamer(token_callback) if token_callback is not None else None
    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "input_features": inputs["input_features"],
        "audio_feature_lengths": inputs["audio_feature_lengths"],
        "audio_chunk_mapping": inputs["audio_chunk_mapping"],
        "generation_config": generation_config,
    }
    if streamer is not None:
        generate_kwargs["streamer"] = streamer

    with torch.inference_mode(), (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else torch.no_grad()
    ):
        try:
            outputs = model.generate(**generate_kwargs)
        except TypeError as exc:
            if streamer is None or "streamer" not in str(exc):
                raise
            generate_kwargs.pop("streamer", None)
            outputs = model.generate(**generate_kwargs)

    generated_ids = outputs[0][prompt_len:]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prompt_len": prompt_len,
        "generated_tokens": int(generated_ids.numel()),
    }

def parse_transcript(text: str, **parser_kwargs) -> list[TranscriptSegment]:
    parser = TranscriptStreamParser(**parser_kwargs)
    segments = parser.feed(text)
    segments.extend(parser.close())
    return segments

device = torch.device("cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
model = _BaseAutoModelClass.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize").to(dtype=dtype).to(device).eval()

model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_path = "MOSS/output.wav" # 10 mins for now

processor = AutoProcessor.from_pretrained(model_id)

messages = build_transcription_messages(audio_path)
result = generate_transcription(
    model,
    processor,
    messages,
    max_new_tokens=2048, # was 2048, this should reduce ram use?
    do_sample=False,
    device=device,
    dtype=dtype,
)

print("rory unparsed result =",result,"\n\n\n")

excepted = "[9.26][S01]哈喽大家好，我是小俊[10.73][11.02][S01]今天我们的嘉宾是Google DeepMind研究员姚舜宇[14.31][14.63][S01]硅谷有两个很有名的姚舜宇，一个之前在OpenAI跳槽去了腾讯出任腾讯的首席科学家[21.51][21.88][S01]他之前也来过我们节目[23.31][23.62][S01]那今天我邀请的是另一位姚舜宇，他此前在Anthropic，现在在Google DeepMind[29.55][29.88][S01]我们从近期一系列的模型巨变开始聊起[33.75][34.29][S01]那接下来就是我对舜宇的访谈[36.98][37.06][S02]Anthropic作为一个公司来说，它能够实行这种就是比较top down的机制，是一个很独特的事[43.27][43.61][S01]这对于其他模型公司很难吗[45.01][45.38][S02]很难，比如说OpenAI就干不了，就是Gemini也比较难，大公司和和startup它打法本来就不一样，因为startup重要的是make bet[56.47][56.84][S02]就是我得我得赌一件事[58.67][59.15][S02]是我觉得大家现在就是每个人都是冲浪的人，本质上是那个浪，而不是你那个冲浪的人，因为AI这个事本来也不太需要脑子[67.69][68.94][S01]不太需要脑子[69.91][69.74][S02]真的不太需要脑子[70.79][70.81][S01]需要什么[71.42][71.93][S02]我觉得这个这个行业就是最重要的特质就是靠谱，就是做事细，然后对自己做的事负责任，这是最重要的特质[81.92][87.89][S01]硅谷不是有两个姚舜宇吗？你要不要先给大家介绍一下你自己，然后给大家科普一下两个姚舜宇的区别[94.52][94.94][S02]啊可以，对，就是呃我叫姚舜宇，然后显然也有一个跟我呃几乎同名的朋友，然后呃我们俩主要履历也有一些overlap，所以说可能看起来非常的难以区分。对，然后[109.31][109.78][S02]呃我是我以前是做呃学物理的，然后我本科的时候在呃清华啊那时候做宁态理论，然后后来去斯坦福啊做呃理论高能物理，然后和量子信息黑洞相关的一些方面。[124.11][124.49][S02]然后呃离开斯坦福之后，去呃伯克利短暂的待了两个星期的postdoc过后，然后就离离职了，去了Anthropic，然后在Anthropic待了一年，[137.01][137.38][S02]啊去年九月底十月初的时候呃加入了Gemini。[141.01][141.47][S02]对，然后呃如果大家非要区分的话，我觉得最大的区分就是那个舜宇他一开始就是一直都是做CS，就是计算机相关的。然后我其实呃从某种意义上来说是个半道出家。对，就是我之前是做理论物理为主的。对。[157.41][157.88][S01]你们是不是好朋友？[159.11][159.25][S01]你们好像大学就认识，而且是一级的对吧？[161.58][161.75][S02]对[161.95][161.92][S01]他是一个什么样的人？你是一个什么样的人？你评价一下他，你也评价一下自己。[165.51][166.52][S02]对对对，我们本科就认识，因为我们本科是一级的，然后在清华，但他一开始就是学计算机的嘛，所以他在那个姚班就是计算机科学实验班，然后呃我是学物理，所以我在机科班。[175.69][175.98][S02]对，然后呃后来他去了普林，我去斯坦福，然后这可能也是另一个有点令人费解的点，就是好像这个普世世界里觉得斯坦福应该是学计算机的人该去的地方，然后觉得普林斯顿是学物理人该去的地方，但我俩然后反过来，[191.61][192.38][S02]所以说也可能产生了一些费解的事情。[194.89][194.89][S02]对，然后我俩其实也还真的挺不一样，我觉得他是一个比我有趣的多的人。[200.01][200.45][S02]我觉得我我从他身上也是在过去也是能学习到了一些和我很不一样的点。比如说他可能花了很多时间去思考，比如在AI方面，他花了很多时间去思考，就是人和AI的交互呀，然后包括一些产品上的事情。然后我觉得其实对我来说呃是一个很不一样的朋友，然后我也从他那学到了很多东西。[220.94][221.68][S01]你们之前在硅谷的时候多久见一次面？[223.51][223.51][S01]你们现在是不是还频繁打电话多频繁？[225.67][226.23][S02]呃，我们在硅谷的时候见面确实挺频繁的，可能每每几个星期吧，但是好像见面主要是为了凑一块玩。[236.34][236.98][S01]玩啥[237.45][238.04][S02]就是真的就是纯玩，就是可能出去散散步，扯扯有的没的，然后可能有时候吃个饭打个牌啊之类的。[247.11][247.11][S02]对，[247.38][247.38][S02]对，然后他回去之后，其实我们也也是也还是经常会打电话。[251.91][252.32][S01]最近一次电话聊啥了？好像就是前一两个星期[254.95][255.34][S02]啊你怎么知道的？呃，可能就是会过几个月，然后然后就catchup一下呃大家最近的近况吧。[264.24][264.24][S02]对。[264.55][265.01][S01]他是不是多次想把你拉过去？[266.71][267.24][S02]啊[268.55][270.93][S02]可能有这个意思吧，但是但是我觉得不关键不关键。[274.51][275.49][S01]你为什么不去？[276.12][276.12][S02]我觉得对我自己来说，我呃没想清楚吧。嗯，我觉得呃多半是我自己的原因，然后呃我也没有去任何[285.11][285.98][S02]呃中国的地方，然后我觉得主要原因是因为呃在去年的九月或者八九月这个时候，我觉得呃那时候我离开离开Anthropic，然后离开之后决定要去哪的时候，[298.99][298.99][S02]最大的动机是呃我想学一些不一样的东西。[302.51][303.21][S02]呃，对我来说我可能就没有去考虑，[306.01][306.01][S02]呃，没有没有更着重的去考虑说能够我去领导一个项目，或者领导一个project之类的。我更多的是是那个时候更多的是优先去学习一些东西，所以那时候选择去了Gemini。[318.01][319.57][S01]我发现你们两个老被放在一起比较和讨论，对你来说是困扰更多还是享受更多？[324.06][324.29][S02]啊，我没什么感觉，然后因因为我这个人也不太关注社交媒体，所以我其实真的没什么感觉。[333.61][334.65][S01]嗯[334.92][335.80][S01]因为那个舜宇他之前在去年的时候说AI进入了the second half，进入下半场，这个成为了一个非常有名的观点。你觉得今天的AI在一个什么样的时期？你能给它一个定义吗？[347.41"
assert result["text"] == excepted

changes = []
speakers = []
parsed =  parse_transcript(result["text"])
# get changes
speaker = -1 # init
for i in range(len(parsed)):
  if parsed[i].speaker != speaker:
    speaker = parsed[i].speaker
    changes.append(parsed[i].start)
    speakers.append(int(parsed[i].speaker.replace("S","")))

for segment in parsed:
    print(segment.start, segment.end, segment.speaker, segment.text)

print("rory changes =", changes)
print("rory speakers =",speakers)
