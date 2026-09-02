import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Any, Callable
import numpy as np
import copy
from pathlib import Path

from transformers.audio_utils import load_audio
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from dataclasses import dataclass
from collections import OrderedDict
import os

# todo just use needed entry...
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("afmoe", "AfmoeForCausalLM"),
        ("apertus", "ApertusForCausalLM"),
        ("arcee", "ArceeForCausalLM"),
        ("aria_text", "AriaTextForCausalLM"),
        ("axk1", "AXK1ForCausalLM"),
        ("axk2", "AXK2ForCausalLM"),
        ("bamba", "BambaForCausalLM"),
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("big_bird", "BigBirdForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("bitnet", "BitNetForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("blt", "BltForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cohere", "CohereForCausalLM"),
        ("cohere2", "Cohere2ForCausalLM"),
        ("cohere2_moe", "Cohere2MoeForCausalLM"),
        ("cohere_compass_text", "CohereCompassForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("cwm", "CwmForCausalLM"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("dbrx", "DbrxForCausalLM"),
        ("deepseek_v2", "DeepseekV2ForCausalLM"),
        ("deepseek_v3", "DeepseekV3ForCausalLM"),
        ("deepseek_v32", "DeepseekV32ForCausalLM"),
        ("deepseek_v4", "DeepseekV4ForCausalLM"),
        ("diffllama", "DiffLlamaForCausalLM"),
        ("doge", "DogeForCausalLM"),
        ("dots1", "Dots1ForCausalLM"),
        ("electra", "ElectraForCausalLM"),
        ("emu3", "Emu3ForCausalLM"),
        ("ernie", "ErnieForCausalLM"),
        ("ernie4_5", "Ernie4_5ForCausalLM"),
        ("ernie4_5_moe", "Ernie4_5_MoeForCausalLM"),
        ("exaone4", "Exaone4ForCausalLM"),
        ("exaone_moe", "ExaoneMoeForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("falcon_h1", "FalconH1ForCausalLM"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("flex_olmo", "FlexOlmoForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("gemma3n", "Gemma3nForConditionalGeneration"),
        ("gemma3n_text", "Gemma3nForCausalLM"),
        ("gemma4", "Gemma4ForConditionalGeneration"),
        ("gemma4_assistant", "Gemma4AssistantForCausalLM"),
        ("gemma4_text", "Gemma4ForCausalLM"),
        ("gemma4_unified", "Gemma4UnifiedForConditionalGeneration"),
        ("gemma4_unified_assistant", "Gemma4UnifiedAssistantForCausalLM"),
        ("gemma4_unified_text", "Gemma4UnifiedForCausalLM"),
        ("git", "GitForCausalLM"),
        ("glm", "GlmForCausalLM"),
        ("glm4", "Glm4ForCausalLM"),
        ("glm4_moe", "Glm4MoeForCausalLM"),
        ("glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
        ("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
        ("got_ocr2", "GotOcr2ForConditionalGeneration"),
        ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gpt_oss", "GptOssForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("granite", "GraniteForCausalLM"),
        ("granite_swa", "GraniteSWAForCausalLM"),
        ("granitemoe", "GraniteMoeForCausalLM"),
        ("granitemoe_swa", "GraniteMoeSWAForCausalLM"),
        ("granitemoehybrid", "GraniteMoeHybridForCausalLM"),
        ("granitemoeshared", "GraniteMoeSharedForCausalLM"),
        ("helium", "HeliumForCausalLM"),
        ("hrm_text", "HrmTextForCausalLM"),
        ("hunyuan_v1_dense", "HunYuanDenseV1ForCausalLM"),
        ("hunyuan_v1_moe", "HunYuanMoEV1ForCausalLM"),
        ("hy_v3", "HYV3ForCausalLM"),
        ("hyperclovax", "HyperCLOVAXForCausalLM"),
        ("inkling_text", "InklingForCausalLM"),
        ("jais2", "Jais2ForCausalLM"),
        ("jamba", "JambaForCausalLM"),
        ("jetmoe", "JetMoeForCausalLM"),
        ("laguna", "LagunaForCausalLM"),
        ("lfm2", "Lfm2ForCausalLM"),
        ("lfm2_moe", "Lfm2MoeForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("llama4", "Llama4ForCausalLM"),
        ("llama4_text", "Llama4ForCausalLM"),
        ("longcat_flash", "LongcatFlashForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mellum", "MellumForCausalLM"),
        ("mimo_v2_flash", "MiMoV2FlashForCausalLM"),
        ("minicpm3", "MiniCPM3ForCausalLM"),
        ("minimax", "MiniMaxForCausalLM"),
        ("minimax_m2", "MiniMaxM2ForCausalLM"),
        ("minimax_m3_vl_text", "MiniMaxM3VLForCausalLM"),
        ("ministral", "MinistralForCausalLM"),
        ("ministral3", "Ministral3ForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("mllama", "MllamaForCausalLM"),
        ("modernbert-decoder", "ModernBertDecoderForCausalLM"),
        ("moshi", "MoshiForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("musicgen", "MusicgenForCausalLM"),
        ("musicgen_melody", "MusicgenMelodyForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("nanochat", "NanoChatForCausalLM"),
        ("nemotron", "NemotronForCausalLM"),
        ("nemotron_h", "NemotronHForCausalLM"),
        ("olmo", "OlmoForCausalLM"),
        ("olmo2", "Olmo2ForCausalLM"),
        ("olmo3", "Olmo3ForCausalLM"),
        ("olmo_hybrid", "OlmoHybridForCausalLM"),
        ("olmoe", "OlmoeForCausalLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("phi4_multimodal", "Phi4MultimodalForCausalLM"),
        ("phimoe", "PhimoeForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("qwen2_moe", "Qwen2MoeForCausalLM"),
        ("qwen3", "Qwen3ForCausalLM"),
        ("qwen3_5", "Qwen3_5ForCausalLM"),  # VLM compatibility
        ("qwen3_5_moe", "Qwen3_5MoeForCausalLM"),  # VLM compatibility
        ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
        ("qwen3_5_text", "Qwen3_5ForCausalLM"),
        ("qwen3_moe", "Qwen3MoeForCausalLM"),
        ("qwen3_next", "Qwen3NextForCausalLM"),
        ("qwen4_exp", "Qwen4ExpForCausalLM"),  # VLM compatibility
        ("qwen4_exp_text", "Qwen4ExpForCausalLM"),
        ("recurrent_gemma", "RecurrentGemmaForCausalLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("seed_oss", "SeedOssForCausalLM"),
        ("smollm3", "SmolLM3ForCausalLM"),
        ("solar_open", "SolarOpenForCausalLM"),
        ("stablelm", "StableLmForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("trocr", "TrOCRForCausalLM"),
        ("vaultgemma", "VaultGemmaForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xlstm", "xLSTMForCausalLM"),
        ("xmod", "XmodForCausalLM"),
        ("youtu", "YoutuForCausalLM"),
        ("zamba", "ZambaForCausalLM"),
        ("zamba2", "Zamba2ForCausalLM"),
        ("zaya", "ZayaForCausalLM"),
    ]
)

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

MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

    # override to give better return typehint
    @classmethod
    def from_pretrained(
        cls: type["AutoModelForCausalLM"],
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args,
        **kwargs,
    ) -> "_BaseModelWithGenerate":
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_path = "MOSS/output.wav" # 10 mins for now

device = torch.device("cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
).to(dtype=dtype).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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
