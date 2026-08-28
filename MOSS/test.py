import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Any, Callable
import numpy as np
import copy
from pathlib import Path
from transformers.generation.streamers import BaseStreamer

from transformers.audio_utils import load_audio
from dataclasses import dataclass

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


model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_path = "output.wav"

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

print(result["text"])

def parse_transcript(text: str, **parser_kwargs) -> list[TranscriptSegment]:
    parser = TranscriptStreamParser(**parser_kwargs)
    segments = parser.feed(text)
    segments.extend(parser.close())
    return segments


for segment in parse_transcript(result["text"]):
    print(segment.start, segment.end, segment.speaker, segment.text)
