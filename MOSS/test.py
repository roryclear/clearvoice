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

def parse_transcript(text: str, **parser_kwargs) -> list[TranscriptSegment]:
    parser = TranscriptStreamParser(**parser_kwargs)
    segments = parser.feed(text)
    segments.extend(parser.close())
    return segments


model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_path = "MOSS/output.wav"

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

excepted = "[0.00][S01]并不代表用户体验这个模型的能力增长变慢。因为可能从百分之五十到百分之六十，他可能感觉诶好了一点，但很有可能比如说从百分之七十到百分之七十五，他发现好的比百分之五十到百分之六十那个还多。[13.06][13.17][S02]嗯。[13.45][13.62][S01]这是完全有可能。[14.38][14.61][S02]如果是百分之八十到百分之九十，百分之九十到百分之百，这个感受会更显著。[18.43][18.62][S01]呃，那也不一定，因为可能过了可能到百分之八十到百分之九十，用户就发现没有任何区别，甚至还变差了。[24.98][25.46][S02]你说完全没有变慢，你是基于什么标准？[27.56][27.59][S01]我觉得是基于呃我个人作为一个研究员的感觉，就是我觉得我我个人得到的感受是这个模型学东西的能力越来越强。以前可能让模型学会干一件事情，需要动很多脑筋。[42.69][43.02][S02]哦。[43.32][43.24][S01]但现在可能不需要动那么多那么多脑筋了。最重要的事儿你是要把这个问题定义清楚，然后想清楚怎么去构建合适的数据。[51.02][51.14][S02]嗯。[51.42][51.28][S01]当然数据现在数据就更广泛，指向环境啊之类也都在的，呃，包括在内了。然后呃剩下的事情好像很多时候是是顺其自然的了。[62.62][62.97][S01]对。[63.21][63.62][S02]学习能力变强是为什么呀？模型的学习能力变强。[66.18][67.02][S01]我觉得可能一方面呃[69.34][71.47][S01]原因可能有很多方面，但我觉得可能一方面也是因为呃预训练其实在过去的几个月里，我觉得还是越来越强了的。[79.01][79.07][S02]预训练。[79.59][79.62][S01]对对，模型的预训练其实在过去几个月里还是变强了。[82.88][82.95][S02]嗯。[83.21][83.27][S01]我觉得这个可能是一个呃[85.55][86.62][S01]从某种意义上来说比较有争议的事儿。因为呃几个月以前，我觉得就是很多人已经在讨论预训练的这个SKIN老是不是已经到头了。[95.54][95.62][S02]嗯。[95.92][96.04][S01]啊，我的体验是没有，而且我的感觉是在未来的四个月也没有看到到头的迹象。[104.98][106.56][S01]对。[106.82][107.01][S02]嗯，你觉得到头是为什么呢？[109.11][109.62][S01]我觉得嗯我我我显然不知道大家觉得到头的原因是什么，因为我自己没觉得到头。但是我觉得我的猜测是一个人觉得一个规律到头了，无非以下两种情况。[126.00][126.08][S02]啊。[126.34][126.34][S01]啊，一个情况是他觉得这个规律的适用范围到头了啊，就可能就是就可能从根本意义上讲，SKIN老就是没有办法无穷延展下去的，维持有可能是对的啊。但是这是一种猜测，就是这个人可能觉得这个这个规律适用范围到到头了。[144.71][144.96][S01]另一种可能是这个人觉得这个规律其中的有一个条件不能满足了。比如说他觉得数据就已经撞上墙了，那我完全没有办法延展下去了，这是另外一种可能性。但是其实还有第三种可能性，其他可能性就是其实嗯他这个工作哪里有一个BUG,他自己没发现，所以他觉得到头了。[169.52][168.52][S02]哦。[168.86][169.84][S02]哦。[170.18][170.22][S01]对，呃，我觉得从我的观点，从我的观感上来说呢，我觉得呃[176.88][179.14][S01]可能绝大多数撞到墙的人是因为第三种。[182.12][182.95][S02]是有BUG。[183.59][183.65][S01]嗯。[183.89][184.14][S02]是哪种BUG？[184.82][185.23][S01]我觉得呃BUG是有很多种可能性的。比如说一种可能性是你SKIN老做的时候，一些科学的假设没有做对。就是说你选什么样的TOKENHORIZON,就是每一每一个大小的模型，选什么样的这个这个期待的训练的数据量，然后怎么这个数据量呃是呃这个数据是从哪里选，然后呃有可能这些比较科学的选择没有选清楚是一种可能性。但是我觉得还有一种可能性，就是纯粹有个BUG.[214.31][215.67][S01]这个其实在业界我觉得也不惊奇。很多时候[220.23][221.32][S01]修好一个BUG带来的进展是远大于一些很很神奇的技巧的。[227.02][227.66][S01]对。[227.88][227.94][S02]哦。[228.28][228.43][S01]然后呃当然还有另外的的情况，我觉得我就刚才给的这种两种例子，反正是我见到过比较比较多的情况。[236.93][237.51][S02]那你们的BUG怎么办？你们怎么解决BUG问题的？[240.03][240.92][S01]我觉得。[241.58][241.77][S02]我感觉这更像是一个信念的问题。因为当你遇到一个BUG,你觉得它不能解除，你就会说这个到头了。当你遇到个BUG,我觉得哦这个肯定可以解决，那你就觉得这还没有到头，因为肯定每个人都要遇到BUG。[253.61][254.51][S01]对我觉得我觉得呃这可能就像你说的，就其中有一些比较比较信念性的东西，但对我来说更重要的一件事儿是做事系统。[263.71][264.77][S02]嗯。[265.11][264.84][S01]就是你当你一个一个事情[268.88][269.92][S01]和你预测的不一样的时候，你能不能系统性的排除各种可能性，这个我觉得是是一个很重要的事儿。[276.74][276.88][S02]嗯。[277.24][278.15][S01]这个是我觉得JAMMY和TALIA做的比较好的事儿，就是尤其在预训练上吧。就是说当某一个尺度上的行为可能和你想象中不一样的时候，大家能够去去设计合理的我们所谓的ABlation实验，合理的这种实验能够看出来测你的一些想象中的可能的因素，是不是真的因素。我觉得这个这个做做问题的系统性才是才是关键。[306.57][306.72][S02]嗯。[306.98][306.86][S01]对。[307.14][308.88][S02]你觉得嗯模型能力还能提高，那它的驱动力数据算力算法，你觉得它的驱动力主要来源于哪个？[315.62][316.21][S01]呃，[316.99][318.69][S01]我觉得其实都有，但是嗯从某种意义上来说，数据和算力两个事儿其实是很强关联的一件事儿。[328.45][328.56][S02]数据和算力嗯。[329.58][329.51][S01]对，因为呃你算力上去了，自然就会需要更多的数据。[332.69][33"
assert result["text"] == excepted


for segment in parse_transcript(result["text"]):
    print(segment.start, segment.end, segment.speaker, segment.text)
