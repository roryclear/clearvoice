from clearvoice import omni, waveform_to_wav_bytes, SAMPLING_RATE, save_voice
import subprocess, os

def save_audio(start, end):
  os.remove("tmp.wav")
  subprocess.run(["ffmpeg", "-ss", start, "-to", end, "-i", "podcast/podcast.mp4", "-vn", "-acodec", "pcm_s16le", "tmp.wav"])

def save_voice_from_audio(start, end, voice_name, text):
  save_audio(start=start, end=end)
  save_voice(audio="tmp.wav", voice_name=voice_name,
             text=text)

from dataclasses import dataclass
import re

@dataclass
class Subtitle:
  text: str
  start: float
  end: float

def timestamp_to_seconds(timestamp):
  h, m, s, ms = re.split("[:,]", timestamp)
  return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt_file(file):
    with open(file, "r") as f:
      lines = [line.strip() for line in f]
      file_content = '\n'.join(lines)
    
    subtitles = []
    blocks = re.split(r'\n\s*\n', file_content.strip())
    
    for block in blocks:
      lines = block.strip().split('\n')
      if len(lines) < 2: continue
      timestamp_line = None
      text_lines = []
      
      for i, line in enumerate(lines):
        if '-->' in line:
          timestamp_line = line
          index_line = lines[0] if i > 0 else None
          # Text is everything after timestamp line
          text_lines = lines[i+1:]
          break
      
      if timestamp_line:
        start, end = timestamp_line.split(" --> ")
        text = '\n'.join(text_lines).strip()
        subtitles.append(Subtitle(text=text, start=timestamp_to_seconds(start.strip()), end=timestamp_to_seconds(end.strip())))
  
    return subtitles

if __name__ == "__main__":
  subtitles_cn = parse_srt_file("podcast/english.srt")
  print(subtitles_cn)

  subtitles_en = parse_srt_file("podcast/chinese.srt")
  print(subtitles_en)
  exit()
  model = omni()
  save_voice_from_audio(start="00:05:35.5", end="00:05:47.5", voice_name="host",
                        text="因为那个顺雨他之前在去年的时候说, AI进入了the second half, 进入了下半场这个成为了一个非常有名的观点, 你觉得今天的AI在一个什么样的时期, 你能给它一个定义吗")
  audio = model.generate(
    text="hello, this is the host of the podcast speaking english now, can you understand me or not? thank you for listening.",
    cv_path="voices/host.cv",
    num_steps=32,
    language="en"
  )
  with open("outputs/host_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))

  save_voice_from_audio(start="00:08:55", end="00:09:09", voice_name="guest",
                        text="我觉得呢, 其实是有意愿的成分在的, 尤其在过去的情况下主要是意愿, 就是当大家能从纸面上就看出区别的时候, 那时候意愿肯定是占大多数的")

  audio = model.generate(
    text="hello, this is the guest of the podcast speaking english now, can you understand me or not? thank you for listening.",
    cv_path="voices/guest.cv",
    num_steps=32,
    language="en"
  )
  with open("outputs/guest_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))

  # for now manually just say when speaker changes...
  voice_chages = [94, 157, 166]
  host = True
  #for t in voice_chages:


  print("here")