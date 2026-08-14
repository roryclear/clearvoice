from clearvoice import omni, waveform_to_wav_bytes, SAMPLING_RATE, save_voice
import subprocess, os

def save_audio(start, end):
  os.remove("tmp.wav")
  subprocess.run(["ffmpeg", "-ss", str(start), "-to", str(end), "-i", "podcast/podcast.mp4", "-vn", "-acodec", "pcm_s16le", "tmp.wav"])

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
  subtitles_en = parse_srt_file("podcast/english.srt")
  print(subtitles_en)

  subtitles_cn = parse_srt_file("podcast/chinese.srt")
  print(subtitles_cn)
  #exit()
  model = omni()
  '''
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
  '''
  #exit()

  # for now manually just say when speaker changes...
  voice_changes = [94, 158, 166]
  host = True
  time = 86 # skip into
  change_idx = 0
  sub_idx = 0
  while subtitles_cn[sub_idx].end < time: sub_idx+=1
  #while sub_idx < len(subtitles_cn):
  while subtitles_cn[sub_idx].start < voice_changes[-1]:
    text_cn = subtitles_cn[sub_idx].text
    text_en = subtitles_en[sub_idx].text
    i = 1
    while subtitles_cn[sub_idx+i].end < voice_changes[change_idx]:
      print("end =",subtitles_cn[sub_idx+i].end, voice_changes[change_idx])

      if subtitles_cn[sub_idx+i].end - subtitles_cn[sub_idx].start < 10: # max 15 sec ref
        text_cn += subtitles_cn[sub_idx+i].text
        if subtitles_cn[sub_idx+i].start - subtitles_cn[sub_idx+i-1].end > 0:
          text_cn += ", "
        else:
          text_cn += " "

      # todo dup
      if subtitles_cn[sub_idx+i].start - subtitles_cn[sub_idx+i-1].end > 0:
        text_en += ", "
      else:
        text_en += " "

      text_en += subtitles_en[sub_idx+i].text
      i+=1
    print(text_cn, "\n", text_en)
    print("start =",subtitles_cn[sub_idx].start, "end =",subtitles_cn[sub_idx+i].end)
    new_voice = (subtitles_cn[sub_idx+i].end - subtitles_cn[sub_idx].start) > 5 # min 5 seconds sample
    if new_voice:
      save_voice_from_audio(start=subtitles_cn[sub_idx].start, end=subtitles_cn[sub_idx+i].end, voice_name="tmp", text=text_cn)
      audio = model.generate(
        text=text_en,
        cv_path="voices/tmp.cv",
        num_steps=16,
        language="en"
      )
      with open("outputs/tmp.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))
    else:
      print("TOO SHORT, USE BACKUP VOICE")
      exit()

    # slop ffmpeg normally is ok...
    start_time = subtitles_cn[sub_idx].start
    end_time = subtitles_cn[sub_idx + i].end

    # Output to temp file first
    temp_output = "podcast/podcast_en_temp.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", "podcast/podcast_en.wav",
        "-i", "outputs/tmp.wav",
        "-filter_complex",
        f"[0:a]atrim=0:{start_time},asetpts=PTS-STARTPTS[before];"
        f"[0:a]atrim={end_time},asetpts=PTS-STARTPTS[after];"
        f"[1:a]apad=whole_dur={end_time - start_time}[replacement];"
        f"[before][replacement][after]concat=n=3:v=0:a=1[outa]",
        "-map", "[outa]",
        temp_output
    ]

    subprocess.run(cmd, check=True)

    # Replace original with temp file
    os.replace(temp_output, "podcast/podcast_en.wav")
      
    sub_idx+=i
    change_idx+=1
    host = not host

  print("here")