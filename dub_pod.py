from clearvoice import omni, waveform_to_wav_bytes, SAMPLING_RATE, save_voice
import subprocess, os

def save_audio(start, end):
  os.remove("tmp.wav") if os.path.exists("tmp.wav") else None
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
  number: int

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
    subtitle_number = None
    
    for i, line in enumerate(lines):
      if '-->' in line:
        timestamp_line = line
        # The number is usually the line before timestamp
        if i > 0:
          try:
            subtitle_number = int(lines[i-1])
          except ValueError:
            subtitle_number = None
        # Text is everything after timestamp line
        text_lines = lines[i+1:]
        break
    
    if timestamp_line:
      start, end = timestamp_line.split(" --> ")
      text = '\n'.join(text_lines).strip()
      text = re.sub(
        r'\([^)]*\)|'
        r'\[[^\]]*\]|'
        r'\{[^}]*\}|'
        r'（[^）]*）|'
        r'［[^］]*］|'
        r'｛[^｝]*｝|'
        r'「[^」]*」|'
        r'『[^』]*』',
        '',
        text
      )
      text = re.sub(r'\s+', ' ', text).strip()

      subtitles.append(Subtitle(
        number=subtitle_number,
        text=text, 
        start=timestamp_to_seconds(start.strip()), 
        end=timestamp_to_seconds(end.strip())
      ))

  return subtitles

if __name__ == "__main__":
  subtitles_en = parse_srt_file("podcast/english.srt")
  subtitles_cn = parse_srt_file("podcast/chinese.srt")

  # manual fixes for now, auto generate warning thing and some chinese subs are split in two vs english, n^2 don't care
  for i in range(len(subtitles_en)): subtitles_en[i].number -= 1
  while len(subtitles_cn) != len(subtitles_en):
    for i in range(len(subtitles_en)):
      if subtitles_cn[i].start != subtitles_en[i].start:
        subtitles_cn[i-1].text += " " + subtitles_cn[i].text
        subtitles_cn[i-1].end = subtitles_cn[i].end
        subtitles_cn.pop(i)
        break 

  # check fix worked:
  #for i in range(len(subtitles_en)):
  #  print("rory i=",i)
  #  print(subtitles_en[i], subtitles_cn[i])
  #  assert [subtitles_en[i].start, subtitles_en[i].end] == [subtitles_cn[i].start, subtitles_cn[i].end]

  print(len(subtitles_en), len(subtitles_cn))
  model = omni()
  
  save_voice_from_audio(start="00:05:35.5", end="00:05:47.5", voice_name="host",
                        text="因为那个顺雨他之前在去年的时候说, AI进入了the second half, 进入了下半场这个成为了一个非常有名的观点, 你觉得今天的AI在一个什么样的时期, 你能给它一个定义吗")
  '''
  audio = model.generate(
    text="hello, this is the host of the podcast speaking english now, can you understand me or not? thank you for listening.",
    cv_path="voices/host.cv",
    num_steps=32,
    language="en"
  )
  with open("outputs/host_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))
  '''
  

  save_voice_from_audio(start="00:08:55", end="00:09:09", voice_name="guest",
                        text="我觉得呢, 其实是有意愿的成分在的, 尤其在过去的情况下主要是意愿, 就是当大家能从纸面上就看出区别的时候, 那时候意愿肯定是占大多数的")
  '''
  audio = model.generate(
    text="hello, this is the guest of the podcast speaking english now, can you understand me or not? thank you for listening.",
    cv_path="voices/guest.cv",
    num_steps=32,
    language="en"
  )
  with open("outputs/guest_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))
  '''
  
  #exit()

  subprocess.run([
      "ffmpeg",
      "-y",
      "-i", "podcast/podcast_en.wav",
      "-af", "volume=0",
      "-c:a", "pcm_s16le",
      "podcast/podcast_en_temp.wav"
  ], check=True)
  os.replace("podcast/podcast_en_temp.wav", "podcast/podcast_en.wav")

  # for now manually just say when speaker changes...english srt! use first num after
  voice_changes = [42, 74, 81, 112, 116, 120, 121, 131]
  host = True
  time = 86 # skip intro
  change_idx = 0
  sub_idx = 0
  while subtitles_en[sub_idx].start < time: sub_idx += 1
  #while sub_idx < len(subtitles_cn):
  while subtitles_en[sub_idx].number < voice_changes[-1]:
    text_cn = subtitles_cn[sub_idx].text
    text_en = subtitles_en[sub_idx].text
    i = 1
    while subtitles_en[sub_idx+i].number < voice_changes[change_idx]-1: # todo, en and cn are different but it should be ok lol
      if subtitles_en[sub_idx+i].end - subtitles_en[sub_idx].start < 10 and subtitles_en[sub_idx+i].start < subtitles_en[voice_changes[change_idx]-1].start: # max 15 sec ref
        text_cn += subtitles_cn[sub_idx+i].text
        cn_end = subtitles_en[sub_idx+i].end
        if subtitles_en[sub_idx+i].start - subtitles_en[sub_idx+i-1].end > 0:
          text_cn += ", "
        else:
          text_cn += " "

      # todo dup
      if subtitles_en[sub_idx+i].start - subtitles_en[sub_idx+i-1].end > 0:
        text_en += ", "
      else:
        text_en += " "

      text_en += subtitles_en[sub_idx+i].text
      i+=1
    print(text_cn, "\n", text_en)
    print("start =",subtitles_cn[sub_idx].start, "end =",subtitles_cn[sub_idx+i].end)

    new_voice = (subtitles_cn[sub_idx+i].end - subtitles_cn[sub_idx].start) > 5 # min 5 seconds sample
    if new_voice:
      voice = "voices/tmp.cv"
    else:
      print("TOO SHORT?", host)
      voice = "voices/host.cv" if host else "voices/guest.cv"

    save_voice_from_audio(start=subtitles_cn[sub_idx].start, end=cn_end, voice_name="tmp", text=text_cn)
    audio = model.generate(
      text=text_en,
      cv_path=voice,
      num_steps=32,
      language="en"
    )
    with open("outputs/tmp.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))

    # slop ffmpeg normally is ok...
    start_time = subtitles_cn[sub_idx].start
    end_time = subtitles_cn[sub_idx + i].end

    # speed up or slow down to fit sub duration
    actual_duration = len(audio) / SAMPLING_RATE
    start_time = subtitles_cn[sub_idx].start
    end_time = subtitles_cn[sub_idx + i].end
    target_duration = end_time - start_time
    speed = actual_duration / target_duration
    if speed > 0.7 and speed < 1.7:
      with open("outputs/tmp.wav", "wb") as f:
          f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))
      subprocess.run([
          "ffmpeg", "-y",
          "-i", "outputs/tmp.wav",
          "-filter:a", f"atempo={speed}",
          "outputs/tmp2.wav"
      ], check=True)
      os.replace("outputs/tmp2.wav", "outputs/tmp.wav")

    temp_output = "podcast/podcast_en_temp.wav"

    duration_ms = int(start_time * 1000)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", "podcast/podcast_en.wav",
        "-i", "outputs/tmp.wav",
        "-filter_complex",
        # Delay tmp.wav until start_time
        f"[1:a]adelay={duration_ms}:all=1[overlay];"
        # Mix it on top of the existing podcast_en.wav
        "[0:a][overlay]amix=inputs=2:duration=first:dropout_transition=0[outa]",
        "-map", "[outa]",
        "-c:a", "pcm_s16le",
        temp_output
    ]
    subprocess.run(cmd, check=True)
    os.replace(temp_output, "podcast/podcast_en.wav")

    sub_idx+=i
    change_idx+=1
    host = not host
  print("here")