from clearvoice import omni, waveform_to_wav_bytes, SAMPLING_RATE, save_voice, REF_AUDIO_LEN
import subprocess, os, math

MAX_REF_AUDIO_LEN=15

def save_audio(start, end):
  if os.path.exists("tmp.wav"): os.remove("tmp.wav")
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
  # todo just rewrite the cn srt so it matches
  for i in range(len(subtitles_en)): subtitles_en[i].number -= 1
  while len(subtitles_cn) != len(subtitles_en):
    for i in range(len(subtitles_en)):
      if subtitles_cn[i].start != subtitles_en[i].start:
        subtitles_cn[i-1].text += " " + subtitles_cn[i].text
        subtitles_cn[i-1].end = subtitles_cn[i].end
        subtitles_cn.pop(i)
        break

  # use lower end if they don't match
  for i in range(len(subtitles_en)):
    if subtitles_cn[i].end != subtitles_en[i].end:
      subtitles_cn[i].end = min(subtitles_cn[i].end, subtitles_en[i].end)
      subtitles_en[i].end = subtitles_cn[i].end
  # check fix worked:
  #for i in range(len(subtitles_en)):
  #  print("rory i=",i)
  #  print(subtitles_en[i], subtitles_cn[i])
  #  assert [subtitles_en[i].start, subtitles_en[i].end] == [subtitles_cn[i].start, subtitles_cn[i].end]

  model = omni()
  
  save_voice_from_audio(start="00:05:35.5", end="00:05:44.0", voice_name="host",
                        text="因为那个顺雨他之前在去年的时候说, AI进入了the second half, 进入了下半场这个成为了一个非常有名的观点")
  '''
  audio = model.generate(
    text="hello, this is the host of the podcast speaking english now, can you understand me or not? thank you for listening.",
    cv_path="voices/host.cv",
    num_steps=32,
    language="en"
  )
  with open("outputs/host_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))
  '''
  

  save_voice_from_audio(start="00:01:35.161", end="00:01:42.669", voice_name="guest",
                        text="啊 可以 对，就是我叫姚顺宇然后显然也有一个跟我几乎同名的朋友")
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

  # reset output file
  '''
  subprocess.run([
      "ffmpeg",
      "-y",
      "-i", "podcast/podcast_en.wav",
      "-af", "volume=0",
      "-c:a", "pcm_s16le",
      "podcast/podcast_en_temp.wav"
  ], check=True)
  os.replace("podcast/podcast_en_temp.wav", "podcast/podcast_en.wav")
  '''

  # for now manually just say when speaker changes...english srt! use first num after
  # for start now use host = False, and add the start of pod to start of voice_changes
  voice_changes = [38, 42, 74, 81, 112, 116, 120, 121, 131, 133, 139, 140, 147, 148, 172, 175, 180, 188, 224, 230, 234, 278, 282]
  host = False
  time = 86 # skip intro
  #time = 4*60 + 34 + 0.140
  change_idx = 0
  sub_idx = 0

  #time = (4*60) + 12 + 0.385
  subprocess.run([
      "ffmpeg",
      "-y",
      "-i", "podcast/podcast_en.wav",
      "-af", f"volume=enable='gte(t,{time})':volume=0",
      "-c:a", "pcm_s16le",
      "podcast/podcast_en_temp.wav"
  ], check=True)
  os.replace("podcast/podcast_en_temp.wav", "podcast/podcast_en.wav")


  while subtitles_en[sub_idx].number < voice_changes[-1]:
    text_cn = subtitles_cn[sub_idx].text
    text_en = subtitles_en[sub_idx].text
    i = 1
    print(subtitles_en[sub_idx+i].number, voice_changes[change_idx]-1, "HERE RORY")
    while subtitles_en[sub_idx+i].number < voice_changes[change_idx]-1: # todo, en and cn are different but it should be ok lol
      if subtitles_en[sub_idx+i].end - subtitles_en[sub_idx].start < MAX_REF_AUDIO_LEN and subtitles_en[sub_idx+i].start < subtitles_en[voice_changes[change_idx]-1].start: # max 15 sec ref
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
    if subtitles_cn[sub_idx].start < time:
      sub_idx+=i
      change_idx+=1
      host = not host
      continue
    new_voice = (subtitles_cn[sub_idx+i].end - subtitles_cn[sub_idx].start) > 5 # min 5 seconds sample
    if subtitles_en[sub_idx].number > 42 and subtitles_en[sub_idx].number < 74: new_voice = False # broken bit because guest talks a little
    if new_voice:
      voice = "voices/tmp.cv"
      save_voice_from_audio(start=subtitles_cn[sub_idx].start, end=cn_end, voice_name="tmp", text=text_cn)
    else:
      print("TOO SHORT?", host)
      voice = "voices/host.cv" if host else "voices/guest.cv"

    audio = model.generate(
      text=text_en,
      cv_path=voice,
      num_steps=32,
      language="en",
    )
    with open("outputs/tmp.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))

    start_time = subtitles_cn[sub_idx].start
    end_time = subtitles_cn[sub_idx+i-1].end
    target_duration = end_time - start_time

    wav_bytes = waveform_to_wav_bytes(audio, SAMPLING_RATE)
    actual_duration = len(audio) / SAMPLING_RATE

    speed = actual_duration / target_duration
    if speed > 0.7 and speed < 1.3:
      with open("outputs/tmp.wav", "wb") as f:
          f.write(wav_bytes)
      
      # Apply atempo
      subprocess.run([
          "ffmpeg", "-y",
          "-i", "outputs/tmp.wav",
          "-filter:a", f"atempo={speed}",
          "outputs/tmp2.wav"
      ], check=True)
      os.replace("outputs/tmp2.wav", "outputs/tmp.wav")
      
      actual_duration = target_duration
    else:
      with open("outputs/tmp.wav", "wb") as f:
          f.write(wav_bytes)

    duration_ms = max(0, int((start_time * 1000)))
    temp_output = "podcast/podcast_en_temp.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", "podcast/podcast_en.wav",
        "-i", "outputs/tmp.wav",
        "-filter_complex",
        f"[1:a]adelay={duration_ms}:all=1[overlay];"
        "[0:a][overlay]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[outa]",
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