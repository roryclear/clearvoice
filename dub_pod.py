from clearvoice import omni, waveform_to_wav_bytes, SAMPLING_RATE, save_voice, REF_AUDIO_LEN
import subprocess, os, math

MAX_REF_AUDIO_LEN=15

def save_audio(start, end):
  if os.path.exists("tmp.wav"): os.remove("tmp.wav")
  subprocess.run(["ffmpeg", "-ss", str(start), "-to", str(end), "-i", "podcast/podcast.mp4", "-vn", "-acodec", "pcm_s16le", "tmp.wav"])

def save_voice_from_audio(start, end, voice_name, text):
  save_audio(start=start, end=end)
  save_voice(audio="tmp.wav", voice_name=voice_name, text=text)

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

      # remove laughing hahahas in en and cn
      text = re.sub(r'(?:ha){2,}', '', text, flags=re.IGNORECASE)
      text = re.sub(r'(?:ha){2,}|哈{2,}', '', text, flags=re.IGNORECASE)

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

  # todo, just use the previous sample instead of a random one for short segments?
  save_voice_from_audio(start="00:05:35.5", end="00:05:44.0", voice_name="2",
                        text="因为那个顺雨他之前在去年的时候说, AI进入了the second half, 进入了下半场这个成为了一个非常有名的观点")
  audio = model.generate(
    text=f"hello, this is speaker {2} speaking english now, can you understand me or not? thank you for listening.",
    cv_path=f"voices/{2}.cv",
    num_steps=32,
    language="en"
  )
  with open(f"outputs/{2}_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))

  
  save_voice_from_audio(start="00:01:35.161", end="00:01:42.669", voice_name="1",
                        text="啊 可以 对，就是我叫姚顺宇然后显然也有一个跟我几乎同名的朋友")
  audio = model.generate(
    text=f"hello, this is speaker {1} speaking english now, can you understand me or not? thank you for listening.",
    cv_path=f"voices/{1}.cv",
    num_steps=32,
    language="en"
  )
  with open(f"outputs/{1}_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))
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
  voice_changes = [38, 42, 74, 81, 112, 116, 120, 121, 131, 133, 139, 140, 147, 148, 172, 175, 180, 188, 224, 230, 234, 235, 263, 278, 282, 291, 292, 317, 318, 323, 324, 350, 355, 356, 359, 362, 363, 367, 368, 389, 394, 398, 401,
                  420, 421, 427, 428, 432, 433, 448, 454, 469, 476, 500, 502, 523, 530, 547, 548, 579, 580, 598, 599, 602, 605, 612, 613, 614, 616, 621, 623, 632, 635, 637, 638, 641, 642, 652, 653, 675, 678, 680, 697, 722, 728,
                  759, 760, 771, 774, 775, 777, 800, 802, 807, 809, 827, 829, 834, 835, 851, 852, 861, 862, 885, 886, 891, 893, 914, 915, 918, 920, 921, 930, 939, 940, 955, 961, 965]

  voice_changes =  [9.26, 37.06, 43.61, 45.38, 68.94, 69.74, 70.81, 71.93, 87.89, 94.94, 157.88, 161.75, 161.92, 166.52, 221.68, 226.23, 236.98, 238.04, 252.32, 255.34, 265.01, 267.24, 275.49, 276.12, 319.57, 324.29, 334.65]
  speakers = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]

  time = 0

  subprocess.run([
      "ffmpeg",
      "-y",
      "-i", "podcast/podcast_en.wav",
      "-af", f"volume=enable='gte(t,{time})':volume=0",
      "-c:a", "pcm_s16le",
      "podcast/podcast_en_temp.wav"
  ], check=True)
  os.replace("podcast/podcast_en_temp.wav", "podcast/podcast_en.wav")


  sub_idx = 0
  speaker_idx = 0
  time = voice_changes[speaker_idx]
  use_voice = True
  while subtitles_en[sub_idx].start < voice_changes[-1]:
    use_voice = (voice_changes[speaker_idx] - voice_changes[speaker_idx-1]) > 5
    cn_str = ""
    if use_voice:
      n_subs = 0
      while subtitles_cn[sub_idx+n_subs].end - subtitles_cn[sub_idx].start < 15 and subtitles_en[sub_idx+n_subs].start < voice_changes[speaker_idx]-0.1:
        if n_subs > 0:
          if subtitles_cn[sub_idx+n_subs].start > subtitles_cn[sub_idx].end: cn_str += ","
          cn_str += " "
        cn_str += subtitles_cn[sub_idx+n_subs].text
        n_subs += 1
      print("RORY CN LENGTH =",subtitles_cn[sub_idx+n_subs].end - subtitles_cn[sub_idx].start)

    en_str = ""
    n_subs = 0
    while subtitles_en[sub_idx+n_subs].start < voice_changes[speaker_idx] - 0.1:
      if n_subs > 0:
        if subtitles_cn[sub_idx+n_subs].start > subtitles_cn[sub_idx].end: en_str += ","
        en_str += " "
      en_str += subtitles_en[sub_idx+n_subs].text
      n_subs += 1

    print("rory cn sample =",cn_str)
    print("rory en subs =",en_str)
    print("speaker =",speakers[speaker_idx],"\n\n\n")
    print("rory use_voice =",use_voice)
    speaker_idx+=1
    sub_idx+=n_subs

  print("here")