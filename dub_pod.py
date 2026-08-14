from clearvoice import omni, waveform_to_wav_bytes, SAMPLING_RATE, save_voice
import subprocess, os

def save_audio(start, end):
  os.remove("tmp.wav")
  subprocess.run(["ffmpeg", "-ss", start, "-to", end, "-i", "podcast/podcast.mp4", "-vn", "-acodec", "pcm_s16le", "tmp.wav"])

if __name__ == "__main__":
  # get host voice
  # it's 12 seconds long remember
  model = omni()
  save_audio(start="00:05:35.5", end="00:05:47.5")
  HOST_LEN = 12
  save_voice(audio="tmp.wav", voice_name="host",
             text="因为那个顺雨他之前在去年的时候说, AI进入了the second half, 进入了下半场这个成为了一个非常有名的观点, 你觉得今天的AI在一个什么样的时期, 你能给它一个定义吗")
  audio = model.generate(
    text="hello, this is the host of the podcast speaking english now, can you understand me or not? thank you for listening.",
    cv_path="voices/host.cv",
    num_steps=32,
    language="en"
  )
  with open("outputs/host_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))

  save_audio(start="00:08:55", end="00:09:09")
  save_voice(audio="tmp.wav", voice_name="guest",
             text="我觉得呢, 其实是有意愿的成分在的, 尤其在过去的情况下主要是意愿, 就是当大家能从纸面上就看出区别的时候, 那时候意愿肯定是占大多数的")

  audio = model.generate(
    text="hello, this is the guest of the podcast speaking english now, can you understand me or not? thank you for listening.",
    cv_path="voices/guest.cv",
    num_steps=32,
    language="en"
  )
  with open("outputs/guest_test.wav", "wb") as f: f.write(waveform_to_wav_bytes(audio, SAMPLING_RATE))

  print("here")