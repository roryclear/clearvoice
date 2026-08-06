import json


def opt(file_a, file_b):
  a_bufs = get_buffers(file_a)

  data_a = json.load(open(file_a))
  data_b = json.load(open(file_b))

  for element in data_a:
    if "copyin" in element.keys() and element["copyin"]["dest"] not in a_bufs:
      data_b.insert(0, element)
      data_a.remove(element) # todo this is slow
  for element in data_a:
    if "buff_alloc" in element.keys() and element["buff_alloc"]["num"] not in a_bufs:
      data_b.insert(0, element)
      data_a.remove(element) # todo this is slow
  
  with open(file_a, "w") as f: json.dump(data_a, f)
  with open(file_b, "w") as f: json.dump(data_b, f)

def get_buffers(file_name):
  data = json.load(open(file_name))
  buffers = set()
  for element in data:
    if "call" in element.keys():
      for b in element["call"]["buffers"]: buffers.add(b)
  return buffers


def get_all_buffers(files):
  file_bufs = []
  for file in files: file_bufs.append(get_buffers(file))

  seen = set()
  for i in reversed(range(len(file_bufs))):
    file_bufs[i] -= seen
    seen |= file_bufs[i]

  ret = {}
  for i in range(len(file_bufs)):
    ret[files[i]] = list(file_bufs[i])
  return ret

rets = get_all_buffers(["0.rc", "1.rc", "2.rc", "100.rc"])
with open("buffers.json", "w") as f: json.dump(rets, f)

#print("here")
#opt("0.rc", "1.rc")
#print("done")


