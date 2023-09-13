import lzstring
import json


with open('simpleproject.hssp','r', encoding="utf-16") as file:
    data = file.read()



x = lzstring.LZString()
json = x.decompressFromUTF16(data)


# f = open('simpleproject.json')
# data = json.load(f)

print(json)
