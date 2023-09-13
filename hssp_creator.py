import lzstring
import json

f = open('simpleproject2.json')
data = json.load(f)


x = lzstring.LZString()
hssp = x.compressToUTF16(json.dumps(data))


with open('simpleproject.hssp','w', encoding="utf-16") as file:
    file.write(hssp)
print(hssp)
