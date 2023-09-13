import json
import csv

with open("data.csv") as f:
    data = f.read()
data = data.replace(",", ";")
data = data.replace("\n", "\r\n")

f = open("simpleproject2.json")
j_data = json.load(f)

j_data["dataStore"]["tableCSV"] = data

csvfile = open("data.csv", "r")
reader_variable = csv.reader(csvfile, delimiter=",")


tableData = []

for row in reader_variable:
    item = {}
    item["A"] = row[0]
    item["B"] = row[1]
    tableData.append(item)
j_data["chartParametersStore"]["type"] = "scatter"
j_data["dataStore"]["tableRowData"] = tableData

# Serializing json
json_object = json.dumps(j_data, indent=4)

with open("simpleproject3.json", "w") as outfile:
    outfile.write(json_object)

print(data)
