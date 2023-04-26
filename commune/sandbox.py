import jsonlines
import zstandard

import commune

import json

# Open the JSON file
path = '/tmp/commune/dataset/pile/pile/pile/shard-1/01.jsonl'
with open(path, 'r') as f:
    # Loop through each line in the file
    cnt = 0
    t = commune.timer()
    for line in f:
        # Load the JSON data from the line
        data = json.loads(line)
        # Do something with the data, for example print a specific key
        # print(data['text'])
        cnt+= 1
        if cnt % 100 == 0:
            print(cnt/t.seconds)
        # break
