import commune as c
model = c.connect('model.vicuna.7b', network='local')
import json


def ask(question='What is the meaning of Life? options: [0,2,3,4]', output_schema={'answer': 'str'}):
    output_schema = json.dumps(output_schema)
    prompt = f"{question} \nANSWER the MC json({output_schema}):\n```json"
    output = model.generate(prompt, max_new_tokens=15)
    return output
c.print(ask())
