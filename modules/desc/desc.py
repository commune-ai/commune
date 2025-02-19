import commune as c
import os
import json

class Desc:
    public_functions = ["generate", "info"]
    def __init__(self, 
                 model='anthropic/claude-3.5-sonnet',
                **kwargs):
        
        self.model = c.module('model.openrouter')(model=model, **kwargs)

    def forward(self, module='store', 
                task= 'summarize the following in the core pieces of info and ignore the rest',
                 **kwargs):

        anchors = ['<JSONSTART>', '<JSONEND>']
        code = {c.code(module)}
        text = f"""
        --CONTEXT--
        {code}
        --TASK--
        {task}
        --FORMAT--
        in json format please DICT(key->aboutkey) WHERE KEY IS THE OBJECT OF INTEREST AND VALUE IS THE VALUE
        DO IT BETWEEN THESE TAGS FOR PARSING
        {anchors[0]}
        DICT(key:value)
        {anchors[1]}
        """
        output = ''
        for ch in self.model.generate(text, stream=1, **kwargs):
            output += ch
            print(ch, end='')
        return  json.loads(output.split(anchors[0])[-1].split(anchors[1])[0])




