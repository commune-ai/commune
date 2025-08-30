import commune as c
import os
import json

class Desc:
    def forward(self,   
                module='store', 
                task= 'summarize the following in the core pieces of info and ignore the rest',
                model='anthropic/claude-3.5-sonnet',
                max_age=200,
                cache=True,
                update=False,
                 **kwargs):
                 
        code  = c.code(module)
        code_hash =  c.hash(code)
        path = c.abspath('~/.desc/' + code_hash + '.json')
        
        result = c.get(path, max_age=max_age, update=update)
        if cache and result != None:
            print(f'Found {module} in cache')
            return result
    
        anchors = ['<JSONSTART>', '<JSONEND>']
        text = f"""
        --CONTEXT--
        {code}
        --TASK--
        {task}
        --FORMAT--
        in json format please DICT(key->aboutkey) WHERE KEY IS THE OBJECT OF INTEREST AND VALUE IS THE VALUE
        DO IT BETWEEN THESE TAGS FOR PARSING
        {anchors[0]}DICT(key:value){anchors[1]}
        """
        output = ''
        for ch in c.ask(text, stream=1, **kwargs):
            output += ch
            print(ch, end='')
        
        output =  json.loads(output.split(anchors[0])[-1].split(anchors[1])[0])
        c.put(path, output)
        return output




