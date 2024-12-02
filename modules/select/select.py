import commune as c
import json
class Select:

    def forward(self, 
                query="update a subnet", 
                module='subspace'):
        options  =  list(c.module(module).fns())
        prompt = f"""
        ----
        QUERY
        ----
        {query}
        ---- 
        INSTRUCTION
        ----
        SELECT FROM THE OPTIONS
        {options}
        ----
        OUTPUT FORMAT (JSON STRING BETWEEN THE ANCHORS)
        ----
        RETURN THE INDICES OF THE SELECTED OPTIONS
        <OUTPUT>DICT(data:LIST[LIST(name:str, score:int[0,100])])</OUTPUT>
        ----
        OUTPUT
        ----
        """
        output = ''
        for ch in c.ask(prompt, model='sonnet'):
            print(ch, end='')
            if ch == '<':
                break
            output += ch
        output = output.split('<OUTPUT>')[1].split('</OUTPUT>')[0]
        output = json.loads(output)["data"]
        output_schema = {k:c.schema(module + '/'+k) for k,v in output}
        return output_schema

