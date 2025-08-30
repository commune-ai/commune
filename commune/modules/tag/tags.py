
import commune as c
import json
class Tags:
    def forward(self, module='openrouter', goal='RETURN TAGS AS A LIST AS THE CONTENT'):
        text = f'''
        --CONTENT--
        {c.code_map(module)}
        --GOAL--
        {goal}
        --FORMAT--
        <START_OUTPUT>JSON(data=['tag1', 'tag2'])<END_OUTPUT>
        '''
        model = c.module('openrouter')()
        output = ''
        for ch in  model.forward(text,process_input=False, stream=1):
            print(ch)
            output += ch
        
        output = output.split('START_OUTPUT>')[-1].split('<END_OUTPUT')[0]
        return json.loads(output)