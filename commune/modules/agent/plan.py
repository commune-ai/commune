

import os
import json
import commune as c

class Agent:
    def __init__(self, agent='agent'):
        self.agent = c.module(agent)()
    def forward(self, text, *extra_text, path='./', run=False, **kwargs):
        text = text + ' '.join(list(map(str, extra_text)))
        anchors= ['<START_OUTPUT>','<END_OUTPUT>']
        prompt = f"""

        INSTRUCTION:
        YOU NEED TO PLAN THE NEXT SET OF ACTIONS IN THE COMMAND LINE
        IF YOU ARE UNSURE YOU CAN READ THE FILE AND THEN DECIDE YOU CAN 
        DECIDE TO RUN THE COMMANDS AND WE CAN FEED IT TO YOU AGAIN SO YOU 
        HAVE MULTIPLE CHANCES TO GET IT RIGHT
        TASK:
        {text}
        CONTEXT:
        {c.files(path)}
        OUTPUT FORMAT:
        {anchors[0]}LIST[dict(cmd:str, reason:str)]{anchors[1]}
        """

        output = ''
        for ch in self.agent.forward(prompt, **kwargs):
            output += ch
            print(ch, end='')
            if anchors[1] in output:
                break
        
        plan =  json.loads(output.split(anchors[0])[1].split(anchors[1])[0])

        if run:
            input_response = input(f'Run plan? (y/n): {plan}')
            if input_response.lower() in ['y', 'yes']:
                for p in plan:
                    print('Running command:', p['cmd'])
                    c.cmd(p['cmd'])

        return plan

   