 
    


import os
import json
import commune as c


class Reduce:
    def __init__(self, agent='agent'):
        self.agent = c.module(agent)()
    def forward(self, text, max_chars=69000 , timeout=40, max_age=30, model='google/gemini-2.0-flash-001'):
        if os.path.exists(text):
            text = c.file2text(text)
        text = str(text)
        num_chars = len(text)
        if num_chars >= max_chars :
            batch_text = [text[i:i+max_chars] for i in range(0, num_chars, max_chars)]
            futures =  [c.submit(self.forward, [batch], timeout=timeout) for batch in batch_text]
            output = []
            try:
                for future in c.as_completed(futures, timeout=timeout):
                    print(f'Future: {future}')
                    output += [str(future.result())]
            except TimeoutError as e:
                print(f'TimeoutError: {e}')
            final_length = len(text)
        text = f'''
        GOAL
        summarize the following into tupples and make sure you compress as much as oyu can
        CONTEXT
        {text}
        OUTPUT JSON AS FOLLOWS FOR PARSING ONLY OUTPUT WITHIN THE FORMAT
        JSON FORMAT
        
        List[Dict[str, str]]
        '''

        data =  self.agent.forward(text, timeout=timeout, max_age=max_age, model=model)
        output = ''
        for ch in data:
            output += ch
            print(ch, end='')
        # REMOVE THE FIRST JSON THING FROM THE OUTPUT
        start_json ="""```json"""
        end_json = """```"""
        output = output.split(start_json)[-1].split(end_json)[0]
        try:
            result =  json.loads(output)
        except json.JSONDecodeError as e:
            print(f'Error decoding JSON: {e}')
            result = output
        return result

