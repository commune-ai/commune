


import os
import json
import commune as c

class VerifyFn:
    def forward(self, fn='module/ls'):
            code = c.code(fn)
            prompt = {
                'goal': 'make the params from the following  and respond in JSON format as kwargs to the function and make sure the params are legit',
                'code': code, 
                'output_format': '<JSON_START>DICT(params:dict)</JSON_END>',
            }
            response =  c.ask(prompt, preprocess=False)
            output = ''
            for ch in response:
                print(ch, end='')
                output += ch
                if '</JSON_END>' in output:
                    break
            params_str = output.split('<JSON_START>')[1].split('</JSON_END>')[0].strip()
            params = json.loads(params_str)['params']

            result =  c.fn(fn)(**params)

            ## SCORE THE RESULT
            prompt = {
                'goal': '''score the code given the result, params 
                and code out of 100 if the result is indeed the result of the code
                make sure to only respond in the output format''',
                'code': code, 
                'result': result,
                'params': params,
                'output_format': '<JSON_START>DICT(score:int, feedback:str, suggestions=List[dict(improvement:str, delta:int)]</JSON_END>',
            }

            response =  c.ask(prompt, preprocess=False)

            output = ''
            for ch in response:
                print(ch, end='')
                output += ch
                if '</JSON_END>' in output:
                    break
                
            return json.loads(output.split('<JSON_START>')[1].split('</JSON_END>')[0].strip())
