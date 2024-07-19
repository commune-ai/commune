import commune as c 
import os

class Agent(c.Module):



    prompt = """
    GIVEN THE FOLLOWING QUERY
    YOU ARE A CODER THAT IS FEARLESS AND CAN SOLVE ANY PROBLEM
    THE QUERY IS AS FOLLOWS

    ---START OF QUERY---
    {text}
    -- END OF QUERY
    
    THIS IS YOUR CURRENT CODEBASE THAT YOU CAN IMPROVE PROVIDED BELOW
    --- START OF FILE ({file}) ---
    {code}
    --- END OF FILE  ({file})---

    RESPOND IN THE FULL CODE PLEASE AND NOTHING ELSE, COMMENT IF YOU WANT TO ADD ANYTHING ELSE.
    """

    def __init__(self, model='model.openrouter'):
        self.model =  c.module(model)()
    
    def code(self, text ,  file ,  trials=1, code = None):

        code = code or self.read_code(file)
        text = text 
        for trials in range(trials):
            text = self.prompt.format(text=text, code=code, file=file)
            code = self.model.generate(text)
            self.write_code(file, code, replace=True)

        return code

    def write_code(self, file, code, replace=False):
            if replace and os.path.exists(file):
                os.remove(file)
            
            with open(file, 'a') as f:
                f.write(code)
    
    def read_code(self, file):
        if not os.path.exists(file):
            return None
        with open(file, 'r') as f:
            code = f.read()
        return code

            




    def code_function_score(self, code):
        self.model 
        