
import commune as c 
import os

class Agent(c.Module):

    prompt = """
    GIVEN THE FOLLOWING QUERY
    ---OBJECTIVE---
    {objective}
    ---USER---
    {text}
    --- CODE ---
    {code}
    --- NEW CODE ---
        """  
    def __init__(self, 
                 model='model.openrouter', 
                 objective='YOU ARE A CODER THAT IS FEARLESS AND CAN SOLVE ANY PROBLEM THE QUERY IS AS FOLLOW, RESPOND IN THE FULL CODE PLEASE AND NOTHING ELSE, COMMENT IF YOU WANT TO ADD ANYTHING ELSE.'):
        self.model =  c.module(model)()
        self.objective = objective
    
    def forward(self, 
            text ,  
            file=None , 
            trials=1, 
            code = None,
            stream=False,
            objective=None,
             ):
        
        """
        params:
        text: str: the text that you want to generate code from
        file: str: the file that you want to append the code to
        trials: int: the number of trials to run
        code: str: the code that you want to improve
        stream: bool: stream the output

        """
        if trials > 1:
            for trial in range(trials):
                c.print(f"Trial {trial}")
                code = self.forward(text=text, 
                                file=file, 
                                code=code, 
                                stream=stream
                                )
            return code
        if file != None and code == None:
            code = self.read_code(file)
        objective = objective or self.objective
        text = self.prompt.format(text=text, code=code, file=file, objective=objective)
        code = self.model.generate(text, stream=stream)
        if file :
            self.write_code(file, code, replace=True)
        return code

    def write_code(self, file, code, replace=True):
        # if this is a generator 
        if os.path.exists(file):
            os.remove(file)
        if c.is_generator(code):
            for i, code in enumerate(code):
                with open(file, 'a') as f:
                    f.write(code)
        else:
            with open(file, 'a') as f:
                f.write(code)
        
    def read_code(self, file):
        if not os.path.exists(file):
            return None
        with open(file, 'r') as f:
            code = f.read()
        return code
