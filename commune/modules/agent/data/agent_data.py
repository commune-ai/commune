import commune as c
import json
class Demo(c.Module):
    instruciton = """

        

    """
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, timeout=30) -> int:
        model = c.connect('model.openai') # connect to the model

        input = json.dumps({
            'instruction': self.instruction, 
            'response': None,
        })

        # get the docs

        return model.generate(input, timeout=timeout)

    