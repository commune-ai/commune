import commune as c
import json
class Demo(c.Module):
    instruction = "Fill in the template for a gpt agent."

    example = " Make a gpt that can do math." 

    template = {
        "name": "math",
        "description": "A demo agent.",
        "prompt": "Make a gpt that can do math.",
        
    }
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, description) -> int:
        x = json.dumps({
            'instructions': self.instruction,
            'description': description,
            'template': self.template,
            'output_template': "FILL IN THE TEMPLATE",
        })
        
        
        return c.call("model.openai/generate", x)
    