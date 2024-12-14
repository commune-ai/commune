import commune as c
import json
import os

class Agent(c.Module):
    

    def edit(self, module: str, task="make it better"):
        task =  " ".join(task)
        if c.module_exists(module):
            path = c.get_path(module)
        assert os.path.exists(path), f"Path {path} does not exist"
        code = c.get_text(path)
        prompt = f"""
        CONTEXT
        {code}
        TASK
        {task}
        """
        return c.ask(prompt)
    

    def ask(self, prompt, *extra_prompts, **kwargs):
        prompt = ' '.join([prompt] + list(extra_prompts))
        return c.ask(prompt, *extra_prompts, **kwargs)
    
    

        
        