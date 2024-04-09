import commune as c

class AgiFam(c.Module):
    def __init__(self, 
                 model= 'model.openrouter', 
                 prompts=['Answer in json ()']):
        self.model = c.module(model)()
        self.prompts = prompts


    def add_prompt(self, prompt:str, idx:int = None):
        if idx:
            self.prompts.insert(idx, prompt)
        else:
            self.prompts.append(prompt)
        return {'prompts': self.prompts, 'idx': idx}
    
    def rm_prompt(self, idx:int):   
        del self.prompts[idx]
        return {'prompts': self.prompts, 'idx': idx}
    

    def forward(self, prompt:str, **kwargs):
        for p in self.prompts:
            prompt += p + '\n'
        return self.model.forward(prompt, **kwargs)
    
    def ask(self, *prompt, **kwargs):
        prompt = ' '.join(prompt)
        return self.forward(prompt, **kwargs)
    
    

    

    