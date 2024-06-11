import commune as c

class Basi(c.Module):
    whitelist = ['call', 'call_many']
    def __init__(self, model='google/gemini-pro' , prompt_path=None, **kwargs):
        self.model = c.module('model.openrouter')(model=model)
        self.prompt_path = self.dirpath() + '/prompt.txt'

    @property
    def prompt(self):
        prompt = c.get_text(self.prompt_path)   
        return prompt   

    def call(self, text, model='google/gemini-pro', **kwargs) -> int:
        text = ' '.join(text)
        prompt = self.prompt + text
        return self.model.forward(prompt, model=model, **kwargs)
    
    def call_many(self, text, n=10, **kwargs):
        futures = []
        for i in range(n): 
            kwargs['text'] = ' '.join(text)
            futures += [c.submit(self.call, kwargs=kwargs)]
        return c.wait(futures, timeout=10)
    

    def app(self):
        return c.m('basi.app')().app()

    
Basi.run(__name__)