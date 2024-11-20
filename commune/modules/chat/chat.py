import commune as c
import os

class Chat(c.Module):

    def __init__(self, 
                 max_tokens=420000, 
                 prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.',
                 model = None,
                 history_path='history',
                **kwargs):
        
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.model = c.module('model.openrouter')(model=model, **kwargs)
        self.history_path = self.resolve_path(history_path)

    def generate(self,  text = 'whats 2+2?' , model= 'anthropic/claude-3.5-sonnet',  temperature= 0.5, max_tokens= 1000000,stream=True,  ):
        text = self.process_text(text)
        return self.model.generate(text, stream=stream, model=model, max_tokens=max_tokens,temperature=temperature )
    
    forward = generate

    def ask(self, *text, **kwargs): 
        text = ' '.join(list(map(str, text)))
        return self.generate(text, **kwargs)

    def process_text(self, text):
        new_text = ''
        for word in text.split(' '):
            if any([word.startswith(ch) for ch in ['.', '~', '/']]) and os.path.exists(word):
                word = c.file2text(word)
                print(word.keys())
            new_text += str(word)
        return new_text

    
    def summarize(self, path='./', max_chars=10000): 
        if c.module_exists(path):
            c.print(f'Summarizing Module: {path}')
            text = c.code(path)
        elif os.path.isdir(path):
            c.print(f'Summarizing DIRECTORY: {path}')
            paths = c.ls(path) 
            for p in paths:
                return self.summarize(p)
        elif os.path.isfile(path):
            c.print(f'Summarizing File: {path}')
            text = c.file2text(path)
        prompt = f'''
        GOAL
        summarize the following into tupples 
        CONTEXT
        {text}
        OUTPUT
        '''
        return c.ask(prompt)



    def models(self):
        return self.model.models()