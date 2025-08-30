import commune as c
import os
import json

class Agent:

    prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.'
    def __init__(self, 
                 provider='model.openrouter', 
                 model = 'anthropic/claude-opus-4',
                **kwargs):
        self.provider = c.module(provider)(**kwargs)
        self.model = model
        self.model2info = self.provider.model2info
        self.models = self.provider.models

    def forward(self, text = 'whats 2+2?' ,  
                    temperature= 0.5,
                    max_tokens= 1000000, 
                    preprocess=True,
                    model='anthropic/claude-opus-4', 
                    stream=True,
                    **kwargs):
        if preprocess:
            text = self.preprocess(text)
        params = {
                'message': text, 
                'temperature': temperature, 
                'max_tokens': max_tokens, 
                'model': model,
                'stream': stream,
            **kwargs}
        tx = {
            'params': params,
            'model': model or self.model,
        }
        tx_id = c.hash(tx)
        print('tx_id', tx_id)
        result =  self.provider.forward(**params)
        



        
        return self.postprocess(result)

    def ask(self, *text, **kwargs): 
        return self.forward(' '.join(list(map(str, text))), **kwargs)

    def preprocess(self, text, threshold=1000):
            new_text = ''
            is_function_running = False
            words = text.split(' ')
            fn_detected = False
            fns = []
            for i, word in enumerate(words):
                prev_word = words[i-1] if i > 0 else ''
                # restrictions can currently only handle one function argument, future support for multiple
                magic_prefix = f'@'
                if word.startswith(magic_prefix) and not fn_detected:
                    word = word[len(magic_prefix):]
                    fns += [{'fn': word, 'params': [], 'idx': i + 2}]
                    fn_detected=True
                else:
                    if fn_detected:
                        fns[-1]['params'] += [word]
                        fn_detected = False
            c.print(fns)
            for fn in fns:
                print('Running function:', fn)
                result = c.fn(fn['fn'])(*fn['params'])
                fn['result'] = result
                print('Result:', result)
                text =' '.join([*words[:fn['idx']],'-->', str(result), *words[fn['idx']:]])
            return text
        
    def postprocess(self, result):
        return result