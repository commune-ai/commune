import commune as c
import os
import json

class Chat(c.Module):
    description = "This module is used to chat with an AI assistant"
    def __init__(self, 
                 max_tokens=420000, 
                 prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.',
                 model = None,
                **kwargs):
        
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.model = c.module('model.openrouter')(model=model, **kwargs)

    def generate(self,  text = 'whats 2+2?' , model= 'anthropic/claude-3.5-sonnet',  temperature= 0.5, max_tokens= 1000000,stream=True,  ):
        text = self.process_text(text)
        return self.model.generate(text, stream=stream, model=model, max_tokens=max_tokens,temperature=temperature )
    
    forward = generate

    def ask(self, *text, **kwargs): 
        text = ' '.join(list(map(str, text)))
        return self.generate(text, **kwargs)
    
    def process_text(self, text, threshold=1000):
        new_text = ''
        for word in text.split(' '):
            conditions = {
                "file": any([word.startswith(ch) for ch in ['.', '~', '/']]),
                "module": word.startswith('c_') and c.module_exists(word.split('_')[1]),
            }
            if conditions['file']:
                word = c.file2text(word)
            new_text += str(word)
        return new_text
 
    def reduce(self, text, max_chars=10000 , timeout=5, max_age=30, model='openai/o1-mini'):
        
        if os.path.exists(text): 
            text = str(c.file2text(text))
        elif c.module_exists(text):
            text = c.code(text)

        original_length = len(text)
        code_hash = c.hash(text)
        path = f'summary/{code_hash}' 

        text = f'''
        GOAL
        summarize the following into tupples and make sure you compress as much as oyu can
        CONTEXT
        {text}
        OUTPUT FORMAT ONLY BETWEEN THE TAGS SO WE CAN PARSE
        <OUTPUT>DICT(data=List[Dict[str, str]])</OUTPUT>
        '''
        print(f"TEXTSIZE : {len(text)}")
        compress_ratio = 0
        text_size = len(text)
        if len(text) >= max_chars * 2 :
            batch_text = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
            print(f"TEXTSIZE : {text_size} > {max_chars} BATCH SIZE: {len(batch_text)}")
            futures =  [c.submit(self.reduce, [batch], timeout=timeout) for batch in batch_text]
            text = ''
            cnt = 0
            try:
                n = len(batch_text)
                progress = c.progress(n)
                
                for future in c.as_completed(futures, timeout=timeout):
                    text += str(future.result())
                    cnt += 1
                    progress.update(1)
                    print(f"SUMMARIZED: {cnt}/{n} COMPRESSION_RATIO: {compress_ratio}")
                return text
            except Exception as e:
                print(e)
            
            final_length = len(text)
            compress_ratio = final_length/original_length
            result = { 'compress_ratio': compress_ratio, 'final_length': final_length, 'original_length': original_length}
            print(result)
            return text
        if "'''" in text:
            text = text.replace("'''", '"""')
        
        data =  c.ask(text, model=model, stream=0)
        return data

    def models(self):
        return self.model.models()
    