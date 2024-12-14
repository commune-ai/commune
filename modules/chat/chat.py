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

    def ask(self, *text, module = None, **kwargs): 
        text = ' '.join(list(map(str, text)))
        if module != None:
            text = c.code(module) + text
        return self.generate(text, **kwargs)
    
    def process_text(self, text, threshold=1000):
        new_text = ''
        for word in text.split(' '):
            conditions = {
                "file": any([word.startswith(ch) for ch in ['.', '~', '/']]),
                "code": word.startswith('code/'),
                'c': word.startswith('c/')
            }
            if conditions['file']:
                print('READING FILE -->', word)
                word = c.file2text(word)
            if conditions['code']:
                word = word[len('code/'):]
                print('READING MODULE -->', word)
                word = c.code(word)
                
            new_text += str(word)
        return new_text
    
 
    def reduce(self, text, max_chars=10000 , timeout=40, max_age=30, model='openai/o1-mini'):
        if os.path.exists(text): 
            path = text
            if os.path.isdir(path):
                print('REDUCING A DIRECTORY -->', path)
                future2path = {}
                path2result = {}
                paths = c.files(path)
                progress = c.tqdm(len(paths), desc='Reducing', leave=False)
                while len(paths) > 0:
                    for p in paths:
                        future = c.submit(self.reduce, [p], timeout=timeout)
                        future2path[future] = p
                    try:
                        for future in c.as_completed(future2path, timeout=timeout):
                            p = future2path[future]
                            r = future.result()
                            paths.remove(p)
                            path2result[p] = r
                            print('REDUCING A FILE -->', r)
                            progress.update(1)
                    except Exception as e:
                        print(e)
                return path2result
            else:
                assert os.path.exists(path), f'Path {path} does not exist'
                print('REDUCING A FILE -->', path)
                text = str(c.get_text(path))
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
        if len(text) >= max_chars * 2 :
            batch_text = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
            futures =  [c.submit(self.reduce, [batch], timeout=timeout) for batch in batch_text]
            output = ''
            try:
                for future in c.as_completed(futures, timeout=timeout):
                    output += str(future.result())
            except Exception as e:
                print(e)
            final_length = len(text)
            result = { 'compress_ratio': final_length/original_length, 
                      'final_length': final_length, 
                      'original_length': original_length, 
                      "data": text}
            return result
        if "'''" in text:
            text = text.replace("'''", '"""')
        
        data =  c.ask(text, model=model, stream=0)
        def process_data(data):
            try:
                data = data.split('<OUTPUT>')[1].split('</OUTPUT>')[0]
                return data
            except:
                return data
        return {"data": process_data(data)}

    def models(self):
        return self.model.models()
    