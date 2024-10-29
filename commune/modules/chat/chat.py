import commune as c
import os

class Chat(c.Module):

    def __init__(self, 
                 max_tokens=420000, 
                 password = None,
                 prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.',
                 name='chat',
                 model = None,
                 history_path='history',
                **kwargs):

        self.max_tokens = max_tokens
       
        self.set_module(model, 
                        password = password,
                        name = name,
                        history_path=history_path, 
                        prompt=prompt,
                        **kwargs)
        
    def set_module(self,
                    model, 
                   history_path='history', 
                   name='chat',
                   password=None,
                   prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.',
                   key=None,
                    **kwargs):
        self.prompt = prompt
        self.admin_key = c.pwd2key(password) if password else self.key
        self.model = c.module('model.openrouter')(model=model, **kwargs)
        self.models = self.model.models()
        self.history_path = self.resolve_path(history_path)
        return {'success':True, 'msg':'set_module passed'}

    def call(self, 
            input = 'whats 2+2?' ,
            temperature= 0.5,
            max_tokens= 1000000,
            model= 'anthropic/claude-3.5-sonnet', 
            key = None,
            stream=True, 
            ):
        # key = self.resolve_key(key)
        data = c.locals2kwargs(locals())
        signature = self.key.ticket(c.hash(data))
        return signature
    

    def generate(self,  
            text = 'whats 2+2?' ,
            model= 'anthropic/claude-3.5-sonnet', 
            temperature= 0.5,
            max_tokens= 1000000,
            context = None,
            stream=True, 
            ):
        text = self.process_text(text, context=context)
        output =  self.model.generate(text, stream=stream, model=model, max_tokens=max_tokens,temperature=temperature )
        for token in output:
            yield token
    forward = generate

    def ask(self, *text, **kwargs): 
        return self.generate(' '.join(list(map(str, text))), **kwargs)

    def process_text(self, text, context=None):
        text = self.prompt + text
 
        if context != None:
            context = str(context)
            if c.exists(context):
                context =  str(c.file2text(context))
            elif c.module_exists(context):
                context =  c.code(context)
            text = context + text
        
        return text

    def save_data(self, data):
        path = self.data2path(data)
        return c.put(path, data)
    
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

    def user_files(self):
        return c.get(self.data['path'])

    def save_data(self, address, data):
        return c.put(self.history_path + '/' + address +'/data.json', data)
    
    def get_data(self, address):
        return c.get(self.history_path + '/' + address +'/data.json')

        
    def clear_history(self, address):
        return c.rm(self.history_path +  '/'+ address)
    
    def history_paths(self, address:str=None):
        paths = []
        if address == None:
            for user_address in self.user_addresses():
                 paths += self.history_paths(user_address)
        else:
            paths = c.ls(self.history_path + '/'+ address)
        return paths
    
    def save_data(self, data):
        path = self.history_path + '/'+ data['address'] + '/' + str(data['time']) + '.json'
        return c.put(path, data)
    
    def history(self, address:str=None, columns=['datetime', 
                                                 'input', 
                                                 'output', 
                                                 'prompt',
                                                 'model', 
                                                 'temperature',
                                                   'max_tokens'], df=False):
        paths = self.history_paths(address)
        history =  []
        for i, path in enumerate(paths):
            try:
                print(paths)
                h = c.get(path)
                h.update(h.pop('data'))
                h['datetime'] = c.time2datetime(h.pop('time'))
                h = {k:v for k,v in h.items() if k in columns}
                history.append(h)
            except Exception as e:
                print(e)
        # sort by time
    
        history = sorted(history, key=lambda x: x['datetime'], reverse=True)
        if df:
            history = c.df(history)
        return history
    
    def user_addresses(self, display_name=False):
        users = [u.split('/')[-1] for u in c.ls(self.history_path)]
        return users

    def models(self):
        return self.model.models()

    def pricing(self, *args, **kwargs):
        return self.model.pricing(*args,  **kwargs)
