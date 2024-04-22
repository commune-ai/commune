import openai
from typing import *
import commune as c

class UsageTracker:
    def __init__(self, tokenizer='gpt2',
                  max_output_tokens=10_000_000,
                  max_input_tokens=10_000_000, 
                 **kwargs):

        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens

        self.set_tokenizer(tokenizer)
        self.usage = {
            'input': 0,
            'output': 0,
        }
        self.start_time = c.time()

    
    @property
    def age(self):
        return c.time() - self.start_time
    
    def usage_check(self ):
        too_many_output_tokens = self.usage['output'] < self.max_output_tokens
        too_many_input_tokens = self.usage['input'] < self.max_input_tokens
        return bool(too_many_output_tokens and too_many_input_tokens)
    

    def register_tokens(self, prompt:str, mode='input'):
        if not isinstance(prompt, str):
            prompt = str(prompt)
        input_tokens = self.num_tokens(prompt)
        self.usage[mode] += input_tokens

        assert self.usage_check(), \
                f"Too many tokens,output: {self.max_input_tokens} {self.max_output_tokens} output tokens, {self.usage}"
    
        return {'msg': f"Registered {input_tokens} {mode} tokens", 'success': True}
    


    def set_tokenizer(self, tokenizer: str = 'gpt2'):
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
        except ValueError:
            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        tokenizer.pad_token = tokenizer.eos_token 
            
        self.tokenizer = tokenizer
    
        return self.tokenizer

    def num_tokens(self, text:str) -> int:
        num_tokens = 0
        tokens = self.tokenizer.encode(text)
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            for i, token in enumerate(tokens):
                num_tokens += len(token)
        else:
            num_tokens = len(tokens)
        return num_tokens



class OpenAILLM(c.Module):

    libs = ['openai', 'transformers']
    
    prompt = """{x}"""

    whitelist = ['generate']
    
    def __init__(self, api_key = None,
                max_tokens : int= 250,
                max_stats: int= 175,
                tokenizer: str= 'gpt2',
                save:bool = False,
                prompt:bool = None,
                max_input_tokens: int = 10_000_000,
                max_output_tokens: int = 10_000_000,
                **kwargs
                ):
        

        self.set_config(kwargs=locals())
        self.usage = UsageTracker(tokenizer=tokenizer, max_output_tokens=max_output_tokens, max_input_tokens=max_input_tokens)
        self.birth_time = c.time()
        self.set_api_key(api_key)
        self.set_prompt(prompt)
        # self.test()
        
    @property
    def age(self):
        return c.time() - self.birth_time
        

    @classmethod
    def random_api_key(cls):
        api_keys = cls.api_keys()
        assert len(api_keys) > 0, "No valid API keys found, please add one via ```c openai add_api_key <api_key>```"
        api_key = c.choice(api_keys)

        return api_key
  
    def set_api_key(self, api_key: str = None) -> str:
        if api_key==None and  len(self.keys()) > 0 :
            api_key = self.random_api_key()
        self.api_key = api_key
        openai.api_key = self.api_key
        return {'msg': f"API Key set to {openai.api_key}", 'success': True}

    def resolve_prompt(self, *args, prompt = None, **kwargs):
        if prompt == None:
            prompt = self.prompt
            prompt_variables  = self.prompt_variables
        else:
            assert isinstance(prompt, str)
            prompt_variables = self.get_prompt_variables(prompt)
        
                    
        if len(args) > 0 :
            assert len(args) == len(prompt_variables), f"Number of arguments must match number of prompt variables: {self.prompt_variables}"
            kwargs = dict(zip(prompt_variables, args))

        for var in prompt_variables:
            assert var in kwargs

        prompt = prompt.format(**kwargs)
        return prompt
    
    
    def generate(self,
                prompt:str = 'sup?',
                model:str = 'gpt-4-vision-preview',
                presence_penalty:float = 0.0, 
                frequency_penalty:float = 0.0,
                temperature:float = 0.9, 
                max_tokens:int = 4096, 
                top_p:float = 1,
                choice_idx:int = 0,
                api_key:str = None,
                role:str = 'user',
                history: list = None,
                stream =  False,
                **kwargs) -> str:
        
        t = c.time()

        self.usage.register_tokens(prompt, mode='input')

        openai.api_key = api_key or self.api_key

        params = dict(
                    model = model,
                    presence_penalty = presence_penalty, 
                    frequency_penalty = frequency_penalty,
                    temperature = temperature, 
                    max_tokens = max_tokens, 
                    top_p = top_p
                    )
        
        messages = [{"role": role, "content": prompt}]
        
        if history:
            messages = history + messages

    
        response = openai.chat.completions.create(messages=messages, stream=stream, **params)
        
        if stream:
            def stream_response(response):
                for r in response:
                    token = r.choices[choice_idx].delta.content
                    self.usage.register_tokens(token, mode='output')
                    yield token
            return stream_response(response)
        else:
            output_text = response = response.choices[choice_idx].message.content
            self.usage.register_tokens(output_text, mode='output')
            return output_text

    
    

    
    
    _stats = None
    _stats_update_time = 0


    forward = call = generate

    @classmethod
    def chat(cls, *args, **kwargs):
        return cls().forward(*args, **kwargs)
        
    @property
    def history(self):
        return self.config.get('history', [])
    @history.setter
    def history(self, history):
        self.config['history'] = history

    def set_prompt(self, prompt: str):
        if prompt == None:
            prompt = self.prompt
        self.prompt = prompt
        assert isinstance(self.prompt, str), "Prompt must be a string"
        self.prompt_variables = self.get_prompt_variables(self.prompt)

        
    @staticmethod   
    def get_prompt_variables(prompt):
        variables = []
        tokens = prompt.split('{')
        for token in tokens:
            if '}' in token:
                variables.append(token.split('}')[0])
        return variables



    @classmethod
    def test(cls, input:str = 'What is the meaning of life?',**kwargs):
        module = cls()
        output = module.generate(input)
        assert isinstance(output, str)
        return {'success': True, 'msg': 'test'}

    
    @classmethod
    def verify_api_key(cls, api_key:str, text:str='ping', verbose:bool = True):
        model = cls(api_key=api_key)
        output = model.forward(text, max_tokens=1, api_key=api_key, retry=False)
        if 'error' in output:
            c.print(f'ERROR \u2717 -> {api_key}', output['error'], color='red', verbose=verbose)
            return False
        else:
            # checkmark = u'\u2713'
            c.print(f'Verified \u2713 -> {api_key} ', output, color='green', verbose=verbose)
        return True
    

    @classmethod
    def validate(cls, text = 'What is the meaning of life?', max_tokens=10):
        prefix = cls.module_path()
        jobs = []
        servers = c.servers(prefix)
        for s in servers:
            job = c.call(module=s, 
                         fn='forward', 
                         text=text, 
                         temperature=0.0,
                         max_tokens=max_tokens,
                        return_future=True
                        )
            jobs.append(job)
        assert len(jobs) > 0, f'No servers found with prefix {prefix}'
        results = c.gather(jobs)
        response = {}
        for s, result in zip(c.servers(prefix), results):
            response[s] = result

        return response
   
    
    @classmethod
    def add_key(cls, api_key:str):
        assert isinstance(api_key, str), "API key must be a string"
        api_keys = list(set(cls.get('api_keys', []) + [api_key]))
        cls.put('api_keys', api_keys)
        return {'msg': f"API Key set to {api_key}", 'success': True}
    
    @classmethod
    def rm_key(cls, api_key:str):
        new_keys = []
        api_keys = cls.api_keys()
        for k in api_keys: 
            if api_key in k:
                continue
            else:
                new_keys.append(k)
        cls.put('api_keys', new_keys)
        return {'msg': f"Removed API Key {api_key}", 'success': True}
                
    
    @classmethod
    def api_keys(cls):
        return  cls.get('api_keys', [])
    
    @classmethod
    def save_api_keys(cls, api_keys:List[str]):
        cls.put('api_keys', api_keys)
        return {'msg': f"Saved API Keys", 'success': True}
   
    def embed(self, text:str, **kwargs):
        pass