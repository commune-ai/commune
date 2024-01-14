import openai
import os
import torch
from typing import Union, List, Any, Dict
import commune as c
import json
# class OpenAILLM(c.Module):


class OpenAILLM(c.Module):
    
    prompt = """{x}"""

    whitelist = ['generate']
    
    def __init__(self, api_key = None,
                max_tokens : int= 250,
                max_stats: int= 175,
                tokenizer: str= 'gpt2',
                models: List[str]= ['gpt-3.5-turbo-0613', 'gpt-3.5-turbo'],
                save:bool = False,
                prompt:bool = None,
                max_input_tokens: int = 10_000_000,
                max_output_tokens: int = 10_000_000,
                **kwargs
                ):
        self.set_config(kwargs=locals())
        self.output_tokens = 0
        self.input_tokens = 0
        
        self.birth_time = c.time()
        self.set_api_key(api_key)
        self.set_prompt(prompt)
        self.set_tokenizer(tokenizer)
        # self.test()
        
    @property
    def age(self):
        return c.time() - self.birth_time
        
    @property
    def too_many_tokens(self ):
        too_many_output_tokens = self.output_tokens > self.config.max_output_tokens
        too_many_input_tokens = self.input_tokens > self.config.max_input_tokens
        return bool(too_many_output_tokens or too_many_input_tokens)
    
    @property
    def input_tokens_per_hour(self):
        if self.age % 3600 == 0:
            self.input_tokens =  0
        return  self.input_tokens / self.age * 3600

    @property
    def output_tokens_per_hour(self):
        if self.age % 3600 == 0:
            self.output_tokens =  0
        return  self.output_tokens / self.age * 3600
    

    hour_limit_count = {}
    def ensure_token_limit(self, input:str , output:str ):
        text = input + output
        tokens = self.tokenizer(text)['input_ids']
        hour = c.time() // 3600
        if hour not in self.hour_limit_count:
            self.hour_limit_count[hour] = 0


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
    
    

    def is_error(self, response):
        return 'error' in response

    def is_success(self, response):
        return not self.is_error(response)

    def call(self, text):
        return self.forward(text, role='user')
    
    def generate(self,
                prompt:str = 'sup?',
                model:str = 'gpt-3.5-turbo',
                presence_penalty:float = 0.0, 
                frequency_penalty:float = 0.0,
                temperature:float = 0.9, 
                max_tokens:int = 100, 
                top_p:float = 1,
                choice_idx:int = 0,
                api_key:str = None,
                role:str = 'user',
                history: list = None,
                **kwargs) -> str:
        t = c.time()
        if not model in self.config.models:
            f"Model must be one of {self.config.models}"
        
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

        assert self.too_many_tokens == False, f"Too many tokens, {self.input_tokens} input tokens and {self.output_tokens} output tokens where generated and the limit is {self.config.max_input_tokens} input tokens and {self.config.max_output_tokens} output tokens"

        response = openai.ChatCompletion.create(messages=messages, **params)
            
        output_text = response = response['choices'][choice_idx]['message']['content']
        input_tokens = self.num_tokens(prompt)
        output_tokens = self.num_tokens(output_text)

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        latency = c.time() - t

        stats = {
            'prompt': prompt,
            'response': output_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'latency': latency,
            'history': history,
            'timestamp': t,
        }

        # self.add_stats(tag=t, stats=stats)

        return output_text

    _stats = None
    _stats_update_time = 0

    @classmethod
    def stats(cls, skip_keys = ['prompt', 'response', 'history'], refresh_interval=5):
        if cls._stats != None or c.time() % refresh_interval > (c.time() - cls._stats_update_time):
            stat_paths = cls.ls('stats')
            cls._stats = [cls.get(path) for path in stat_paths]
            cls._stats_update_time = c.time()
        if cls._stats == None:
            cls._stats = stats
        stats = [{k:v for k,v in cls.get(path).items() if k not in skip_keys} for path in stat_paths]

        return  stats
    
    
    @classmethod
    def tokens_per_period(cls, timescale='m'):
        stats = cls.stats()

        if timescale == 's':
            period = 1
        elif timescale == 'm':
            period = 60
        elif timescale == 'h':
            period = 3600
        else:
            raise NotImplemented(timescale)
        

        
        one_hour_ago = c.time() - period
        stats = [s for s in stats if s['timestamp'] > one_hour_ago]
        tokens_per_period = sum([s['input_tokens'] + s['output_tokens'] for s in stats])
        return tokens_per_period

    def add_stats(self, tag:str, stats:dict,  ):
        self.put(f'stats/{tag}.json', stats)
        saved_stats_paths = self.ls('stats')
        if len(saved_stats_paths) > self.config.max_stats:
            # remove the oldest stat
            sorted(saved_stats_paths, key=lambda x: int(x.split('.')[0]))
            self.rm(saved_stats_paths[0])

        return {'msg': f"Saved stats for {tag}", 'success': True}

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



    def num_tokens(self, text:str) -> int:
        num_tokens = 0
        tokens = self.tokenizer.encode(text)
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            for i, token in enumerate(tokens):
                num_tokens += len(token)
        else:
            num_tokens = len(tokens)
        return num_tokens
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

    def decode_tokens(self,input_ids: Union[torch.Tensor, List[int]], **kwargs) -> Union[str, List[str], torch.Tensor]:
        return self.tokenizer.decode(input_ids, **kwargs)
    
    def encode_tokens(self, 
                 text: Union[str, List[str], torch.Tensor], 
                 return_tensors='pt', 
                 padding=True, 
                 truncation=True, 
                 max_length=256,
                 **kwargs):
        
        return self.tokenizer(text, 
                         return_tensors=return_tensors, 
                         padding=padding, 
                         truncation=truncation, 
                         max_length=max_length)

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

