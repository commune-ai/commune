import openai
import os
import torch
from typing import Union, List, Any, Dict
import commune as c
import json
# class OpenAILLM(c.Module):


class OpenAILLM(c.Module):
    
    prompt = """{x}"""

    
    def __init__(self, 
                 config: Union[str, Dict[str, Any], None] = None,
                 password=None,
                **kwargs
                ):
        
        
        
        config = self.set_config(config, kwargs=kwargs)
        self.set_tag(config.tag)
        self.set_password(password)
        self.set_api(api=config.api, password=self.password)
        self.set_prompt(config.get('prompt', self.prompt))
        self.set_tokenizer(config.tokenizer)
        self.set_stats(config.stats)
        
        self.params  = dict(
                 model =self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
        )
        

        
    def set_stats(self, stats):
        if stats == None:
            stats = {}
        assert isinstance(stats, dict)
        self.stats = stats 
    
    
    def save(self, tag=None):
        tag = self.resolve_tag(tag)
        self.config.stats = self.stats
        self.put(tag, self.config)
        
    def set_password(self, password):
        self.password = password
    def resolve_password(self, password = None):
        if password == None:
            password = self.password
        return password
        
    def load(self, tag=None, password = None):
        tag = self.resolve_tag(tag)
        password = self.resolve_password(password)
        
        config = self.get(tag, self.config)
        self.init(self, config=config, password=password)
        
        
        
    def set_api(self, api: str, password:str=None) -> None:
        if self.is_encrypted(api):

            api = c.decrypt(api, password=password)
            
            print('decrypting api', api, password)
            
        assert isinstance(api, str), "API Key must be a string"
        openai.api_key  =  os.getenv(api, api)
        
        self.putc('api', openai.api_key, password=password)
        
    
    
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
    
    
    def resolve_params(self, params):
        resolved_params = {}
        for k,v in params.items():
            if k in self.params:
                assert isinstance(params[k], type(v)), f"Parameter {k} must be of type {type(v)}"
                resolved_params[k] = v
        return {**self.params , **resolved_params}
    
        

        
    def forward(self,
                *args,
                prompt:str=None,
                role = 'user',
                choice_idx:int = 0,
                add_history = False,
                save_history = False,
                **kwargs) -> str:
        

        prompt = self.resolve_prompt(*args, prompt=prompt, **kwargs)
        params = self.resolve_params(kwargs)
        messages = [{"role": role, "content": prompt}]
        response = openai.ChatCompletion.create(
            messages=messages,
            **params
        )
        
        # update token stats
        for k,v in response['usage'].items():
            self.stats[k] = self.stats.get(k, 0) + v
            
        response = response['choices'][choice_idx]['message']

        # if c.jsonable(response):
        #     response = json.loads(response)
        if add_history:
            self.history = self.history +  [*messages,response]
            self.save()
            
        c.stwrite(self.history)
        return response['content']
            
    def call(self):
        print('bro')
        
        
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
    def test(cls, **params:dict):
        model = cls(**params)
        cls.print(model.forward('What is the meaning of life?'))
        
    @classmethod
    def encapi(cls, password):
        return cls.encryptc('api',password=password)
    @classmethod
    def decapi(cls, password):
        return cls.decryptc('api',password=password)
    
    @classmethod
    def encrypted(cls):
        return cls.is_encrypted('api')
        return cls.decryptc('api',password=password)
         
    def set_tokenizer(self, tokenizer: str):

        if tokenizer == None and hasattr(self, 'tokenizer'):
            return self.tokenizer
             
        if tokenizer == None:
            tokenizer = 'gpt2'
        from transformers import AutoTokenizer

        if isinstance(tokenizer, str):
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
    def st(cls):
        import streamlit as st
        model = cls()
        
        buttons = {}
        st.write(c.python2types(model.__dict__))
        response = 'bro what is up?'
        prompt = '''
        {x}
        Document this in a markdown format that i can copy
        '''
        
        
        st.write(model.forward(model.fn2str()['forward'], prompt=prompt, max_tokens=1000))
        
        
        
        # for i in range(10):
        #     response = model.forward(prompt='What is the meaning of life?', max_tokens=1000)
        #     st.write(response, model.stats)
        # st.write(model.forward(prompt='What is the meaning of life?'))
        # model.save()
        # model.test()
        # st.write('fuckkkkffffff')
        
        
        
    
        
if __name__ == '__main__':
    OpenAILLM.run()


