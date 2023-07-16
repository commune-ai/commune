import openai
import os
import torch
from typing import Union, List, Any, Dict
import commune as c
import json
# class OpenAILLM(c.Module):


class OpenAILLM(c.Module):
    
    prompt = """{x}"""

    whitelist = ['forward', 'chat', 'ask']
    
    def __init__(self, 
                 config: Union[str, Dict[str, Any], None] = None,
                 password:str='whadupfam',
                 tag : str = None,
                 load: bool = True,
                 save: bool = True,
                 api: str = None,
                **kwargs
                ):
        
        
        config = self.set_config(config, kwargs=locals())
        self.set_tag(tag)
        self.set_stats(config.stats)
        self.set_api(api)
        self.set_prompt(config.get('prompt', self.prompt))
        self.set_tokenizer(config.tokenizer)
        
        
        self.params  = dict(
                 model =self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
        )
        
        if save:
            self.save(tag=self.tag)

        
    def set_stats(self, stats):
        if stats == None:
            stats = {}
        assert isinstance(stats, dict)
        self.stats = stats 


    def resolve_state_path(self, tag):
        tag = self.resolve_tag(tag)
        return os.path.join('states', f'{tag}.json')
    
    
    def save(self, tag=None):
        path = self.resolve_state_path(tag)
        self.config.stats = self.stats
        self.put(path, self.config)

    def load(self, tag=None):
        path = self.resolve_state_path(tag)
        config = self.get(os.path.join('states', f'{tag}.json'), self.config)
        return config
        
    @classmethod
    def resolve_api_key(cls, api_key:str = None) -> str:
        if api_key == None:
            api_keys = cls.api_keys()
            assert len(api_keys) > 0, "No API keys found"
            api_key = c.choice(api_keys)

        if isinstance(api_key, str):
            api_key = os.getenv(api_key, api_key)
        assert isinstance(api_key, str),f"API Key must be a string,{api_key}"
    
        return api_key

    random_api_key = resolve_api_key
    
    def set_api(self, api: str = None) -> str:
        api = self.resolve_api_key(api)
        openai.api_key  =  os.getenv(api, api)
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
    
    def ask(self, question:str, max_tokens:int = 1000, ) : 
        prompt = """

        Given a question, answer it as a json
        history
        Question: {question}
        """
        
        return self.forward(question=question, 
                            role='user', 
                            max_tokens=max_tokens,
                             prompt=prompt)


    def create(self, *args, **kwargs):
        
        try:

            response = openai.ChatCompletion.create(
                *args,
                **kwargs
            )
            return response
        except Exception as e:
            c.print(e)
            self.params['model'] = c.choice(self.config.models)
            openai.api_key = self.random_api_key()

        
            

        
    def forward(self,
                *args,
                prompt:str = None,
                # params
                model:str = None,
                presence_penalty:float = None, 
                frequency_penalty:float = None,
                temperature:float = None, 
                max_tokens:int = None, 
                top_p:float = None,
                role = 'user',
                add_history : bool = False,
                return_json : bool = False,
                choice_idx:int = 0,

                **kwargs) -> str:


        prompt = self.resolve_prompt(*args, prompt=prompt, **kwargs)
        params = self.resolve_params(locals())
        messages = [{"role": role, "content": prompt}]
        response = self.create(messages=messages, **params)
        # update token stats
        for k,v in response['usage'].items():
            self.stats[k] = self.stats.get(k, 0) + v
            
        response = response['choices'][choice_idx]['message']

        if return_json and c.jsonable(response):
            response = json.loads(response)
        
        if add_history:
            self.history = self.history +  [*messages,response]
            self.save()
            
        # c.stwrite(self.history)
        return response['content']

    def resolve_params(self, params = None):
        if params == None:
            params = {}
        params = c.locals2kwargs(params)
        output_params = {}
        for p in self.params:
            if p in  self.params:
                if params.get(p) == None:
                    output_params[p] = self.params[p]
                else:
                    assert isinstance(params[p], type(self.params[p])), f"Parameter {p} must be of type {type(self.params[p])}, not {type(params[p])}"
                    output_params[p] = params[p]
            
        return output_params


    call = forward
        
        
        
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


    api_key_path = 'api_keys'

    @classmethod
    def add_api_key(cls, api_key, k=api_key_path):
        api_keys = cls.get(k, [])
        assert api_key in api_keys, f"API key {api_key} not added"
        assert cls.verify_api_key(api_key), f"Invalid API key {api_key}"
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put(k, api_keys)
        return {'msg': f'added api_key {api_key}'}


    @classmethod
    def add_api_keys(cls, keys):
        for k in keys:
            cls.add_api_key(k)


    @classmethod
    def set_api_keys(cls, api_keys, k=api_key_path):
        assert isinstance(api_keys, list)
        cls.put(k, api_keys)
        return {'msg': f'added api_key {api_keys}'}

    @classmethod
    def rm_api_key(cls, api_key, k=api_key_path):
        api_keys = []
        found_key = False
        for i, api_k in enumerate(cls.get('api_keys', [])):
            if api_key != api_k:
                api_keys.append(api_key)
            else:
                found_key = True
        # cls.set_api_keys(api_keys)

        if not found_key:
            return {'msg': f'api_key {api_key} not found', 'api_keys': api_keys}
        else:
            cls.set_api_keys(api_keys)
            return {'msg': f'removed api_key {api_key}', 'api_keys': api_keys}


    
    @classmethod
    def valid_api_keys(cls):
        api_keys = cls.api_keys()
        valid_api_keys = []
        for api_key in api_keys:
            if cls.is_valid_api_key(api_key):
                valid_api_keys.append(api_key)
            else:
                c.print('Invalid API key: ' + api_key, color='red')

        return valid_api_keys
                

    @classmethod
    def num_valid_api_keys(cls):
        return len(cls.valid_api_keys())

    @classmethod
    def api_keys(cls):
        return cls.get('api_keys', [])


    def check_api_keys(self):
        verify_api_keys = self.valid_api_keys()
        self.set_api_keys(verify_api_keys)
        return {'msg': f'Verified {len(verify_api_keys)} api_keys', 'api_keys': verify_api_keys}

        
    
    @classmethod
    def test(cls,
            module=None,
            input = 'What is the meaning of life?',
            model = 'gpt3.5-turbo'
    ):
        if module == None:
            module = cls()

        for i in range(10):
            c.print(module.ask(input))


    
    @classmethod
    def is_valid_api_key(cls, api_key:str):
        model = cls(api=api_key)
        try:
            output = model.forward('What is the meaning of life?', max_tokens=4)
        except Exception as e:
            c.print(str(e), color='red')
            return False
        return True
    verify_api_key = is_valid_api_key 

    @classmethod
    def restart_miners(cls, *args,**kwargs):
        for m in cls.miners(*args, **kwargs):
            c.restart(m)
         
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


