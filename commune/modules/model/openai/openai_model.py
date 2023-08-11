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
                 **kwargs
                ):
        
        
        config = self.set_config(config, kwargs=kwargs)
        self.set_tag(config.tag)
        self.set_stats(config.stats)
        self.set_api_key(config.api_key)
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
        
        if config.save:
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
        
    def resolve_api_key(self, api_key:str = None) -> str:
        if api_key == None:
            if hasattr(self, 'api_key'):
                api_key = self.api_key
            else:
                api_key = self.random_api_key()

        assert isinstance(api_key, str),f"API Key must be a string,{api_key}"
        openai.api_key = api_key
        return api_key

    @classmethod
    def random_api_key(cls):
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            api_key = 'OPENAI_API_KEY'
            api_key = os.getenv(api_key, api_key)
        else:
            api_key = c.choice(cls.api_keys())
        return api_key
  
    def set_api_key(self, api_key: str = None) -> str:
        self.api_key = self.resolve_api_key(api_key)
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
    
    def ask(self, question:str, max_tokens:int = 1000) : 
        '''
        Ask a question and return the answer
        
        Args:
            question (str): The question to ask
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.
        Returns:
            str: The answer to the question
        '''

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
            return {'error': str(e)}
            self.params['model'] = c.choice(self.config.models)


    def is_error(self, response):
        return 'error' in response

    def is_success(self, response):
        return not self.is_error(response)

    def call(self, text):
        return self.forward(text, role='user')
        
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
                api_key:str = None,

                **kwargs) -> str:
        
        api_key = self.resolve_api_key(api_key)
        prompt = self.resolve_prompt(*args, prompt=prompt, **kwargs)
        params = self.resolve_params(locals())
        response = self.create(messages=[{"role": role, "content": prompt}], **params)
        if self.is_error(response):
            return response
        
        assert 'usage' in response, f"Response must contain usage stats: {response}"
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

    generate = call = forward
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
        c.print(api_keys)
        if api_key in api_keys:
            return {'error': f'api_key {api_key} already added'}
        verified = cls.verify_api_key(api_key)
        if not verified:
            return {'error': f'api_key {api_key} not verified'}
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put(k, api_keys)
        assert api_key in cls.api_keys(), f"API key {api_key} not added"
        return {'msg': f'added api_key {api_key}'}


    @classmethod
    def add_api_keys(cls, *keys):
        for k in keys:
            cls.add_api_key(k)


    @classmethod
    def set_api_keys(cls, api_keys: List[str], k: str=api_key_path):
        assert isinstance(api_keys, list)
        cls.put(k, api_keys)
        return {'msg': f'added api_key {api_keys}'}

    @classmethod
    def rm_api_key(cls, api_key, k=api_key_path):

        api_keys = cls.get('api_keys', [])
        if api_key not in api_keys:
            return {'error': f'api_key {api_key} not found', 'api_keys': api_keys}

        api_idx = None
        for i, api_k in enumerate(api_keys):
            if api_key != api_k:
                api_idx = i
        if api_idx == None:
            return {'error': f'api_key {api_key} not found', 'api_keys': api_keys}

        del api_keys[api_idx]
        cls.set_api_keys(api_keys)

        return {'msg': f'removed api_key {api_key}', 'api_keys': api_keys}

    @classmethod
    def update(cls):
        cls.set_api_keys(cls.valid_api_keys())
    
    @classmethod
    def valid_api_keys(cls, verbose:bool = True):
        api_keys = cls.api_keys()
        valid_api_keys = []
        for api_key in api_keys:
            if verbose:
                c.print(f'Verifying API key: {api_key}', color='blue')
            if cls.is_valid_api_key(api_key):
                valid_api_keys.append(api_key)
        return valid_api_keys
    valid_keys = verify_api_keys = valid_api_keys

    @classmethod
    def num_valid_api_keys(cls):
        return len(cls.valid_api_keys())

    @classmethod
    def api_keys(cls):
        return cls.get('api_keys', [])


    def remove_invalid_api_keys(self):
        verify_api_keys = self.valid_api_keys()
        self.set_api_keys(verify_api_keys)
        return {'msg': f'Verified {len(verify_api_keys)} api_keys', 'api_keys': verify_api_keys}

        
    
    @classmethod
    def test(cls,
            input = 'What is the meaning of life?',**kwargs
    ):
        if module == None:
            module = cls()


        c.print(module.ask(input))


    
    @classmethod
    def is_valid_api_key(cls, api_key:str, text:str='ping'):
        model = cls(api=api_key)
        output = model.forward(text, max_tokens=1, api_key=api_key)
        if 'error' in output:
            c.print(output['error'], color='red')
            return False
        else:
            c.print(f'API key {api_key} is valid {output}', color='green')
        return True
    verify_key = verify_api_key = is_valid_api_key 

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
    # @classmethod
    # def serve(cls, *args, **kwargs):
    #     name = cls.name()
 
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


