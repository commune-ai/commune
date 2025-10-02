from typing import Generator
import requests
import json
import openai
import time
import os
import commune as c
import random

class OpenRouter:
    api_key_path = 'apikeys' # path to store api keys (relative to storage_path)
    env_varname = 'OPENROUTER_API_KEY' # environment variable name for api key
    def __init__(
        self,
        api_key = None,
        url: str = 'https://openrouter.ai/api/v1',
        timeout: float = None,
        prompt:str=None,
        model = 'anthropic/claude-opus-4',
        max_retries: int = 10,
        path = '~/.commune/openrouter',
        key = None,
        **kwargs
    ):
        """
        Initialize the OpenAI with the specified model, API key, timeout, and max retries.

        Args:
            model (OPENAI_MODES): The OpenAI model to use.
            api_key (API_KEY): The API key for authentication.
            url (str, optional): can be used for openrouter api calls
            timeout (float, optional): The timeout value for the client. Defaults to None.
            max_retries (int, optional): The maximum number of retries for the client. Defaults to None.
        """
        self.store = c.mod('store')(path)
        self.url = url
        self.client = openai.OpenAI(
            base_url=self.url,
            api_key=self.api_key(api_key),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.prompt = prompt

    def forward(
        self,
        message: str,
        *extra_text , 
        history = None,
        prompt: str =  None,
        system_prompt: str = None,
        stream: bool = False,
        model:str = None,
        max_tokens: int = 10000000,
        temperature: float = 1.0,
        **kwargs
    ) -> str :
        """
        Generates a response using the OpenAI language model.

        Args:
            message (str): The message to send to the language model.
            history (ChatHistory): The conversation history.
            stream (bool): Whether to stream the response or not.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature to use.

        Returns:
        Generator[str] | str: A generator for streaming responses or the full streamed response.
        """
        model = model or self.model
        prompt = prompt or system_prompt
        if len(extra_text) > 0:
            message = message + ' '.join(extra_text)
        history = history or []
        prompt = prompt or self.prompt
        message = message + prompt if prompt else message
        model = self.resolve_model(model)
        model_info = self.get_model_info(model)
        num_tokens = len(message)
        print(f'Sending {num_tokens} tokens -> {model}')
        max_tokens = min(max_tokens, model_info['context_length'] - num_tokens)
        messages = history.copy()
        messages.append({"role": "user", "content": message})
        result = self.client.chat.completions.create(model=model, messages=messages, stream= bool(stream), max_tokens = max_tokens, temperature= temperature  )

        item = {
            'model': model,
            'params': {
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
            },
            'time': time.time(),  
        }
        
        item['hash'] = c.hash(item)
        item['result'] = ''
        path = f"history/{item['hash']}"
        if stream:
            def stream_generator( result):
                for token in result:
                    token = token.choices[0].delta.content
                    item['result'] += token
                    yield token
                self.store.put(path, item)
            return stream_generator(result)
        else:
            item['result'] = result.choices[0].message.content
            self.store.put(path, item)
            return item['result']
        
    generate = forward

    def history(self, path:str = None, max_age:int = 0, update:bool = False):
        """
        Get the history of the last requests
        """
        history = self.store.items('history', max_age=max_age, update=update)
        return history

    def resolve_model(self, model=None):
        model = model or self.model
        models =  self.models()
        model = str(model)
        if str(model) not in models:
            if ',' in model:
                models = [m for m in models if any([s in m for s in model.split(',')])]
            else:
                models = [m for m in models if str(model) in m]
            print(f"Model {model} not found. Using {models} instead.")
            assert len(models) > 0
            model = models[0]

        return model

    def api_key(self, api_key: str = None):
        """
        get the api keys
        """
        api_key = api_key or 'OPENROUTER_API_KEY'
        env_varname = self.env_varname
        # first check environment variable
        if env_varname in os.environ:
            return os.environ[env_varname]

        keys = self.store.get(self.api_key_path, [])
        if len(keys) > 0:
            return random.choice(keys)
        else:
            return ''
            print(f"No API key found in store.")

    def keys(self):
        """
        Get the list of API keys
        """
        return self.store.get(self.api_key_path, [])

    def add_key(self, key):
        keys = self.store.get(self.api_key_path, [])
        keys.append(key)
        keys = list(set(keys))
        self.store.put(self.api_key_path, keys)
        return keys

    def rm_key(self, key):
        keys = self.store.get(self.api_key_path, [])
        keys = [k for k in keys if k != key]
        self.store.put(self.api_key_path, keys)
        return keys

    def model2info(self, search: str = None, path='models', max_age=100, update=False):
        models = self.store.get(path, default={}, max_age=max_age, update=update)
        if len(models) == 0:
            print('Updating models...')
            response = requests.get(self.url + '/models')
            models = json.loads(response.text)['data']
            self.store.put(path, models)
        models = self.filter_models(models, search=search)
        return {m['id']:m for m in models}
    
    def models(self, search: str = None, path='models', max_age=60, update=False):
        return list(self.model2info(search=search, path=path, max_age=max_age, update=update).keys())

    
    def model_infos(self, search: str = None, path='models', max_age=0, update=False):
        return list(self.model2info(search=search, path=path, max_age=max_age, update=update).values())
    
    def get_model_info(self, model):
        model = self.resolve_model(model)
        model2info = self.model2info()
        return model2info[model]
    
    @classmethod
    def filter_models(cls, models, search:str = None):
        if search == None:
            return models
        if isinstance(models[0], str):
            models = [{'id': m} for m in models]
        if ',' in search:
            search = [s.strip() for s in search.split(',')]
        else:
            search = [search]
        models = [m for m in models if any([s in m['id'] for s in search])]
        return [m for m in models]
    
    def pricing(self, search: str = None , ascending=False, sortby='completion', df=True,   **kwargs):
        pricing =  [{'name': k , **v['pricing']} for k,v in self.model2info(search=search, **kwargs).items()]
        if df:
            return c.df(pricing).sort_values(sortby, ascending=ascending)
        return pricing

    def get_token_count(self, text: str):
        # rough estimate of token count
        return len(text.split()) // 0.75


    def get_cost(self, params: dict, result:dict,  model: str = None):


        if 'messages' in params: # chat model
            text = params.get('messages', '')
            model = self.resolve_model(model)
            model_info = self.get_model_info(model)
            pricing = model_info['pricing']
            tokens = self.get_token_count(text)
            cost = tokens * pricing['completion'] + tokens * pricing['prompt']
            return cost
        elif 'prompt' in params: # text model
            text = params.get('prompt', '')
            model = self.resolve_model(model)
            model_info = self.get_model_info(model)
            pricing = model_info['pricing']
            tokens = self.get_token_count(text)
            cost = tokens * pricing['completion'] + tokens * pricing['prompt']
            return cost
    
        else:
            raise ValueError('Cannot compute cost for non-chat models yet.')

    
    def test(self):
        response  =  self.forward('Hello, how are you?', stream=False)
        print(response)
        assert isinstance(response, str)
        print('Test passed')
        stream_response = self.forward('Hello, how are you?', stream=True)
        print(next(stream_response))
        return {'status': 'success'}