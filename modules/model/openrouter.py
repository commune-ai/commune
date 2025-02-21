from typing import Generator
import requests
import json
import openai
import commune as c

class OpenRouter:

    def __init__(
        self,
        api_key = None,
        base_url: str = 'https://openrouter.ai/api/v1',
        timeout: float = None,
        max_retries: int = 10,
        **kwargs
    ):
        """
        Initialize the OpenAI with the specified model, API key, timeout, and max retries.

        Args:
            model (OPENAI_MODES): The OpenAI model to use.
            api_key (API_KEY): The API key for authentication.
            base_url (str, optional): can be used for openrouter api calls
            timeout (float, optional): The timeout value for the client. Defaults to None.
            max_retries (int, optional): The maximum number of retries for the client. Defaults to None.
        """

        self.prompt = None

        self.authenticate(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(
        self,
        message: str,
        *extra_text , 
        history = None,
        prompt: str =  None,
        system_prompt: str = None,
        stream: bool = False,
        model:str = 'anthropic/claude-3.5-sonnet',
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
        prompt = prompt or system_prompt
        if len(extra_text) > 0:
            message = message + ' '.join(extra_text)
        history = history or []
        prompt = prompt or self.prompt
        message = message + prompt if prompt else message
        print('model', model)
        model = self.resolve_model(model)
        model_info = self.get_model_info(model)
        num_tokens = len(message)
        print(f'Sending {num_tokens} tokens -> {model}')
        max_tokens = min(max_tokens, model_info['context_length'] - num_tokens)
        messages = history.copy()
        messages.append({"role": "user", "content": message})
        result = self.client.chat.completions.create(model=model, messages=messages, stream= bool(stream), max_tokens = max_tokens, temperature= temperature  )
    

        if stream:
            def stream_generator( result):
                for token in result:
                    yield token.choices[0].delta.content
            return stream_generator(result)
        else:
            return result.choices[0].message.content
        
    forward = generate


    def resolve_model(self, model=None):
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
    
    def get_key(self):
        return c.module('apikey')(module=self).get_key()

    def authenticate(
        self,
        api_key: str = None,
        base_url: None = None,
        timeout: float = None,
        max_retries: int = 5,
    ) -> 'OpenAI':
        """
        Authenticate the client with the provided API key, timeout, and max retries.

        Args:
            api_key (str): The API key for authentication.
            timeout (float, optional): The timeout value for the client. Defaults to None.
            max_retries (int, optional): The maximum number of retries for the client. Defaults to 0.

        """
        if api_key == None:
            api_key = self.get_key()
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        return {"status": "success", "base_url": base_url}
    

    @staticmethod
    def resolve_path(path):
        return c.storage_path + '/openrouter/' + path

    def model2info(self, search: str = None, path='models', max_age=100, update=False):
        path = self.resolve_path(path)
        models = c.get(path, default={}, max_age=max_age, update=update)
        if len(models) == 0:
            print('Updating models...')
            url = 'https://openrouter.ai/api/v1/models'
            response = requests.get(url)
            models = json.loads(response.text)['data']
            c.put(path, models)
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
    
    def pricing(self, search: str = None , **kwargs):
        pricing =  [{'name': k , **v['pricing']} for k,v in self.model2info(search=search, **kwargs).items()]
        return c.df(pricing).sort_values('completion', ascending=False)
    

    def test(self):
        response  =  self.forward('Hello, how are you?', stream=False)
        print(response)
        assert isinstance(response, str)
        print('Test passed')
        stream_response = self.forward('Hello, how are you?', stream=True)
        print(next(stream_response))
        return {'status': 'success'}