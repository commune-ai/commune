from typing import Generator
import requests
import json
import openai
import commune as c

class OpenRouter(c.Module):
    system_prompt = r""""""

    def __init__(
        self,
        model: str = 'anthropic/claude-3-haiku',
        search = None,
        api_key = None,
        base_url: str | None = 'https://openrouter.ai/api/v1',
        timeout: float | None = None,
        max_retries: int = 10,
    ):
        """
        Initialize the OpenAI with the specified model, API key, timeout, and max retries.

        Args:
            model (OPENAI_MODES): The OpenAI model to use.
            api_key (API_KEY): The API key for authentication.
            base_url (str | None, optional): can be used for openrouter api calls
            timeout (float | None, optional): The timeout value for the client. Defaults to None.
            max_retries (int | None, optional): The maximum number of retries for the client. Defaults to None.
        """
        super().__init__()

        if api_key == None:
            api_key = self.get_api_key()

        self.authenticate(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.model = model

    @c.endpoint(cost=1)
    def generate(
        self,
        message: str,
        *extra_text , 
        history = None,
        system_prompt: str =  None,
        stream: bool = False,
        model:str = None,
        max_tokens: int = 100000,
        text = None,
        temperature: float = 1.0,
    ) -> str | Generator[str, None, None]:
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
        if len(extra_text) > 0:
            message = message + ' '.join(extra_text)
        history = history or []
        system_prompt = system_prompt or self.system_prompt
        message = message + system_prompt
        model = model or self.model
        stream = bool(stream)
        messages = history.copy()
        # append new message to model
        messages.append({"role": "user", "content": message})

        result = self.client.chat.completions.create(
                                                    model=model,
                                                    messages=messages,
                                                    stream=stream, 
                                                    max_tokens = max_tokens,
                                                    temperature= temperature, 
                                                    )
        if stream:
            return self.stream_generator(result)
        else:
            return result.choices[0].message.content
        
    forward = generate

    def stream_generator(self, result):
        for token in result:
            yield token.choices[0].delta.content

    def authenticate(
        self,
        api_key: str,
        base_url: None = None,
        timeout: float | None = None,
        max_retries: int = 5,
    ) -> 'OpenAI':
        """
        Authenticate the client with the provided API key, timeout, and max retries.

        Args:
            api_key (str): The API key for authentication.
            timeout (float | None, optional): The timeout value for the client. Defaults to None.
            max_retries (int, optional): The maximum number of retries for the client. Defaults to 0.

        """
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        return {"status": "success", "base_url": base_url}
    
    def get_model_info(self, model):
        url = f'https://openrouter.ai/api/v1/models/{model}'
        response = requests.get(url)
        return json.loads(response.text)['data']
    
    def model2info(self, search: str = None, path='models', max_age=0, update=False):
        models = self.get(path, default={}, max_age=max_age, update=update)
        if len(models) == 0:
            print('Updating models...')
            url = 'https://openrouter.ai/api/v1/models'
            response = requests.get(url)
            models = json.loads(response.text)['data']
            self.put(path, models)
    
        models = self.filter_models(models, search=search)
        models = {m['id']:m for m in models}
        return models
    
    def models(self, search: str = None, path='models', max_age=0, update=False):
        return list(self.model2info(search=search, path=path, max_age=max_age, update=update).keys())

    
    def model_infos(self, search: str = None, path='models', max_age=0, update=False):
        return list(self.model2info(search=search, path=path, max_age=max_age, update=update).values())
    
    def get_model_info(self, model):
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
    