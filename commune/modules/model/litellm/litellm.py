import commune as c
import litellm
from typing import *
import os




class LiteLLM(c.Module):
    description = 'A lightweight LLM module'

    def __init__(self, api_key:str = 'OPENAI_API_KEY'):

        self.set_api(api_key)

    def set_api(self, api_key:str):
        self.api_key = os.getenv(api_key, api_key)

    def resolve_api_key(self, api_key:str):
        if api_key == None:
            api_key = self.api_key
        litellm.api_key = api_key
        return api_key


    def call(self, text: str = 'WHADUP ejdjjsjd' , 
            messages: List[Dict[str, str]]=None,
             model='gpt-3.5-turbo', api_key = None, **kwargs):
            
        api_key = self.resolve_api_key(api_key)

        if messages == None:
            messages = []
        messages.append({'role': 'user', 'content': text}) 
        response = litellm.completion(model=model, messages=messages, **kwargs)

        return response.choices[0].message['content']
        # cohere call

    generate = call

    @classmethod
    def install(cls):
        c.pip_install('litellm')