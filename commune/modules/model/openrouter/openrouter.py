import commune as c
import requests
import json
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

class OpenRouterModule(c.Module):

    def __init__(self,
                url:str = "https://openrouter.ai/api/v1/chat/completions",
                model: str = "mistralai/mistral-7b-instruct",
                role: str = "user",
                http_referer: str = "http://localhost:3000",
                api_key: str = 'OPEN_ROUTER_API_KEY',
                x_title: str = "Communne"
                ):
        self.url = url
        self.set_api_key(api_key)
    

        self.model = model
        self.role = role
        self.http_referer = http_referer
        self.x_title = x_title

    def prompt(self, content: str, text_only:bool = True, model=None):
        model = model or self.model
        response = requests.post(
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.http_referer, 
                "X-Title": self.x_title, 
            },
            data=json.dumps({
                "model": self.model, 
                "messages": [
                {"role": self.role, "content": content}
                ]
            })
            )
        response = json.loads(response.text)
        
        if text_only:
            return response["choices"][0]["message"]["content"]

        return response
    
    generate = prompt

    def set_api_key(self, api_key:str):
        api_key = os.getenv(api_key, None)
        if api_key == None:
            api_keys = self.api_keys()
            assert len(api_keys) > 0, "No API keys found. Please set an API key with OpenRouterModule.set_api_key()"
            api_key = c.choice(api_keys)
        assert isinstance(api_key, str), "API key must be a string"
        self.api_key = api_key   
    

    
    def test(self):
        response = self.prompt("Hello")
        assert isinstance(response, str)
        return {"status": "success", "response": response}
    
    @classmethod
    def models(cls):
        url = 'https://openrouter.ai/api/v1/models'

        response = requests.get(url)
        response = json.loads(response.text)
        return response 