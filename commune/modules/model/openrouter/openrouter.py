import commune as c
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class OpenRouterModule(c.Module):

    def __init__(self,
                url:str = "https://openrouter.ai/api/v1/chat/completions",
                model: str = "mistralai/mistral-7b-instruct",
                role: str = "user",
                http_referer: str = "http://localhost:3000",
                api_key: str = None,
                x_title: str = "Communne"
                ):
        self.url = url
        self.api_key = os.getenv("OPEN_ROUTER_API_KEY", self.get_api_key())
        self.model = model
        self.role = role
        self.http_referer = http_referer
        self.x_title = x_title

    def prompt(self, content: str):
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

        return response.text

    @classmethod
    def set_api_key(cls, api_key:str):
        assert isinstance(api_key, str), "API key must be a string"
        cls.put('api_key', api_key)
        return {'msg': f"API Key has been set to {api_key}", 'success': True}

    @classmethod
    def get_api_key(cls):
        api_key = cls.get('api_key')
        return api_key