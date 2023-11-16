import http.client
import json
import commune as c
from openai import OpenAI
# import base64

# from PIL import Image
# from io import BytesIO



class DallE(c.Module):
    
    whitelist = []

    def __init__(self, api_key:str = None, host='api.openai.com/v1/images', cache_key:bool = True):
        config = self.set_config(kwargs=locals())
        self.conn = http.client.HTTPSConnection(self.config.host)
        self.set_api_key(api_key=config.api_key, cache=config.cache_key)
        
    def set_api_key(self, api_key:str, cache:bool = True):
        if api_key == None:
            api_key = self.get_api_key()

        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

    def generate( self, 
                prompt: str, # required; max len is 1000 for dall-e-2 and 4000 for dall-e-3
                model: str = "dall-e-2", # "dall-e-2" | "dall-e-3"
                n: int = 1, # number of images Must be between 1 and 10. For dall-e-3, only n=1
                quality: str = "standard",  # "standard" | "hd" only supported for dall-e-3
                response_format: str = "url", # "url" or "b64_json"
                size: str = "1024x1024", #  256x256, 512x512, 1024x1024 for dall-e-2.  1024x1024, 1792x1024, 1024x1792 for dall-e-3
                style: str = "vivid", # "vivid" | "natural", only supported for dall-e-3
                api_key: str = None, # api_key "sk-..."
                ) -> str: 
        api_key = api_key if api_key != None else self.api_key

        client = OpenAI(api_key = api_key)

        response = client.images.generate(
            prompt=prompt,
            model=model,
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
        )

        return response
    
    def edit( self, 
                prompt: str, # required; max len is 1000 for dall-e-2 and 4000 for dall-e-3
                image: str, # required; PNG file, less than 4MB,
                mask: str, # fully transparent areas; PNG file, less than 4MB,
                model: str = "dall-e-2", # "dall-e-2"
                n: int = 1, # number of images Must be between 1 and 10
                response_format: str = "url", # "url" or "b64_json"
                size: str = "1024x1024", #  256x256, 512x512, 1024x1024 
                api_key: str = None, # api_key "sk-..."
                ) -> str: 
        api_key = api_key if api_key != None else self.api_key

        client = OpenAI(api_key = api_key)

        response = client.images.edit(
            prompt=prompt,
            model=model,
            image=open(image, "rb"),
            mask=open(mask, "rb"),
            n=n,
            response_format=response_format,
            size=size,
        )

        return response
    
    def variation( self, 
                image: str, # required; PNG file, less than 4MB,
                model: str = "dall-e-2", # "dall-e-2"
                n: int = 1, # number of images Must be between 1 and 10
                response_format: str = "url", # "url" or "b64_json"
                size: str = "1024x1024", #  256x256, 512x512, 1024x1024 
                api_key: str = None, # api_key "sk-..."
                ) -> str: 
        api_key = api_key if api_key != None else self.api_key

        client = OpenAI(api_key = api_key)

        response = client.images.create_variation(
            model=model,
            image=open(image, "rb"),
            n=n,
            response_format=response_format,
            size=size,
        )

        return response