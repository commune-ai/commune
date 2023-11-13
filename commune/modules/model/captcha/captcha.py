import commune as c
import asyncio

from capmonstercloudclient import CapMonsterClient, ClientOptions
from capmonstercloudclient.requests import RecaptchaV2ProxylessRequest, FuncaptchaProxylessRequest

class Captcha(c.Module):
    def __init__(self, api_key:str = None, host='https://api.capmonster.cloud/', cache_key:bool = True):
        self.set_config(kwargs=locals())
        self.set_api_key(api_key=self.config.api_key, cache=self.config.cache_key)

    def set_api_key(self, api_key:str, cache:bool = True):
        if api_key == None:
            api_key = self.get_api_key()

        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

    def recaptcha2(self,
             website_url: str,
             website_key: str,
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(recaptcha2request)
        
        recaptcha2request = RecaptchaV2ProxylessRequest(
            websiteUrl=website_url, # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=high",
            websiteKey=website_key, # "6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd"
        )

        responses = asyncio.run(_solve_captcha())
        return responses

    
    ## API MANAGEMENT ##

    @classmethod
    def add_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}
    
    @classmethod
    def rm_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   

        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def get_api_key(cls):
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    @classmethod
    def api_keys(cls):
        return cls.get('api_keys', [])
    