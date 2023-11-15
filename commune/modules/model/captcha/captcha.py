import commune as c
import asyncio

from capmonstercloudclient import CapMonsterClient, ClientOptions
from capmonstercloudclient.requests import RecaptchaV2ProxylessRequest, RecaptchaV2Request, FuncaptchaProxylessRequest, FuncaptchaRequest, GeetestProxylessRequest, GeetestRequest, ImageToTextRequest, HcaptchaProxylessRequest, HcaptchaRequest, RecaptchaV2EnterpriseRequest, RecaptchaV2EnterpriseProxylessRequest, RecaptchaV3ProxylessRequest, TurnstileProxylessRequest, TurnstileRequest, HcaptchaComplexImageTaskRequest, RecaptchaComplexImageTaskRequest, FunCaptchaComplexImageTaskRequest

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

    def nocaptcha_proxyless(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha
             website_key: str,  # Recaptcha website key.
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = RecaptchaV2ProxylessRequest(
            websiteUrl=website_url, # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=high",
            websiteKey=website_key, # "6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd"
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def nocaptcha(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha
             website_key: str,  # Recaptcha website key.
             proxyType: str,    # Type of the proxy (http, https, socks3, socks5),
             proxyAddress: str, # Proxy IP address IPv4/IPv6. (not allowed to use hostnames, transparent proxies, local networks)
             proxyPort: str,    # Proxy port
             proxyLogin: str = "",  # Login for proxy which requires authorizaiton (basic)
             proxyPassword: str = "",   # Proxy password
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = RecaptchaV2Request(
            websiteUrl=website_url, # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=high",
            websiteKey=website_key, # "6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd",
            proxyType=proxyType,    # "http"
            proxyAddress=proxyAddress,  # "8.8.8.8"
            proxyPort=proxyPort,    # 8080
            proxyLogin=proxyLogin,
            proxyPassword=proxyPassword
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def hcaptcha_proxyless(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha
             website_key: str,  # Recaptcha website key.
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = HcaptchaProxylessRequest(
            websiteUrl=website_url, # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=high",
            websiteKey=website_key, # "6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd"
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def hcaptcha(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha
             website_key: str,  # Recaptcha website key.
             proxyType: str,    # Type of the proxy (http, https, socks3, socks5),
             proxyAddress: str, # Proxy IP address IPv4/IPv6. (not allowed to use hostnames, transparent proxies, local networks)
             proxyPort: str,    # Proxy port
             proxyLogin: str = "",  # Login for proxy which requires authorizaiton (basic)
             proxyPassword: str = "",   # Proxy password
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = HcaptchaRequest(
            websiteUrl=website_url, # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=high",
            websiteKey=website_key, # "6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd"
            proxyType=proxyType,    # "http"
            proxyAddress=proxyAddress,  # "8.8.8.8"
            proxyPort=proxyPort,    # 8080
            proxyLogin=proxyLogin,
            proxyPassword=proxyPassword
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def funcaptcha_proxyless(self,
             website_url: str,  # Address of a webpage with FunCaptcha
             website_public_key: str,   # FunCaptcha website key
             api_key: str = None,    # API key from https://api.capmonster.cloud/
             funcaptchaApiJSSubdomain: str = None,  # A special subdomain of funcaptcha.com, from which the JS captcha widget should be loaded
             data: str = None,   # Additional parameter that may be required by Funcaptcha implementation. 
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = FuncaptchaProxylessRequest(
            websiteUrl=website_url, # "https://funcaptcha.com/fc/api/nojs/?pkey=69A21A01-CC7B-B9C6-0F9A-E7FA06677FFC",
            websitePublicKey=website_public_key, # "69A21A01-CC7B-B9C6-0F9A-E7FA06677FFC"
            data=data,
            funcaptchaApiJSSubdomain=funcaptchaApiJSSubdomain
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def funcaptcha(self,
             website_url: str,  # Address of a webpage with FunCaptcha
             website_public_key: str,   # FunCaptcha website key
             proxyType: str,    # Type of the proxy (http, https, socks3, socks5),
             proxyAddress: str, # Proxy IP address IPv4/IPv6. (not allowed to use hostnames, transparent proxies, local networks)
             proxyPort: str,    # Proxy port
             proxyLogin: str = "",  # Login for proxy which requires authorizaiton (basic)
             proxyPassword: str = "",   # Proxy password
             api_key: str = None,    # API key from https://api.capmonster.cloud/
             funcaptchaApiJSSubdomain: str = None,  # A special subdomain of funcaptcha.com, from which the JS captcha widget should be loaded
             data: str = None,   # Additional parameter that may be required by Funcaptcha implementation. 
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = FuncaptchaRequest(
            websiteUrl=website_url, # "https://funcaptcha.com/fc/api/nojs/?pkey=69A21A01-CC7B-B9C6-0F9A-E7FA06677FFC",
            websitePublicKey=website_public_key, # "69A21A01-CC7B-B9C6-0F9A-E7FA06677FFC"
            proxyType=proxyType,    # "http"
            proxyAddress=proxyAddress,  # "8.8.8.8"
            proxyPort=proxyPort,    # 8080
            proxyLogin=proxyLogin,
            proxyPassword=proxyPassword,
            data=data,
            funcaptchaApiJSSubdomain=funcaptchaApiJSSubdomain
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def geetest_proxyless(self,
             website_url: str,  # Address of a webpage with FunCaptcha
             gt: str,   # The GeeTest identifier key for the domain. Static value, rarely updated.
             challenge: str,   # A dynamic key.
             api_key: str = None,    # API key from https://api.capmonster.cloud/
             geetestApiServerSubdomain: str = None,  # Optional parameter. May be required for some sites.
             geetestGetLib: str = None,   # Optional parameter. May be required for some sites. Send JSON as a string.
             version: int = 3,  # Version number (default is 3). Possible values: 3, 4.

    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = GeetestProxylessRequest(
            websiteUrl=website_url, # "https://example.com/geetest.php",
            gt=gt,  # "81dc9bdb52d04dc20036dbd8313ed055"
            challenge=challenge,    # "d93591bdf7860e1e4ee2fca799911215"
            geetestApiServerSubdomain=geetestApiServerSubdomain,
            geetestGetLib=geetestGetLib,
            version=version
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def geetest(self,
             website_url: str,  # Address of a webpage with FunCaptcha
             gt: str,   # The GeeTest identifier key for the domain. Static value, rarely updated.
             challenge: str,   # A dynamic key.
             proxyType: str,    # Type of the proxy (http, https, socks3, socks5),
             proxyAddress: str, # Proxy IP address IPv4/IPv6. (not allowed to use hostnames, transparent proxies, local networks)
             proxyPort: str,    # Proxy port
             proxyLogin: str = "",  # Login for proxy which requires authorizaiton (basic)
             proxyPassword: str = "",   # Proxy password
             api_key: str = None,    # API key from https://api.capmonster.cloud/
             geetestApiServerSubdomain: str = None,  # Optional parameter. May be required for some sites.
             geetestGetLib: str = None,   # Optional parameter. May be required for some sites. Send JSON as a string.
             version: int = 3,  # Version number (default is 3). Possible values: 3, 4.

    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = GeetestRequest(
            websiteUrl=website_url, # "https://example.com/geetest.php",
            gt=gt,  # "81dc9bdb52d04dc20036dbd8313ed055"
            challenge=challenge,    # "d93591bdf7860e1e4ee2fca799911215"
            geetestApiServerSubdomain=geetestApiServerSubdomain,
            geetestGetLib=geetestGetLib,
            version=version,
            proxyType=proxyType,    # "http"
            proxyAddress=proxyAddress,  # "8.8.8.8"
            proxyPort=proxyPort,    # 8080
            proxyLogin=proxyLogin,
            proxyPassword=proxyPassword
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def recaptcha2_enterprise_proxyless(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha Enterprise
             website_key: str,  # Recaptcha website key.
             enterprisePayload: str = None, # Some implementations of the reCAPTCHA Enterprise widget may contain additional parameters that are passed to the “grecaptcha.enterprise.render” method along with the sitekey.
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = RecaptchaV2EnterpriseProxylessRequest(
            websiteUrl=website_url, # "https://mydomain.com/page-with-recaptcha-enterprise",
            websiteKey=website_key, # "6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd",
            enterprisePayload=enterprisePayload
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def recaptcha2_enterprise(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha Enterprise
             website_key: str,  # Recaptcha website key.
             proxyType: str,    # Type of the proxy (http, https, socks3, socks5),
             proxyAddress: str, # Proxy IP address IPv4/IPv6. (not allowed to use hostnames, transparent proxies, local networks)
             proxyPort: str,    # Proxy port
             proxyLogin: str = "",  # Login for proxy which requires authorizaiton (basic)
             proxyPassword: str = "",   # Proxy password
             enterprisePayload: str = None, # Some implementations of the reCAPTCHA Enterprise widget may contain additional parameters that are passed to the “grecaptcha.enterprise.render” method along with the sitekey.
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = RecaptchaV2EnterpriseRequest(
            websiteUrl=website_url, # "https://mydomain.com/page-with-recaptcha-enterprise",
            websiteKey=website_key, # "6Lcg7CMUAAAAANphynKgn9YAgA4tQ2KI_iqRyTwd",
            enterprisePayload=enterprisePayload,
            proxyType=proxyType,    # "http"
            proxyAddress=proxyAddress,  # "8.8.8.8"
            proxyPort=proxyPort,    # 8080
            proxyLogin=proxyLogin,
            proxyPassword=proxyPassword
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def recaptcha3_proxyless(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha
             website_key: str,  # Recaptcha website key.
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = RecaptchaV3ProxylessRequest(
            websiteUrl=website_url, # "https://lessons.zennolab.com/captchas/recaptcha/v3.php?level=beta"
            websiteKey=website_key, # "6Le0xVgUAAAAAIt20XEB4rVhYOODgTl00d8juDob"
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def turnstile_proxyless(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha Enterprise
             website_key: str,  # Turnstile key
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = TurnstileProxylessRequest(
            websiteUrl=website_url, # "http://tsmanaged.zlsupport.com"
            websiteKey=website_key, # "0x4AAAAAAABUYP0XeMJF0xoy"
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def turnstile(self,
             website_url: str,  # Address of a webpage with Google ReCaptcha Enterprise
             website_key: str,  # Turnstile key
             proxyType: str,    # Type of the proxy (http, https, socks3, socks5),
             proxyAddress: str, # Proxy IP address IPv4/IPv6. (not allowed to use hostnames, transparent proxies, local networks)
             proxyPort: str,    # Proxy port
             proxyLogin: str = "",  # Login for proxy which requires authorizaiton (basic)
             proxyPassword: str = "",   # Proxy password
             api_key:str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = TurnstileRequest(
            websiteUrl=website_url, # "http://tsmanaged.zlsupport.com"
            websiteKey=website_key, # "0x4AAAAAAABUYP0XeMJF0xoy"
            proxyType=proxyType,    # "http"
            proxyAddress=proxyAddress,  # "8.8.8.8"
            proxyPort=proxyPort,    # 8080
            proxyLogin=proxyLogin,
            proxyPassword=proxyPassword
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def image_to_text(self,
             image_bytes: str,  # File body encoded in base64.
             api_key: str = None,    # API key from https://api.capmonster.cloud/
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = ImageToTextRequest(
            image_bytes=image_bytes
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def compleximage_hcaptcha(self,
             imageUrls: [],   # List with image URLs. Max 18 elements per request. (if imagesBase64 is not filled)
             imagesBase64: [],   # List with images in base64 format. Max 18 elements per request. (if imageUrls is not filled)
             task: str,   # Task text (in English) (e.g. "Please click each image containing a mountain")
             api_key: str = None,    # API key from https://api.capmonster.cloud/
             website_url: str = None,  # URL of the page where the captcha is solved
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = HcaptchaComplexImageTaskRequest(
            imageUrls=imageUrls,   # [ 'https://i.postimg.cc/kg71cbRt/image-1.jpg', 'https://i.postimg.cc/6381Zx2j/image.jpg' ]
            metadata={
                "Task": task,    # "Please click each image containing a mountain"
            },
            imagesUrls=imageUrls,
            imagesBase64=imagesBase64,
            websiteUrl=website_url,    # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=middle"
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def compleximage_recaptcha(self,
             imageUrls: [],   # List with image URLs. Max 18 elements per request. (if imagesBase64 is not filled)
             imagesBase64: [],   # List with images in base64 format. Max 18 elements per request. (if imageUrls is not filled)
             task: str,   # Task text (in English) (e.g. "Please click each image containing a mountain")
             grid: str,   #  Image grid size
             taskDefinition: str,   # Technical value that defines the task type (if `task` is not filled)
             api_key: str = None,    # API key from https://api.capmonster.cloud/
             website_url: str = None,  # URL of the page where the captcha is solved
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = RecaptchaComplexImageTaskRequest(
            imageUrls=imageUrls,   # [ 'https://i.postimg.cc/kg71cbRt/image-1.jpg', 'https://i.postimg.cc/6381Zx2j/image.jpg' ]
            metadata={
                "Task": task,   # "Click on traffic lights"
                "Grid": grid,   # "3x3"
                "TaskDefinition": taskDefinition    # "/m/015qff"
            },
            imagesUrls=imageUrls,
            imagesBase64=imagesBase64,
            websiteUrl=website_url,    # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=middle"
        )

        responses = asyncio.run(_solve_captcha())
        return responses
    
    def compleximage_funcaptcha(self,
             imageUrls: [],   # List with image URLs. Max 18 elements per request. (if imagesBase64 is not filled)
             imagesBase64: [],   # List with images in base64 format. Max 18 elements per request. (if imageUrls is not filled)
             task: str,   # Task text (in English) (e.g. "Please click each image containing a mountain")
             api_key: str = None,    # API key from https://api.capmonster.cloud/
             website_url: str = None,  # URL of the page where the captcha is solved.
    ) -> str:
        api_key = api_key if api_key != None else self.api_key

        client_options = ClientOptions(api_key=api_key)
        cap_monster_client = CapMonsterClient(options=client_options)

        async def _solve_captcha():
            return await cap_monster_client.solve_captcha(request)
        
        request = FunCaptchaComplexImageTaskRequest(
            imageUrls=imageUrls,   # [ "https://i.postimg.cc/s2ZDrHXy/fc1.jpg" ]
            metadata={
                "Task": task,    # "Pick the image that is the correct way up"
            },
            imagesUrls=imageUrls,
            imagesBase64=imagesBase64,
            websiteUrl=website_url,    # "https://lessons.zennolab.com/captchas/recaptcha/v2_simple.php?level=middle"
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
    