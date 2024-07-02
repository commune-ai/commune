import commune as c
import requests

class Web(c.Module):
    @classmethod
    def request(self, url:str, method:str='GET', **kwargs):
        response =  requests.request(method, url, **kwargs)
        if response.status_code == 200:
            return response.text
        else:
            return {'status_code': response.status_code, 'text': response.text}
    @classmethod
    def rget(cls, url:str, **kwargs):
        return cls.request(url, 'GET', **kwargs)
    @classmethod
    def rpost(self, url:str, **kwargs):
        return cls.request(url, 'POST', **kwargs)
    
