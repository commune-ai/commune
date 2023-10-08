import requests
import time
from typing import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import commune as c



urls = [
    'http://50.173.30.254:21027/generate', #3090s models8x1 good
    'http://50.173.30.254:21088/generate',
    'http://50.173.30.254:21086/generate',
    'http://50.173.30.254:21051/generate',
    'http://50.173.30.254:21078/generate',
    'http://50.173.30.254:21090/generate',
    'http://50.173.30.254:21050/generate',
    'http://50.173.30.254:21075/generate',
]

class RemoteModel(c.Module):
    def __init__( self, 
                 urls: List[str] = urls ,
                  num_workers : int= 100):
        
        config = self.set_config(kwargs=locals())
        self.urls = urls
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    @staticmethod
    def remote_call(url, data:dict, timeout:int=10):
        response = requests.post(url, json=data, timeout=timeout)
        response_data = response.json()
        return response_data


    def generate(self, 
                prompt:str = 'Sup fam',
                 n:int=6, 
                 num_tokens:int=160, 
                 top_p = 0.92, 
                 temperature=0.8, 
                 top_k  = 1000,
                 timeout = 60, 
                 num_endpoints = 1):
        
        data = {
            'prompt': prompt,
            'n': n,
            'num_tokens': num_tokens,
            'top_p': top_p,
            'temperature': temperature,
            'top_k': top_k
        }

            # Use ThreadPoolExecutor to send the requests concurrently
        # Submit tasks for execution
        urls = c.shuffle(self.urls)[:num_endpoints]
        futures = [self.executor.submit(self.remote_call, kwargs={'data': data, 'timeout': timeout, 'url': url}) for url in urls]
        results = c.wait(futures, timeout=timeout)
        return results


