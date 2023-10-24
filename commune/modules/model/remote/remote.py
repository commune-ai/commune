import requests
import time
from typing import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import commune as c

class RemoteModel(c.Module):

    urls = [
        'http://0.0.0.0:21027/generate', #3090s models8x1 good
        'http://0.0.0.0:21088/generate',
        'http://0.0.0.0:21086/generate',
        'http://0.0.0.0:21051/generate',
        'http://0.0.0.0:21078/generate',
        'http://0.0.0.0:21090/generate',
        'http://0.0.0.0:21050/generate',
        'http://0.0.0.0:21075/generate',
    ]
    def __init__( self, urls: List[str] = urls , num_workers : int= 100, access={'base_rate': 10}, **kwargs):
        self.urls = urls 
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.set_access_module(**access)

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
                 timeout = 3, 
                 num_endpoints = 1, 
                 return_text:bool = True):
        
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
        futures = [self.executor.submit(self.remote_call, data= data, timeout= timeout,url=url) for url in urls]
        results = c.wait(futures, timeout=timeout)
        # filter results that are not None
        if return_text:
            results = [result for result in results if 'response' in result]
            if len(results) == 0:
                raise Exception(f"Not enough endpoints to service request, {len(results)} < {num_endpoints}")
            return results[0]['response'][0]
        else:
            return results


