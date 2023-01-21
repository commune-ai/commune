
import os
import sys
import json
sys.path.append(os.getcwd())
from commune import Module
import ray
import requests

class RestModule(Module):
    default_config_path = f"client.rest.module"
    def __init__(
        self,
        config=None
    ):
        Module.__init__(self, config=config)

    @property
    def url(self):
        if not hasattr(self,'_url'):
            url = self.config.get('url')
            if not url.startswith('http://'):
                url = 'http://'+url
            if url == None:
                assert 'host' in self.config
                assert 'port' in self.config
                url = f"http://{self.config['host']}:{self.config['port']}"
            self._url = url
        
        return self._url

    @url.setter
    def url(self, value:str):
        self._url = value

    def resolve_url(self,url ):
        if isinstance(url,str):
            return url
        else:
            assert isinstance(self.url, str)
            return self.url


    def get(self, url:str=None,endpoint:str=None, params={},**kwargs):
        url = self.resolve_url(url=url)
        if isinstance(endpoint, str):
            url = os.path.join(url, endpoint)
        return requests.get(url=url, params=params, **kwargs).json()

    def post(self, url:str=None, endpoint:str=None, **kwargs):
        url = self.resolve_url(url=url)
        if isinstance(endpoint, str):
            url = os.path.join(url, endpoint)

        return requests.post(url=url, **kwargs).json()

if __name__ == '__main__':
    import streamlit as st
    rest = RestModule.deploy(actor=False)
    st.write(rest.url)
    # st.write(rest.get(endpoint='module/list'))
    # st.write(rest.get(endpoint='module/start', params={'module': 'gradio.example.module.ExampleModule'}))
    import socket, subprocess
    PORT = 7866
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', PORT))
    sock.listen(5)
    cli, addr = sock.accept()
    # print(api.get(endpoint='launcher/send', 
    #                  params=dict(module='process.bittensor.module.BitModule', fn='getattr', kwargs='{"key": "n"}')))

    


