from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio 
from copy import deepcopy
from typing import Union, Optional
from concurrent import futures
import os, sys
import socket
sys.path = list(set(sys.path + [os.getenv('PWD')])) 
asyncio.set_event_loop(asyncio.new_event_loop())
import torch
from commune.server import ServerModule
from commune.model.remote.utils import encode_topk, decode_topk

import streamlit as st
import bittensor
import commune

class RemoteModelServer(torch.nn.Module):
    model_options = ["EleutherAI/gpt-j-6B", "EleutherAI/gpt-neo-2.7B"]
    def __init__(self,
                model: Union[str, 'Model'] = 'EleutherAI/gpt-neo-125M',
                tokenizer:Union[str, 'tokenizer'] = None,
                ip: str = '0.0.0.0',
                port: int = 50056, 
                config: Optional['commune.Config'] = None,
                serve:bool = None,
                device:str = 'cuda:3',
                eval_mode: bool= True,
                refresh:bool = False
                ):

        torch.nn.Module.__init__(self)
        
        # set server
        self.set_server(ip=ip, port=port, refresh=refresh)
        
        # set model
        self.set_model(model=model, device=device, eval_mode=eval_mode)
        
        # set tokenizer to model name (HF only) if tokenizer == None
        self.set_tokenizer(tokenizer=tokenizer if tokenizer else model)


        if serve:
            self.serve()

    
    def set_server(self, ip:str, port:int, refresh:bool):
        # Setup server
        self.server = commune.server.ServerModule(module = self ,ip=ip, port=port, refresh=refresh)
        self.ip = self.server.ip
        self.port = self.server.port
        return self.server

    def forward(self, *args,no_grad=True, **kwargs):
        if no_grad:
            with torch.no_grad():
                return self._forward(*args,**kwargs)
        else:
            return self._forward(*args,**kwargs)
    def _forward(self, input:str = None, input_ids: torch.Tensor = None, attention_mask: torch.Tensor= None, output_hidden_states:bool=True, topk:int=None, **kwargs):

        if type(input) in [list, str]:
            if isinstance(input, str):
                input = [input]
            input = self.tokenizer( input, add_special_tokens=False, return_tensors="pt", padding=True).to(self.model.device)
        else:
            input = dict(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        output_hidden_states= output_hidden_states
                        )
            # ensure the input_ids and attention mask is a tensor
            for k in ['input_ids', 'attention_mask']:
                v = input[k]
                if isinstance(v,  list):
                    input[k] = torch.tensor(v)
                input[k] = input[k].to(self.model.device)   
                 
        model_output = self.model(**input)
        if topk:
            topk_tensor = encode_topk(model_output.logits, topk=topk)
            output_dict = dict(topk=topk_tensor)
        else:
            output_dict = dict(logits=model_output.logits)
        
        return output_dict


    def __call__(self, data:dict, metadata:dict={}):
        if 'fn' in data and 'kwargs' in data:
            data =getattr(self, data['fn'])(**data['kwargs'])
        else:
            data = self.forward(**data)

        return {'data': data, 'metadata': metadata}


    def serve(self):
        self.server.start()
        
    def set_model(self, model:str, device:str, eval_mode:bool):
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        else:
            raise NotImplemented
        self.model = model.to(device)
        if eval_mode:
            self.model = self.model.eval()

        return self.model


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer == None:
            tokenizer = bittensor.tokenizer()

        self.tokenizer = tokenizer
        
        
        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        return self.tokenizer


    @classmethod
    def streamlit_client_server(cls):
        
        model_server = cls(serve=True)
        st.write(f'## {cls.__name__}')
        st.write('LOCATION: ', __file__)
        
        model_client = commune.model.remote.RemoteModelClient(ip=model_server.ip, port=model_server.port)


        with st.expander('STEP 1: GET RAW TEXT', True):
            input_text = st.text_area('Input Text', 'Whadup fam')
        
        input = dict(model_server.tokenizer([input_text]))
        output = model_client(**input)
        with st.expander('STEP 2: TOKENIZE RAW TEXT INTO TENSOR', True):
            st.write('INPUT TOKENS',input)
            st.write('SHAPE ',torch.tensor(input['input_ids']).shape)

        with st.expander('OUTPUT', True):  
            st.write(output)
            st.write('SHAPE ',output['logits'].shape)

    @classmethod
    def streamlit_server(cls):
        
        model_server = cls(serve=True, refresh=True)
        st.write(f'## {cls.__name__}')
        st.write('LOCATION: ', __file__)
        st.write(model_server.__dict__)
        st.write(model_server.model.device)

        data = {}
        data['fn'] = 'forward'
        data['kwargs'] = dict(input='hey', topk=100)
        st.write(model_server(data=data))


if __name__ == "__main__":
    # RemoteModelServer.streamlit_server()
    commune.ray_init()
    st.write(commune.list_modules())
    # from commune.utils import *


