# make sure you're logged in with `huggingface-cli login`


import os
import sys
from copy import deepcopy
import streamlit as st
import asyncio
import transformers
sys.path = list(set(sys.path + [os.getenv('PWD')])) 
asyncio.set_event_loop(asyncio.new_event_loop())
from commune.utils import dict_put, get_object, dict_has


from commune import Module
import torch
import os
import io
import glob
import numpy as np
import uuid
import pandas as pd
from PIL import Image
import torch
import ray


class TransformerModel(Module):

    def __init__(self, config=None, model:str=None, tokenizer=None,  **kwargs):
        Module.__init__(self, config=config, **kwargs)

        self.load_model(model)
        self.load_tokenizer(tokenizer)

    @property
    def hf_token(self):
        return self.config['hf_token']
    @property
    def model_path(self):
        return self.config['model_path']

    def load_tokenizer(self, tokenizer: Union[str, 'tokenizer']=None):
        tokenizer = tokenizer if tokenizer else self.model_path
        if isinstance(tokenizer, str)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        else:
            raise NotImplemented(type(tokenizer))
        # fixes the tokenizer
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 
    @property
    def model_loader(self):
        return self.config['loader']

    def load_model(self, model=None, loader=None, *args, **kwargs ):
        loader = loader if loader else self.model_loader
        model = model if model else self.model_path
        self.model =  getattr(transformers, loader).from_pretrained(model, *args, **kwargs )
        self.model.to(self.device)

    default_device = 'cuda:0'
    @property
    def device(self):
        st.write(self.config.get('device', default_device))
        return self.config.get('device', default_device)

    def forward(self, input:str="This is the first sentence. This is the second sentence.", tokenize=False):
    
        input = self.tokenizer(
                input, add_special_tokens=False, return_tensors="pt", padding=True,
            ).to(self.model.device)
        return self.model(**input)

    @staticmethod
    def streamlit_pipeline():
        dataset = Module.launch('commune.dataset.text.huggingface', actor=False )
        model = Module.launch('commune.model.transformer', actor=False )
        st.write(model.get_default_actor_name())
        st.write(model.actor_name)
        x = dataset.sample(tokenize=False)['text']
        st.write(x)
        st.write(model.forward(x))

    @classmethod
    def streamlit(cls):
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        Module.init_ray()
        

if __name__ == '__main__':
    TransformerModel.streamlit_pipeline()

