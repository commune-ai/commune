# make sure you're logged in with `huggingface-cli login`


import os
import sys
from copy import deepcopy
import streamlit as st
from commune.utils import dict_put, get_object, dict_has
from commune import Module
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler


class TransformerModel(Module):

    def __init__(self, config=None, model=None, tokenizer=None,  **kwargs):
        Module.__init__(self, config=config, **kwargs)

        self.load_model(model)
        self.load_tokenizer(tokenizer)

    @property
    def hf_token(self):
        return self.config['hf_token']

    def load_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else self.launch(**self.config['tokenizer'])

    def load_model(self, model=None):
        self.model =  model if model else self.launch(**self.config['model'])
        self.model.to(self.device)

    @property
    def device(self):
        st.write(self.config.get('device', 'cuda'))
        return self.config.get('device', 'cuda')

    def forward(self, input:str="This is the first sentence. This is the second sentence.", tokenize=False):
    
        input = self.tokenizer(
                input, add_special_tokens=False, return_tensors="pt", padding=True,
            ).to(self.model.device)
        return self.model(**input)

    def streamlit_pipeline(self):
        dataset = Module.launch('dataset.text', actor={'cpus':1})
        model = Module.launch('model.transformer', actor={'gpus': 0.1, 'cpus':1, 'wrap':True} )
        st.write(model.get_default_actor_name())
        st.write(model.actor_name)
        x = dataset.sample(tokenize=False)
        st.write(model.forward(x))

    @classmethod
    def streamlit(cls):
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        Module.init_ray()
        

if __name__ == '__main__':
    TransformerModel

