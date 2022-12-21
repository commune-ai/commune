from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio 
from copy import deepcopy
from typing import Union, Optional
from concurrent import futures
import os, sys
sys.path = list(set(sys.path + [os.getenv('PWD')])) 
asyncio.set_event_loop(asyncio.new_event_loop())
import torch
from tuwang.server import ServerModule
import streamlit as st
import bittensor
import tuwang
from munch import Munch

class HuggingfaceTransformer:
    def __init__(self,

                model: Union[str, 'Model']="EleutherAI/gpt-j-6B", 
                tokenizer:Union[str, 'tokenizer'] = None,
                config: Optional['tuwang.Config'] = None,
                device:str = 'cuda:1'
                ):

        
        # set model
        self.set_model(model=model)
        self.model.to(device)

        # set tokenizer to model name (HF only) if tokenizer == None
        tokenizer = tokenizer if tokenizer == None else model
        self.set_tokenizer(tokenizer=tokenizer)


    
    def forward(self, input_str:str=None,  input_ids: torch.Tensor = None, attention_mask: torch.Tensor= None, output_hidden_states:bool=True,**kwargs):

        input =  {}

        if isinstance(input_str, str):
            input = self.tokenizer( input_str, add_special_tokens=False, return_tensors="pt", padding=True).to(self.model.device)
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
                assert isinstance(input[k], torch.Tensor), input[k]
                input[k] = input[k].to(self.model.device)   
                 
        model_output = self.model(**input)
        output_dict = Munch(logits=model_output.logits)
        
        # if output_hidden_states:
        #     output_dict['hidden_states'] = model_output.hidden_states

        return output_dict

    __call__ = forward


    def set_model(self, model:str):
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        else:
            raise NotImplemented
        self.model = model
        return self.model


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tokenizer == None:
            tokenizer = bittensor.tokenizer()

        self.tokenizer = tokenizer
        
        
        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        return self.tokenizer

    @classmethod
    def test(cls):
        self = cls()
        input_text='bro whadup'
        input = self.tokenizer(input_text)
        output = self.forward(**input)
        st.write(output)





if __name__ == "__main__":
    HuggingfaceTransformer.test()


