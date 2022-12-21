from transformers import AutoTokenizer
import asyncio 
import torch

from copy import deepcopy
from typing import Union, Optional
from munch import Munch
import os,sys

sys.path = list(set(sys.path + [os.getenv('PWD')])) 
asyncio.set_event_loop(asyncio.new_event_loop())
import tuwang
from tuwang.server import ServerModule
import streamlit as st
import bittensor
from tuwang.model.remote.utils import encode_topk, decode_topk




class RemoteModelClient:
    def __init__(self,
                model=None,
                ip:str='0.0.0.0', 
                port:int=50056 ,
                tokenizer:Union[str, 'tokenizer'] = None):

        # Set up the tokenizer
        tokenizer = model if isinstance(model, str) else tokenizer
        self.set_tokenizer(tokenizer=tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        self.std_tokenizer = bittensor.tokenizer()


        # Setup the Client
        self.ip = ip
        self.port = port
        self.client = tuwang.server.ClientModule(ip=self.ip, port=self.port)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states:bool = True, topk:int=4096 ):
        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': output_hidden_states,
            'topk': topk
        }
        response = self.client.forward(data=data, timeout=10)

        try:
            response_dict = response['data']
            # if topk:
            ouput_dict = {}
            if 'topk' in response_dict:
                ouput_dict['logits'] = decode_topk(response_dict['topk'], topk=topk, vocab_size=self.vocab_size)

            return Munch(ouput_dict)

        except Exception as e:
            st.write(response)
            raise e 



    __call__ = forward 


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer == None:
            tokenizer = bittensor.tokenizer()

        self.tokenizer = tokenizer
        
        
        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        return self.tokenizer


    def test_forward(self, sequence_length=256, batch_size=32):
        dummy_token = 'hello'
        input_text = ' '.join([dummy_token for i in range(sequence_length)])
        input_text_batch = [input_text]*batch_size

        msg = f'''TEST:FORWARD (batch_size: {batch_size}, sequence_len: {sequence_length})) PASSED'''
        print(msg)


    @classmethod
    def streamlit(cls):

        raw_text = ['hey whats up', 'hey sup']
        
        model_client = cls()
        token_batch = model_client.std_tokenizer(raw_text, max_length=256, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]

        st.write(f'## {cls.__name__}')
        st.write('LOCATION: ', __file__)
        

        with st.expander('STEP 1: GET RAW TEXT', True):
            input_text = st.text_area('Input Text', 'null')
            sequence_length = st.select_slider('Batch Size', list(range(0, 512, 32)), 256)
            if input_text  == 'null':
                dummy_token = 'hello'
                input_text = ' '.join([dummy_token for i in range(sequence_length)])
            batch_size = st.select_slider('Batch Size', list(range(8, 128, 8)), 32)
            input_text_batch = [input_text]*batch_size

        # with st.expander('STEP 2: TOKENIZE RAW TEXT INTO TENSOR', False):
        #     st.write('INPUT TOKENS',input)
        #     st.write('SHAPE ',torch.tensor(input['input_ids']).shape)
        with st.expander('OUTPUT', True):
            from tuwang.utils import Timer  
            input = dict(model_client.tokenizer(input_text_batch))
            import time
            with Timer() as t:
                output = model_client(**input)
                time_elapsed = t.seconds

                st.write(output)
                st.write('SHAPE ',output['logits'].shape)
                st.write('Time Elapsed: ', time_elapsed)



if __name__ == "__main__":
    RemoteModelClient.streamlit()



