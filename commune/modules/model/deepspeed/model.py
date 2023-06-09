import os, sys
from pprint import pp
from functools import partial
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
from copy import deepcopy
from typing import Union, Optional
from concurrent import futures
import os, sys
from typing import *
from loguru import logger
import time
from munch import Munch
import argparse
# logger = logger.opt(colors=True)


if os.getenv('USE_STREAMLIT') == 'true':
    import streamlit as st
    
import bittensor
from bittensor.utils.tokenizer_utils import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases
    
import commune
import torch
from commune.server import Server
from commune.model.utils import encode_topk, decode_topk
from commune.utils.torch import *
import commune


# example models

# model_name: str="EleutherAI/gpt-j-6b",
# model_name: str="/nvme/models/gpt-j-6B",
# model_name: str="EleutherAI/gpt-neo-125M",
class DeepspeedModel(commune.Module):
    port_range = [50050, 50100] # the range of ports the moddule can be a server for
    default_ip = '0.0.0.0'
    def __init__(self,
    
                model_name: str= "bigscience/bloom-560m",
                tokenizer:Optional[Union[str, 'tokenizer']]= None,
                ip: str = '0.0.0.0',
                port: int = 50057, # chooses a random port if None within the port_range
                serve:bool = False, # serve the self.server
                device: Optional[str] = None, # defaults to self.model.device
                timeout: Optional[int] = None, # TODO: not used
                compression:Optional[str] = None, # 
                
                # deepspeed params
                task_name:str='text-generation',
                model_path:str = 'model',
                mii_configs:dict={"tensor_parallel": 4, "dtype": "fp16"},

                ):

        # set model (tokenizer is also included)
        self.set_model(
            model_name=model_name, 
            task_name=task_name,
            model_path=model_path, 
            mii_configs=mii_configs,
            device=device
        )

        # set the tokenizer to override the model or defaults to model's tokenizer
        self.set_tokenizer(tokenizer=tokenizer)

        # serve the model 
        if serve:
            self.serve()

    def resolve_ip(self, ip:str)->str:
        if ip == None:
            if hasattr(self, 'ip'):
                ip = self.ip
            else:
                ip = self.default_ip

        assert isinstance(ip, str)
        
        return ip

    def forward(self, *args,no_grad=True, **kwargs):
        # import ipdb; ipdb.set_trace()
        if no_grad:
            with torch.no_grad():
                result = self._forward(*args,**kwargs)
        else:
            result = self._forward(*args,**kwargs)
        # import ipdb; ipdb.set_trace()
        return result

    def _forward(self,  
                token_batch: torch.Tensor = None, 
                attention_mask: torch.Tensor= None, 
                output_hidden_states:bool=True, 
                topk:int=None, 
                verbose:bool = True,
                output_length:int = 1,
                **kwargs):

        # import ipdb; ipdb.set_trace()

        input_dict = dict(
                    token_batch=token_batch,
                    attention_mask=attention_mask,
                    output_hidden_states= output_hidden_states
                    )

        # ensure the token_batch and attention mask is a tensor
        for k in ['token_batch', 'attention_mask']:
            v = input_dict[k]
            if isinstance(v,  list):
                input_dict[k] = torch.tensor(v)
            elif isinstance(v, type(None)):
                del input_dict[k]
                continue
            if isinstance(v,  torch.Tensor):
                input_dict[k] = input_dict[k].to(self.device)

        if verbose:
            print('INPUT_STATISTICS: ',tensor_info_dict(input_dict))

        model_output = self.model(**input_dict)

        if topk:
            topk_tensor = self.encode_topk(model_output.logits[:,-output_length:,:], topk=topk)
            output_dict = dict(topk=topk_tensor)
        else:
            output_dict = dict(logits=model_output.logits[:,-output_length:,:])

        if output_hidden_states:
            output_dict['hidden_states'] = model_output.hidden_states[-1]

        if verbose:
            print('OUTPUT_STATISTICS: ',tensor_info_dict(output_dict))

        return output_dict

    def set_model(self, model_name:str="bigscience/bloom-560m",
                 task_name:str='text-generation',
                  model_path:str = 'model',
                mii_configs:dict={"tensor_parallel": 8, "dtype": "fp16"},
                device:str= None):
        from commune.model.deepspeed.server_client import MIIServerClient

        mii_configs['tensor_parallel'] = min(mii_configs['tensor_parallel'],torch.cuda.device_count())

        pipeline = MIIServerClient(task_name=task_name, model_name=model_name, model_path=model_path,
                                    mii_configs=mii_configs, initialize_grpc_client=False, use_grpc_server=False).model
        self.tokenizer = pipeline.tokenizer
        self.model = pipeline.model
        self.model_name = model_name
        self.device = device if device else self.model.module.device


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None] = None):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        elif tokenizer == None:
            tokenizer = self.tokenizer

        self.tokenizer = tokenizer

        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.std_tokenizer = bittensor.tokenizer()
        # self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        
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
            st.write('SHAPE ',torch.tensor(input['token_batch']).shape)

        with st.expander('OUTPUT', True):
            st.write(output)
            st.write('SHAPE ',output['logits'].shape)

    @staticmethod
    def encode_topk( forward_response_tensor: torch.Tensor , topk:int=4096) -> torch.Tensor:
        """ Returns topk tokens/probabilities given unnormalized logits as input. """

        #import ipdb; ipdb.set_trace()

        logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
        probs = torch.softmax(logits, dim=-1).to(torch.float32)  # normalized probabilities: [batch_size, sequence_len, vocab_size]

        topk_indices = torch.argsort(probs, dim=-1, descending=True)[...,:topk]
        # topk_values, topk_indices = torch.topk(probs, topk) # topk probs and indices: [batch_size, sequence_len, topk]

        topk_values = probs.gather( index=topk_indices, dim=-1)
        encoded_probs = torch.cat([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
        return encoded_probs  # [batch_size, sequence_len, topk + topk]


    @classmethod
    def test_model(cls, batch_size=32, sequence_length=256):
        self = cls(serve=False)
        example = ["My name is Philipp and I"]*batch_size
        token_batch = self.tokenizer(example,return_tensors="pt", max_length=sequence_length, padding='max_length').token_batch.to(self.device)
        
        print('TESTING LOGITS OUTPUT')
        logits = self.forward(token_batch, output_hidden_states=True, topk=None,verbose=True)
        
        print('TESTING TOPK OUTPUT')
        logits = self.forward(token_batch, output_hidden_states=True, topk=None,verbose=True)



    @classmethod
    def argparse(cls):
        parser =  argparse.ArgumentParser()
        parser.add_argument('--port', type=int, help=f'''Port''', default = None)
        parser.add_argument('--model', type=str, help=f'''model to server''', default = 'gpt-j' )
        # parser.add_argument('--device', type=str, help=f'''model to server''', default = '0' )

        args = parser.parse_args()

        return args
    model_name_shortcuts = {
        'opt13b': 'facebook/opt-13b',
        'gptj': 'EleutherAI/gpt-j-6b',
        'gpt125m': 'EleutherAI/gpt-neo-125M'
    }
    @classmethod
    def run(cls):
        args =  cls.argparse()
        
        if args.model in cls.model_name_shortcuts:
            args.model = cls.model_name_shortcuts[args.model]
        
        cls(port=args.port, model_name=args.model, serve=True)

if __name__ == "__main__":
    
    DeepspeedModel.run()



# if __name__ == "__main__":
    
#     # DeepspeedModel = 
#     DeepspeedModel(port=50058, serve=True, model_name='EleutherAI/gpt-j-6b')


# def set_model(self, model_name:str="bigscience/bloom-560m",
#              task_name:str='text-generation',
#               model_path:str = 'model',
#             mii_configs:dict={"tensor_parallel": 8, "dtype": "fp16"}):
#     # Filename: gpt-neo-2.7b-generation.py

#     import deepspeed
#     from transformers import pipeline
#     local_rank = int(os.getenv('LOCAL_RANK', '0'))
#     world_size = 2


#     pipeline = pipeline(task_name, model=model_name,
#                         device=local_rank)


#     self.model = deepspeed.init_inference(pipeline.model,
#                                             mp_size=world_size,
#                                             dtype=torch.float,
#                                             replace_method='auto',
#                         replace_with_kernel_inject=True)

#     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#         print(string)

#     self.tokenizer = pipeline.tokenizer
#     self.model_name = model_name

#     return self.model

