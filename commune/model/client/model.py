from transformers import AutoTokenizer, AutoConfig
import asyncio
import torch
from torch import nn
from pprint import pp

from copy import deepcopy
from typing import Union, Optional
from munch import Munch
import os,sys
import bittensor
import tuwang
from tuwang import Module

# import streamlit as st
# from tuwang.model.utils import encode_topk, decode_topk
from bittensor.utils.tokenizer_utils import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases

class ModelClient(Module, nn.Module):
    def __init__(self,
                model:str = 'model.dendrite',
                tokenizer:Union[str, 'tokenizer'] = None,
                device:str='cuda',
                output_length:int = 8,
                input_length:int = 256,
                topk:int = 4096
                ):
        
        nn.Module.__init__(self)
    
        # # Set up the tokenizer
        self.device = device
        self.topk = topk
        self.output_length = output_length
        self.input_length = input_length
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.set_model(model)
        self.set_tokenizer(tokenizer=tokenizer if tokenizer else self.model_name)


    def set_model(self, model:str):
        self.model = self.connect(model)

        self.model_name = self.model(fn= 'getattr', kwargs = {'k': 'model_name'})
        self.config = Munch(self.model(fn= 'getattr', kwargs = {'k': 'config'}))
        self.config.hidden_size = self.config.get('hidden_size')
        
        
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None, 
                output_hidden_states:bool = False, 
                topk:int=None, 
                verbose:bool = True,
                input_length: int = None,
                output_logits: bool = True,
                server_output_logits: bool = False,
                output_length:int=None,
                **kwargs):

        topk = topk if topk else self.topk
        output_length = output_length if output_length else self.output_length
        input_length = input_length if input_length else self.input_length
        input_ids = input_ids[:,-input_length:]
        attention_mask =  attention_mask[:,-input_length:] if attention_mask != None else attention_mask
        
        
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': False,
            'topk': topk,
            'output_length': output_length,
            'output_logits': server_output_logits,
            'verbose': verbose,
            **kwargs
        }
        # import ipdb; ipdb.set_trace()
        response_dict = self.model(fn='forward', kwargs=deepcopy(model_kwargs))
        # if topk:
        ouput_dict = {}
        if 'topk' in response_dict:
            output_dict = response_dict['topk']
        if output_logits:
            ouput_dict['logits'] = self.decode_topk(response_dict['topk'].to(torch.float64), vocab_size=self.vocab_size)
            
        
        return Munch(ouput_dict)

    __call__ = forward

    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        if isinstance(tokenizer, str):
            if tokenizer == 'model.dendrite':
                tokenizer = bittensor.tokenizer()
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except ValueError:
                    print('resorting ot use_fast = False')
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

        self.tokenizer = tokenizer     
        self.std_tokenizer = bittensor.tokenizer()
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        




        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def test_forward(self, sequence_length=256, batch_size=32):
        dummy_token = 'hello'
        input_text = ' '.join([dummy_token for i in range(sequence_length)])
        input_text_batch = [input_text]*batch_size

        msg = f'''TEST:FORWARD (batch_size: {batch_size}, sequence_len: {sequence_length})) PASSED'''
        print(msg)

    @staticmethod
    def decode_topk(  forward_response_tensor: torch.Tensor,  vocab_size:int=bittensor.__vocab_size__) -> torch.Tensor:
        """ Returns full logits by decoding topk-encoding input. """
        encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
        
        if len(encoded_probs.shape) == 3:
            encoded_probs = torch.stack(encoded_probs.chunk(chunks=2, dim=-1), dim=-1)
        
        batch_size, sequence_len, topk, _ = encoded_probs.shape

        topk_values = encoded_probs[..., 0]  # topk probs: [batch_size, sequence_len, topk]
        topk_indices = encoded_probs[..., 1].long()  # topk probs indices: [batch_size, sequence_len, topk]
        
        if topk_indices[0, 0,-1] == -100:
            topk_values = topk_values[..., :-1]
            topk_indices = topk_indices[..., :-1]
            
        max_index = torch.max(topk_indices).item()
        vocab_size = max(vocab_size, max_index + 1)
        topk_values = topk_values
        topk_indices = topk_indices
        
        
            
        topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
        remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
        remainder_floor = remainder_pmass / (vocab_size - topk)  # divide remainder: [batch_size, sequence_len]

        logits = torch.ones((batch_size, sequence_len, vocab_size), dtype=topk_values.dtype).to(topk_values.device)
        logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]

        
        logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

        return logits  # [batch_size, sequence_len, vocab_size]

    @classmethod
    def sandbox(cls, batch_size= 32, sequence_length=256 , num_batches=10):
        
#        import ipdb; ipdb.set_trace()
        self = cls(model='model.dendrite')
        dataset = tuwang.connect('dataset.bittensor')

        

        from tuwang.utils import Timer
        import time
        
        
        for i in range(num_batches):
            t = Timer()
            sample = dataset(fn='sample')
            targets = sample['input_ids'][:, -1:]
            sample['input_ids'] =  sample['input_ids'][:, :-1]
            t = tuwang.timer()
            # import ipdb; ipdb.set_trace()
            # sample['num_endpoints'] =  40
            pred = self(**sample)
            

            loss = self.loss(pred['logits'].reshape(-1, pred['logits'].size(-1)), targets.reshape(-1))
            print('SHAPE ',pred['logits'].shape,'Time: ', t.seconds, ' Loss: ', loss.item())
            
    @classmethod
    def test_client(cls, batch_size= 8, sequence_length=256 ):
#        import ipdb; ipdb.set_trace()
        self = cls()
        raw_text = ['hey whats up']*batch_size
        token_batch = self.tokenizer(raw_text, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")

        from tuwang.utils import Timer
        input = dict(token_batch)
        import time
        with Timer() as t:
            # import ipdb; ipdb.set_trace()
            print(token_batch['input_ids'].shape)
            output = self(**input)
            time_elapsed = t.seconds
            # print("OUTPUT")
            # print(output)
            print('SHAPE ',output['logits'].shape)
            print('Time Elapsed: ', time_elapsed)

    def get_loss_fct(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:

        """
        Calculate loss_fct, CausalLM loss, next-token prediction loss.
            Args:
                logits (:obj:`torch.FloatTensor`, `required`):
                    [batch_size, sequence_len, bittensor.__network_dim__]
                labels (:obj:`torch.LongTensor`, `required`):
                    [batch_size, sequence_len]

            Returns:
                loss (:obj:`torch.FloatTensor`):
                    scalar
        """
        shift_logits = logits[..., :-1, :]
        logits_seq_len = logits.shape[1]
        shift_labels = labels[..., -logits_seq_len:]
        shift_labels = shift_labels[..., 1:]
        print(f'LOGITS: {shift_logits.shape} LABELS: {shift_labels.shape}')
        loss = self.loss(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        return loss




    # @classmethod
    # def argparse(cls):
    #     import argparse
    #     parser =  argparse.ArgumentParser()
    #     parser.add_argument('--port', type=int, help=f'''Port''', default = 50050)
    #     parser.add_argument('--model_name', type=str, help=f'''Port''', default = 'EleutherAI/gpt-j-6B')


    #     args = parser.parse_args()
    #     # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    #     return args


    @classmethod
    def test_performance(cls, batch_size= 32, sequence_length=256):

        self = cls.default_model()
        raw_text = ['Hello, my name is boby and I want to have a good time']*batch_size
        token_batch = self.tokenizer(raw_text, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")

        from tuwang.utils import tensor_info_dict, tensor_info, Timer
        input = dict(token_batch)
        import time
        with Timer() as t:
            # import ipdb; ipdb.set_trace()
            print('INPUT SCHEMA')
            print(tensor_info_dict(input))
            output = self(**input)
            print('OUTPUT SCHEMA')
            print(tensor_info_dict(output.__dict__))
            print('TIME (s): ', t.seconds)
            print(self.get_loss_fct(logits=output.logits, labels=token_batch.input_ids))


    @classmethod
    def default_model(cls):
        self = cls()
        return self

    @classmethod
    def test_neuron(cls, batch_size=32, sequence_length=12, topk=4096):
        from tuwang.neuron.miner import neuron
        
        self = cls.default_model()
        print(self.state_dict())
        nucleus = neuron(model=self).model
        nucleus.model.train()
        nucleus.model.eval()
        nucleus.model.half()
        nucleus.model.config.hidden_size
        nucleus.model.config.pad_token_id
        nucleus.model.config.eos_token_id
        nucleus.model.named_parameters()
        state_dict = nucleus.model.state_dict()
        print(nucleus.device, 'DEBUG')
        nucleus.model.load_state_dict(state_dict)
        raw_text = ['Hello, my name is boby and I want to have a good time']*batch_size
        inputs_x = self.tokenizer(raw_text, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt").input_ids.T
        nucleus.encode_forward_causallmnext(inputs_x, topk=topk)
 
 
    @classmethod
    def run_neuron(cls):
        import bittensor
        from tuwang.neuron.miner import neuron
        self = cls()
        n = neuron(model=self)  
        n.run()
 
if __name__ == "__main__":
    
    # ModelClient.run()
    
    ModelClient.run_neuron()