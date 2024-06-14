import torch
import os,sys
import asyncio
from transformers import AutoConfig, PreTrainedTokenizerBase
# import streamlit as st


from typing import List, Union, Dict
from munch import Munch

import commune as c
c.new_event_loop()

import bittensor
from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer

from copy import deepcopy

st = c.st()

Bittensor = c.module('bittensor')
class DendriteModel(Bittensor):
    
    def __init__(self,
                wallet:bittensor.wallet = 'ensemble.1',
                subtensor= 'finney',
                tokenizer: bittensor.tokenizer = None,
                metric:str = 'incentive',
                hidden_size = 8,
                netuid=3,
                ):

        Bittensor.__init__(self, wallet=wallet, subtensor=subtensor, netuid=netuid)
        
        
        

        # self.wallet = self.wallet.register(cuda=True ,subtensor=self.subtensor)
        self.metagraph= self.metagraph.load() 
        self.metagraph.sync(netuid=netuid)
        
        # self.sync() 
        self.metric = metric

        self.receptor_pool = bittensor.receptor_pool(wallet=self.wallet, )
        
        st.write(self.top_endpoints())
        
        
    def set_endpoints(self, endpoints: Union[str,bittensor.Endpoint]) -> List[str]:
        self.endpoints = []
        for e in endpoints:
            if isinstance(e, bittensor.Wallet): 
                e = bittensor.endpoint.from_neuron(e.get_neuron())
                self.endpoints.append(e)
            elif isinstance(e,bittensor.Endpoint):
                self.endpoints.append(e)
            else:
                raise NotImplemented
            
        return self.endpoints
        
    def top_endpoints(self, n:int=30, metric=None):
        metric = metric if metric else self.metric
        top_uid_indices = torch.argsort(getattr(self.metagraph, metric), descending=True)[:n]
        endpoints = self.metagraph.endpoint_objs
        return  [endpoints[i] for i in top_uid_indices]
    
    
    
    def process_responses(self, responses):
        return responses[0]
    
    def encode_forward_causallmnext(self, token_batch, std_tokenizer=None, topk: int = 4096, model_output=None):
        return self.forward(input_ids=token_batch, topk=topk)
    def forward(self, 
                input_ids: torch.Tensor, 
                topk: int = 4096,
                attention_mask: torch.Tensor = None, 
                output_hidden_states:bool = False, 
                output_logits:bool = True, 
                num_endpoints:int = 100 ,
                timeout: int = 6,
                max_trials: int = 1,
                max_responses: int = 10,
                verbose:bool = True,
                **kwargs
                ):
        endpoints = self.top_endpoints(n=num_endpoints)
        atleast_one_success = False
        
        trial_count = 0
        t = c.timer()
        
        c.print(f'Endpoints: {endpoints}', color='purple')
        while trial_count < max_trials:
            response = self.receptor_pool.forward(inputs=[input_ids]*len(endpoints) , 
                                                        endpoints=endpoints,
                                                        synapses=[bittensor.synapse.TextCausalLMNext(topk=topk)],
                                                        timeout=timeout)
            # are any of these successes
            if any([any([c==1 for c in codes]) for codes in response[1]]):
                c.print(f'Responses from Server Codes: {response[1]}', color='yellow')
        
            else:
                trial_count += 1


        info = {
            'success_count': 1,
            'code_count': {},
            'max_responses' : max_responses if max_responses else num_endpoints,
            'num_successes': 0,
            'num_endpoints': 0,
            'seconds': 0,
        }
        
        response_tensors = []
        
        for i in range(len(response[1])):
            code  = response[1][i][0]
            if code in info['code_count']:
                info['code_count'][code] += 1
            else:
                info['code_count'][code] = 1
            if code == 1:
                response_tensors += [response[0][i][0]]

                
        info['num_successes'] = len(response_tensors)
        info['num_endpoints'] = len(endpoints)
        info['success_rate'] = info['num_successes'] / (info['num_endpoints'] + 1e-8)
        info['seconds'] = t.seconds
        assert  info['num_successes'] > 0 , f"{info['code_count']}"
        if verbose:
            c.print('INFO: ', info['code_count']) 
            
        output = self.process_responses(response_tensors)
        return output
        

    @classmethod
    def test(cls):
        
        dataset = c.connect('dataset')
        sample = dataset.sample()
        
        c.print(sample['input_ids'].shape)
        model = cls()

        output = model.forward(torch.cat([sample['input_ids'] for i in range(5)])[:32])
        st.write(output)


    # def run_pm2()
        
if __name__ == "__main__":
    
    DendriteModel.test()

    



