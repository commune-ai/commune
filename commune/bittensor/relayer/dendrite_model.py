import torch
import os,sys
import asyncio
from transformers import AutoConfig, PreTrainedTokenizerBase
# import streamlit as st


from typing import List, Union, Dict
from munch import Munch

import commune
commune.new_event_loop()

from commune.block.bittensor.receptor import receptor_pool
import bittensor
from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer

from copy import deepcopy
class DendriteModel(torch.nn.Module, commune.Module):
    
    def __init__(self,
                uids : List[int] = [441],
                wallet:bittensor.wallet = None,
                tokenizer: bittensor.tokenizer = None,
                subtensor: bittensor.subtensor = None,
                metagraph: bittensor.metagraph = None,
                model_name:str = 'model.dendrite',
                metric:str = 'incentive',
                hidden_size = 8
                ):
        
        torch.nn.Module.__init__(self)
        
        self.metric = metric
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.loop = self.set_event_loop(new_loop=True)
        
        
        self.set_event_loop(new_loop=True)
        
        self.model_name = model_name
        self.model_config = Munch(AutoConfig.from_pretrained('gpt2').__dict__)
        self.model_config.hidden_size = hidden_size
        
        self.wallet = wallet if wallet else self.default_wallet()
        self.tokenizer = tokenizer if tokenizer else bittensor.tokenizer()
        self.tokenizer = prep_tokenizer(self.tokenizer, self.tokenizer)
        self.subtensor = subtensor if subtensor else self.default_subtensor()
        
        self.relay_neurons = [self.subtensor.neuron_for_uid(uid) for uid in uids]
        
        self.endpoints = [bittensor.endpoint.from_neuron(neuron) for neuron in self.relay_neurons]
        self.metagraph = metagraph if metagraph else bittensor.metagraph(subtensor=self.subtensor)
        # self.wallet = self.wallet.register(cuda=True ,subtensor=self.subtensor)
        self.metagraph= self.metagraph.load() 
        self.metagraph.sync() 

        self.receptor_pool = receptor_pool(wallet=self.wallet)
        
        
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
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None, 
                output_hidden_states:bool = False, 
                output_logits:bool = True, 
                num_endpoints:int = 20 ,
                topk: int = 4096,
                timeout: int = 3,
                max_trials: int = 1,
                max_responses: int = 10,
                min_successes = 5,
                **kwargs
                ):
        
        
        endpoints = deepcopy(self.endpoints)
        if num_endpoints > 0:
            endpoints += self.top_endpoints(n=num_endpoints)
        atleast_one_success = False
        
        trial_count = 0
        t = commune.timer()
        
        commune.print(endpoints)
        
        while not atleast_one_success and trial_count < max_trials:
            response = self.receptor_pool.forward(inputs=[input_ids]*len(endpoints) , 
                                                        endpoints=endpoints,
                                                        synapses=[bittensor.synapse.TextCausalLMNext()],
                                                        timeout=timeout,
                                                        min_successes=min_successes)
            
            atleast_one_success = any([any([c==1 for c in codes]) for codes in response[1]])
            trial_count += 1
            commune.print(f'Endpoints: {self.endpoints}', color='purple')
            
            commune.print(f'Responses from Server Codes: {response[1]}', color='yellow')
        

        
        
        response_tensors = []
        success_count = 1
        code_count_dict = {}
        max_responses = max_responses if max_responses else num_endpoints
        for i in range(len(response[1])):
            # st.write(response[1][i][0])
            
            # assume the codes are all the same for the endpoint (not a super safe assumption but good for now)
            code  = response[1][i][0]
            
            
            
            if code in code_count_dict:
                code_count_dict[code] += 1
            else:
                code_count_dict[code] = 1
            
            if code == 1:
                response_tensors += [response[0][i][0]]
                # if len(response_tensors)>=max_responses:
                #     break


        
                
        metrics = {}
        metrics['num_successes'] = len(response_tensors)
        metrics['num_endpoints'] = len(endpoints)
        metrics['success_rate'] = metrics['num_successes'] / (metrics['num_endpoints'] + 1e-8)
        metrics['seconds'] = t.seconds
        assert  metrics['num_successes'] > 0 , f'{code_count_dict}'
        
        
        
        print('CODE COUNTER: ', code_count_dict) 
        output_dict = {}
        logits = self.mix_response(response_tensors)

        if output_logits:
            logits = self.mix_response(response_tensors)
            output_dict['logits'] = logits
        
        commune.print(len(response_tensors), 'green')
        if topk:   
            if len(response_tensors) > 1:
                output_dict['topk'] = self.encode_topk(logits[..., -1:, :])
            else:
                output_dict['topk'] = response_tensors[0].unsqueeze(1)

            
        # if output_hidden_states:
        #     raise NotImplemented(f'output_hidden_states = {output_hidden_states}')

        return output_dict
        
    def get_best_endpoints(self):
        return self.endpoints
        
    def get_loss_fct(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if not hasattr(self, 'loss'):
            self.loss = torch.nn.CrossEntropyLoss()

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

    
    @classmethod
    def default_subtensor(cls): 
        return bittensor.subtensor(chain_endpoint=os.getenv('SUBTENSOR'))


    @classmethod
    def default_wallet(cls):
        return bittensor.wallet(name='fish', hotkey='100')
        
    @classmethod
    def default_model(cls):
        self = cls(uids = [441])   
        return self

    @classmethod
    def test_performance(cls, batch_size= 32, sequence_length=256, num_batches=100):

        import time
        
        self = cls.default_model()
        dataset = cls.connect('dataset::bittensor')
        for i in range(num_batches):
            sample = dataset.sample()
            input_ids = sample['input_ids'][:, :-1]
            targets = sample['input_ids'][:,-1:]

            t = commune.timer()
            output = self(input_ids=input_ids)
            print('OUTPUT SCHEMA')
            print('TIME (s): ', t.seconds)
            # print(self.get_loss_fct(logits=pred['logits'], labels=input_ids))
            print(output['topk'].shape, targets.shape)
            print(phrase_cross_entropy(topk_tensor=output['topk'][:,-1], target_phrases=targets))


    @classmethod
    def test(cls): 
        cls.test_performance()
        cls.test_neuron()  
        
        
    @classmethod
    def run_neuron(cls):
        from commune.neuron.miner import neuron
        model = cls.default_model()
        n = neuron(model=model)  
        n.run()
        
    @classmethod
    def mix_response(cls, responses:List[torch.Tensor], routing_scores:list= None):
        
        batch_size = responses[0].shape[0]
        num_models = len(responses)
        device = responses[0].device
        
        if routing_scores == None: 
            routing_scores = torch.full(fill_value=1/num_models,size=(num_models,batch_size)).to(device)
        # st.write(routing_scores)
        
        stacked_responses = torch.cat([ r[...,:-1, :2] for r in responses], dim=0)
        decoded_tensors = cls.decode_topk(stacked_responses[:, None]) # [batch_size, 1, sequence_len, vocab_size]
        print(decoded_tensors.shape, routing_scores.shape, routing_scores.sum(dim=0))

        decoded_tensors = torch.stack(decoded_tensors.chunk(chunks=num_models, dim=0))
        
        # shappe of decoded_tensors: [num_models, batch_size, sequence_len, vocab_size]
       

        merged_tensors = (routing_scores[:,:,None,None]*decoded_tensors).sum(dim=0)
        
        # merged_tensors: [batch_size, sequence_len, vocab_size] -> [batch_size, vocab_size]
        return  merged_tensors



    @staticmethod
    def decode_topk(  forward_response_tensor: torch.Tensor,  vocab_size:int=bittensor.__vocab_size__) -> torch.Tensor:
        """ Returns full logits by decoding topk-encoding input. """
        encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
        
        if len(encoded_probs.shape) == 3:
            encoded_probs = torch.stack(encoded_probs.chunk(chunks=2, dim=-1), dim=-1)
        
        batch_size, sequence_len, topk, _ = encoded_probs.shape

        topk_values = encoded_probs[..., 0]  # topk probs: [batch_size, sequence_len, topk]
        topk_indices = encoded_probs[..., 1].long()  # topk probs indices: [batch_size, sequence_len, topk]
        
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



    @staticmethod
    def encode_topk( forward_response_tensor: torch.Tensor , topk:int=4096, compact:bool = False) -> torch.Tensor:
        """ Returns topk tokens/probabilities given unnormalized logits as input. """

        #import ipdb; ipdb.set_trace()

        logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
        probs = torch.softmax(logits, dim=-1).to(torch.float32)  # normalized probabilities: [batch_size, sequence_len, vocab_size]

        topk_indices = torch.argsort(probs, dim=-1, descending=True)[...,:topk]
        # topk_values, topk_indices = torch.topk(probs, topk) # topk probs and indices: [batch_size, sequence_len, topk]

        topk_values = probs.gather( index=topk_indices, dim=-1)
        if compact:
            encoded_probs = torch.cat([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
        else:
            encoded_probs = torch.stack([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
            
        return encoded_probs  # [batch_size, sequence_len, topk + topk]


    # def run_pm2()
        
if __name__ == "__main__":
    
    DendriteModel.run()

    



