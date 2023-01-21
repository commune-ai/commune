import torch
import os,sys
import asyncio
loop = asyncio.get_event_loop()
from transformers import AutoConfig, PreTrainedTokenizerBase
# import streamlit as st
# asyncio.set_event_loop(asyncio.new_event_loop())

import bittensor
from typing import List, Union, Dict
from munch import Munch
from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer

import tuwang
from tuwang.receptor import receptor_pool
from copy import deepcopy
class DendriteModel(torch.nn.Module, tuwang.Module):
    
    def __init__(self,
                endpoints : List[Union[str, 'bittensor.endpoint']] = [],
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
        self.config = Munch(AutoConfig.from_pretrained('gpt2').__dict__)
        self.config.hidden_size = hidden_size
        self.set_endpoints(endpoints)
        self.wallet = wallet if wallet else self.default_wallet()
        self.tokenizer = tokenizer if tokenizer else bittensor.tokenizer()
        self.tokenizer = prep_tokenizer(self.tokenizer, self.tokenizer)
        self.subtensor = subtensor if subtensor else self.default_subtensor()
        
        self.metagraph = metagraph if metagraph else bittensor.metagraph(subtensor=self.subtensor)
        # self.wallet = self.wallet.register(cuda=True ,subtensor=self.subtensor)
        self.metagraph= self.metagraph.load() 
        # self.metagraph.sync() 

        self.receptor_pool = bittensor.receptor_pool(wallet=self.wallet)
        
        
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
                num_endpoints:int = 30,
                topk: int = 4096,
                timeout = 3,
                max_trials = 1,
                max_responses: int = 1,
                **kwargs
                ):
        
        
        endpoints = self.top_endpoints(n=num_endpoints) + self.endpoints
        atleast_one_success = False
        
        trial_count = 0
        t = tuwang.timer()
        
        print(input_ids, 'DEBUG')
        
        while not atleast_one_success and trial_count < max_trials:
            response = self.receptor_pool.forward(inputs=[input_ids]*len(endpoints) , 
                                                        endpoints=endpoints,
                                                        synapses=[bittensor.synapse.TextCausalLMNext()],
                                                        timeout=timeout)
            
            atleast_one_success = any([any([c==1 for c in codes]) for codes in response[1]])
            trial_count += 1
        

        
        
        response_tensors = []
        success_count = 1
        code_count_dict = {}
        max_responses = max_responses if max_responses else num_endpoints
        for i in range(len(response[1])):
            # st.write(response[1][i][0])
            
            # assume the codes are all the same for the endpoint (not a super safe assumption but good for now)
            code  = response[1][i][0]
            if code == 1:
                response_tensors += [response[0][i][0]]
                if len(response_tensors)>=max_responses:
                    break


            if code in code_count_dict:
                code_count_dict[code] += 1
            else:
                code_count_dict[code] = 1
        
                
        metrics = {}
        metrics['num_successes'] = len(response_tensors)
        metrics['num_endpoints'] = len(endpoints)
        metrics['success_rate'] = metrics['num_successes'] / metrics['num_endpoints']
        metrics['seconds'] = t.seconds
        assert  metrics['num_successes'] > 0 , f'{code_count_dict}'
        
        
        
        logits = self.mix_response(response_tensors)
        output_dict = {}
        
        print('CODE COUNTER: ', code_count_dict) 
        
        if topk:   
            if len(response_tensors) > 1:
                output_dict['topk'] = self.encode_topk(logits[..., -1:, :])
            else:
                output_dict['topk'] = response_tensors[0].unsqueeze(1)
        if output_logits:
            output_dict['logits'] = logits
            
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
        return bittensor.subtensor()


    @classmethod
    def default_wallet(cls):
        return bittensor.wallet(name='fish', hotkey='100')
        
    @classmethod
    def default_model(cls):
        self = cls()   
        return self

    @classmethod
    def test_performance(cls, batch_size= 32, sequence_length=256):

        from tuwang.utils import tensor_info_dict, tensor_info, Timer
        import time
        
        self = cls.default_model()
        raw_text = ['Hello, my name is boby and I want to have a good time']*batch_size
        input_ids = torch.tensor(self.tokenizer(raw_text, max_length=sequence_length+1, truncation=True, padding="max_length", return_tensors="pt").input_ids)
        
        targets = input_ids[:, -1]
        input_ids = input_ids[:,:-1]

        with Timer() as t:
            # import ipdb; ipdb.set_trace()
            # st.write('INPUT SHAPE')
            # print(tensor_info_dict(input))
            output = self(input_ids=input_ids)
            # st.write(output)
            print('OUTPUT SCHEMA')
            # print(tensor_info_dict(output.__dict__))
            print('TIME (s): ', t.seconds)
            print(self.get_loss_fct(logits=pred['logits'], labels=sample['input_ids']))
            # print(phrase_cross_entropy(topk_tensor=output['topk'], target_phrases=targets))




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
        nucleus.model.load_state_dict(state_dict)
        raw_text = ['Hello, my name is boby and I want to have a good time']*batch_size
        inputs_x = self.tokenizer(raw_text, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt").input_ids.to('cuda:0')
        nucleus.encode_forward_causallmnext(inputs_x, topk=topk)
  
    @classmethod
    def test(cls): 
        cls.test_performance()
        cls.test_neuron()  
        
        
    @classmethod
    def run_neuron(cls):
        from tuwang.neuron.miner import neuron
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
    def topk_token_phrases(logits: torch.Tensor, tokenizer: PreTrainedTokenizerBase,
                        topk: int=4096, ignore_index: int = -100) -> torch.Tensor:
        r"""
        Select topk tokenizer logits/phrases and include std_token_phrases counterparts (std_tokenization of token text)
        in topk_tensor output of shape [batch_size, (topk + 1), max_len], where max len of all phrase lists
        (with prob in front) is max_{b,k}(len([prob_k, tok_0_k, tok_1_k, ...])).
        The output topk_tensor also includes a floor_prob for each batch item. The floor probability is the
        mean probability of token phrases not captured in topk, required since the tokenizer vocab_size may
        not be known to the receiver.
        Requires prep_tokenizer(tokenizer, std_tokenizer) to set_std_token_phrases first, to make
        std_token_phrases available here.
            Args:
                logits (:obj:`torch.Tensor`, `required`):
                    [batch_size, vocab_size] Input source logits for last token over a source tokenizer vocabulary.
                tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    Source tokenizer (usually server tokenizer)
                topk (:obj:`int`, `required`):
                    Amount of top phrases to expect (to check for mismatch)
                ignore_index (:obj:`int`, `optional`):
                    Padding value to use for unfilled token positions in a shorter token phrase.

            Returns:
                topk_tensor (:obj:`torch.Tensor`, `required`):
                    [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                    in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                    Content structure:
                    [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                    [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                    [...],
                    [prob_floor_b=0, ignore_index, ..., ignore_index]],
                    [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                    [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                    [...],
                    [prob_floor_b=1, ignore_index, ..., ignore_index]],
                    [...]]
        """
        # Get shape sizes
        batch_size, vocab_size = logits.shape  # [batch_size, vocab_size] only last token prediction

        # Convert logits to probabilities
        logits = logits.float()  # ensure further computations done in float32 for improved precision
        probs = torch.softmax(logits, dim=1)  # [batch_size, vocab_size]

        # TopK phrase selection
        topk_probs, topk_indices = torch.topk(probs, topk)  # topk probs and indices: [batch_size, topk]

        # === Calculate floor probability ===
        topk_pmass = topk_probs.sum(dim=-1)  # [batch_size] topk probability mass
        remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # [batch_size] remainder probability mass
        floor_probs = remainder_pmass / (vocab_size - topk)  # [batch_size]divide remainder

        # convert to list for faster iteration in list comprehension
        topk_probs_list = topk_probs.tolist()
        topk_indices_list = topk_indices.tolist()
        floor_probs_list = floor_probs.tolist()

        # === Construct topk phrases list ===
        probs = []  # collect probability tensors with gradients attached (to be grafted into topk_tensor)
        phrases = []  # form topk token phrases with prob prepend [prob, tok_0, tok_1, ... tok_n]

        for b in range(batch_size):
            # collect probability tensors with gradients attached (to be grafted into topk_tensor)
            probs += [topk_probs[b], floor_probs[b]]  # [tensor(prob_k=0_b, prob_k=1_b, ...), tensor(prob_floor_b)]

            # form topk token phrases with prob prepend [prob, tok_0, tok_1, ... tok_n]
            phrases += [[prob] + tokenizer.std_token_phrases[i]
                        for prob, i in zip(topk_probs_list[b], topk_indices_list[b])]  # [prob_k, tok_0_k, tok_1_k, ...]

            # also add prob_floor for batch item
            phrases += [[floor_probs_list[b]]]  # [prob_floor_b]

        # determine width of topk_tensor as max len of all phrase lists (with prob in front)
        max_len = max([len(p) for p in phrases])  # max_{b,k}(len([prob_k, tok_0_k, tok_1_k, ...]))

        # form single 2D tensor with all phrase and probs (typically to send to axon wire encoding)
        topk_tensor = torch.tensor([p + [ignore_index] * (max_len - len(p))
                                    for p in phrases]).to(logits.device)  # [batch_size * (topk + 1), max_len]

        # grafting probability tensors into first column to attach gradients
        topk_tensor[:, 0] = torch.hstack(probs)  # tensor([prob_k=0_b, prob_k=1_b, ..., prob_floor_b])

        topk_tensor = topk_tensor.reshape(batch_size, topk + 1, max_len)  # [batch_size, (topk + 1), max_len] reshaped

        return topk_tensor  # [batch_size, (topk + 1), max_len] (probability gradients attached in first column)



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

    @classmethod
    def sandbox(cls):
        self = cls(endpoints = [])
        
        data = tuwang.connect('dataset.bittensor')
        sample = data(fn='sample', kwargs=dict(batch_size=32, sequence_length=256))
        targets = sample['input_ids'][:, -1:] 
        sample['input_ids'] = sample['input_ids'][:, :-1] 
        # model = cls.connect('model.transformer::gptj:3')
        sample['topk'] = 4096
        sample['output_logits'] = True
        # sample['autocast'] = True
        t = tuwang.timer()
        pred = self(**sample)        
        print(t.seconds)

        print(phrase_cross_entropy(topk_tensor=pred['topk'][:,0,:,:], target_phrases=targets))
        
        print(pred['logits'].shape)
        
        print(self.loss(pred['logits'].reshape(-1, pred['logits'].size(-1)), targets.flatten()))


    # def run_pm2()
        
if __name__ == "__main__":
    
    DendriteModel.run()

    



