from transformers import AutoTokenizer, AutoConfig
import asyncio
import torch
from torch import nn
from pprint import pp

from copy import deepcopy
from typing import Union, Optional, Dict
from munch import Munch
import os,sys
import commune
from commune import Module

try:
    import bittensor
except RuntimeError:
    commune.new_event_loop()
    import bittensor
# import streamlit as st
# from commune.model.utils import encode_topk, decode_topk
from bittensor.utils.tokenizer_utils import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases


class ModelClient(Module, nn.Module):
    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6B',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b',
         }
    def __init__(self,
                model:Union[str, Dict] = 'model.dendrite',
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

    def set_model(self, model:Union[str, Dict]):
        if isinstance(model, str):
            self.model = commune.connect(model)
        elif isinstance(model, dict):
            self.model = commune.connect(**model)
        else:
            self.model = model
        
        print(self.model)
            
        self.model_name = self.model.getattr('model_name')
        # print(self.model.getattr('model_config'))
        self.config = Munch(self.model.getattr('model_config'))
        self.config.hidden_size = self.config.get('hidden_size')
        
        
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None, 
                topk:int=None, 
                verbose:bool = True,
                input_length: int = None,
                output_topk : bool = False ,
                output_logits: bool = True,
                server_output_logits: bool = False,
                output_hidden_states:bool = False, 
                output_length:int = None,
                **kwargs):
        '''
        
        Forward pass for the model for the client
        Args:
            input_ids: torch.Tensor of shape (batch_size, input_length)
            attention_mask: torch.Tensor of shape (batch_size, input_length)
            top k: int, number of topk logits to return
            verbose: bool, print out the logits
            input_length: int, length of the input
            output_logits: bool, return logits
            server_output_logits: bool, return logits from the server
        '''

        topk = topk if topk else self.topk
        output_length = output_length if output_length else self.output_length
        input_length = input_length if input_length else self.input_length
        input_ids = input_ids[:,-input_length:]
        attention_mask =  attention_mask[:,-input_length:] if attention_mask != None else attention_mask
        
        
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': False,
            'token_remap': True,
            'logit_remap': False,
            'topk': topk,
            'output_length': output_length,
            'output_logits': server_output_logits,
            'verbose': verbose,
            **kwargs
        }
        # import ipdb; ipdb.set_trace()
        response_dict = self.model.forward(**model_kwargs)
        # if topk:
        output_dict = {}
        
        
        if output_logits:
            print(response_dict)
            output_dict['logits'] = self.decode_topk(response_dict['topk'], vocab_size=self.vocab_size)
        if output_topk:
            output_dict['topk'] = response_dict['topk']
            
        
        return Munch(output_dict)

    __call__ = forward

    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        
        
        if isinstance(tokenizer, str):
            if tokenizer == 'bittensor':
                tokenizer = bittensor.tokenizer()
            else:
                tokenizer = self.shortcuts.get(tokenizer, tokenizer)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except ValueError:
                    print('resorting ot use_fast = False')
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

        # print(tokenizer)
        self.tokenizer = tokenizer     
        self.vocab_size = self.tokenizer.vocab_size
        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        self.std_tokenizer = bittensor.tokenizer()
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        self.to_translation_map = get_translation_map(self.tokenizer, self.std_tokenizer)
        self.from_translation_map = get_translation_map(self.std_tokenizer, self.tokenizer)
        
        self.config['pad_token_id'] = self.tokenizer.pad_token_id

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
        dataset = commune.connect('dataset.bittensor')

        

        from commune.utils.time import Timer
        import time
        
        
        for i in range(num_batches):
            t = Timer()
            sample = dataset(fn='sample')
            targets = sample['input_ids'][:, -1:]
            sample['input_ids'] =  sample['input_ids'][:, :-1]
            t = commune.timer()
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

        from commune.utils.time import Timer
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
    def test_performance(cls,model = 'DendriteModel',
                         dataset = 'dataset::bittensor',
                         batch_size= 32, 
                         sequence_length=256,
                         num_batches=2):
        
        from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer

        from commune.utils.torch import  tensor_info_dict
        
        if model != None:
            self = cls(model=model)
        
        dataset = commune.connect(dataset)
        sample = dataset.sample()
        
        for i in range(num_batches):
            sample = dataset.sample(batch_size=32, sequence_length=256)
            targets = sample['input_ids'][:, -1:] 
            sample['input_ids'] = sample['input_ids'][:, :-1] 
            sample['output_logits'] = True
            sample['topk'] = 4096
            # sample['autocast'] = True
            t = commune.timer()
            pred = self(**sample)        
            loss_tuple = phrase_cross_entropy(topk_tensor=pred['topk'][:,0,:,:], target_phrases=targets)
            commune.print(f'Loss : {loss_tuple[0].item()} Time: {t.seconds}', 'cyan')
            

    @classmethod
    def default_model(cls):
        # model = commune.connect(ip='65.49.81.154', port=50050, virtual=False)
        model = commune.connect('TransformerModel::EleutherAI_gpt-neo-125M', virtual=False)
        self = cls(model=model)
        return self
    @classmethod
    def test_neurons(cls, models=['model::gpt2.7b', 'model::gptjt', 'model::opt13b'], *args,**kwargs):
        for model in models:
            cls.print(f'Testing {model}', 'purple')
            cls.test_neuron(model=model, tokenizer='bittensor', *args,**kwargs)
    @classmethod
    def test_neuron(cls, model='model::gpt2.7b', tokenizer='bittensor', num_batches=2, dataset='dataset::bittensor', batch_size=32, sequence_length=12, topk=4096, **model_kwargs):
        from commune.block.bittensor.neuron.miner import neuron
        from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer
        self = cls(model = model, tokenizer=tokenizer)
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
        
        dataset = commune.connect(dataset)
        sample = dataset.sample()
        
        for i in range(num_batches):
            sample = dataset.sample(batch_size=32, sequence_length=256)
            target = sample['input_ids'][:, -1:] 
            inputs_x = sample['input_ids'][:, :-1] 
            t = commune.timer()
            message, _model_output, topk_tensor = nucleus.encode_forward_causallmnext(inputs_x, topk=topk)
            loss_tuple = phrase_cross_entropy(topk_tensor=topk_tensor, target_phrases=target)
            commune.print(f'Loss : {loss_tuple[0].item()} Time: {t.seconds}', 'cyan')
 
    @classmethod
    def run_neuron(cls, model={'ip': '65.49.81.154', 'port': 50050}, tokenizer='gptj'):
        import bittensor
        from commune.block.bittensor.neuron.miner import neuron
        self = cls(model=model, tokenizer=tokenizer)
        n = neuron(model=self)  
        n.run()
 

 
if __name__ == "__main__":
    
    # ModelClient.default_model()
    
    model_kwargs = dict(model={'ip': '65.49.81.154', 'port': 50050}, tokenizer='gptj')
    # ModelClient.test_neuron('model::gpt20b', tokenizer='gpt20b')
    ModelClient.run_neuron(**model_kwargs)
    
    # ModelClient.r()