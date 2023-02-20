import os, sys
from pprint import pp

from functools import partial
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
import torch
from commune.utils.torch import tensor_dict_info
from commune.utils.tokenizer import decode_topk
import streamlit as st
# logger = logger.opt(colors=True)
import commune
commune.new_event_loop()
if os.getenv('USE_STREAMLIT') == 'true':
    import streamlit as st
import os
    
# import torch
import commune
# commune.utils
from torch import nn
    
"""
Examples 



"""



class LayerBlock(torch.nn.Module):
    def __init__(self, in_dim:int=10, out_dim:int=10, norm_fn:Callable = None, act_fn:Callable = None):
        super(LayerBlock, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = torch.nn.Parameter(torch.randn(self.in_dim, self.out_dim))
        self.b = torch.nn.Parameter(torch.randn(self.out_dim))
        
        self.norm_fn = torch.nn.LayerNorm(self.out_dim) if norm_fn == None else norm_fn
        self.act_fn = torch.nn.GELU() if act_fn == None else act_fn
        
        # initialize the parameters
    def init_weights(self):
        in_d = self.W.shape[0]
        y = 1.0/np.sqrt(in_d)
        self.W.data.uniform_(-y, y)
        self.b.data.fill_(0)

    def forward(self, x:torch.Tensor, choice = 'left'):
        
        x = x.to(self.W.device)
        
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        emb = torch.einsum('ij,bi -> bj', [self.W, x]) + self.b
        
        emb = self.norm_fn(emb)
        
        emb = emb.reshape(*original_shape[:-1], emb.shape[-1])
        
        return emb
    
    @classmethod
    def test(cls, in_dim=10, out_dim=100, batch_dim=10):
        linear = Layer(in_dim=in_dim, out_dim=out_dim)
        x = torch.randn([batch_dim, in_dim])
        linear.to('cuda')
        target = torch.randn([batch_dim, out_dim])
        target = target.to('cuda')
        
        
        optimizer = torch.optim.Adam(linear.parameters(), lr=0.1)
        
        for i in range(1000):
            optimizer.zero_grad()
            pred = linear(x=x)

            loss = (pred - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            print(loss)
    


class AutoEncoder(torch.nn.Module, commune.Module):
    def __init__(self, 
                 in_dim = 10,
                 hidden_dim:int=64,
                 num_layers:int=1,
                 device: str = 'cpu',
                 out_dim: Optional[int] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        torch.nn.Module.__init__(self)
        self.build(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, device=device)
        self.set_optimizer(**(optimizer if optimizer != None else {}))
        self.set_device(device)
        
    @property
    def device(self) -> str:
        return self._device
    
    def set_device(self, device:str) -> str:
        self.to(device)
        self._device = device
        return device
    @property
    def device (self) -> str:
        return self._device

    def build(self, in_dim:int,
              hidden_dim:int, 
              out_dim:int, 
              device='cpu',
              num_layers:int=1):
        if out_dim == None:
            out_dim = in_dim
        
        # build the encoder
        encoder_blocks = [LayerBlock(in_dim, hidden_dim)]
        for i in range(num_layers):
            encoder_blocks.append(LayerBlock(hidden_dim, hidden_dim))
        self.encoder = torch.nn.Sequential(*encoder_blocks)
        
        # build the decoder
        
        decoder_blocks = []
        for i in range(num_layers):
            decoder_blocks.append(LayerBlock(hidden_dim, hidden_dim))
        
        decoder_blocks += [LayerBlock(hidden_dim, out_dim)]
        self.decoder = torch.nn.Sequential(*decoder_blocks)
    def forward(self, x):
        emb = self.encoder(x.to(self.device))
        emb = self.decoder(emb)
        return emb

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def loss(self, x):
        return torch.nn.functional.mse_loss(self.forward(x), x.to(self.device))

    def learn_step(self, x) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f'AutoEncoder()'

    def __str__(self):
        return f'AutoEncoder()'
    

    
    def set_optimizer(self, **params) -> 'optimizer':
        self.optimizer = self.get_optimizer(**params)
        return self.optimizer
    
    def get_optimizer(self, optimizer=None, **params) -> 'optimizer':
        if optimizer == None:
            optimizer =  torch.optim.Adam
        elif isinstance(optimizer, str):
            optimizer = commune.import_object(optimizer_class)
        elif isinstance(optimizer, type):
            return optimizer_class
        
        
        params = params.pop('params', {'lr': 0.1})
        optimizer = optimizer(self.parameters(), **params)
        
        return optimizer
    
    @classmethod
    def train(cls,
             in_dim:int=512, 
             out_dim:int=512 ,
             hidden_dim:int=256, 
             num_layers:int = 1,
             batch_dim:int=32,
             device:str='cuda',
             num_batches:int=3000,
             steps_per_batch:int = 1):
        self = cls(in_dim=in_dim, 
                   out_dim=out_dim,
                   device=device, 
                   num_layers=num_layers,
                   optimizer = {'lr': 0.0001},)
                

        for i in range(num_batches):
            
            x = torch.randn([batch_dim, in_dim])
            for j in range(steps_per_batch):
                print(self.learn_step(x))

if __name__ == "__main__":
    AutoEncoder.run()



class AdapterModel( nn.Module, commune.Module):

    def __init__(self,
                # models: List[str] = ['model::gpt125m', 'model::gpt2.7b', 'model::gptj', 'model::opt13b'],
                models: List[str] = ['model::gpt125m'],

                tokenizer: 'tokenizer' = 'bittensor',
                optimizer:  'torch.optimizer' = None,
                metrics: Dict= None,
                load: bool = True,
                tag= None,
                device = None,
                ):
        nn.Module.__init__(self)
        self.layer = commune.import_object('commune.model.layer.LayerBlock')()
        self.tag = tag
        
        self.model_device = 'cpu'
        
        self.model_name = 'ensemble'
        
        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)

        # set model and tokenizer
        self.set_model(models=models)

        # set tokenizer to model name (HF only) if tokenizer == None
        
        self.set_optimizer(optimizer=optimizer)
        
        self.set_metrics(metrics=metrics)
        
        self.set_stats()
        
        if load:
            self.load()
        
        
        
    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None, *args, **kwargs):
        
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.optimizer = optimizer
        return self.optimizer
    @classmethod
    def test(cls, topk=4096, output_length=20):
        
        model = cls()
        sample = commune.connect('dataset::bittensor').sample()

        sample.update(dict(
            output_hidden_states=True,
            hidden_dim_bounds = [0, 100],
            output_logits=False, 
            output_topk=True, 
            output_length=output_length,
            token_remap = False , 
            logit_remap = False,
            topk=topk
        ))
        
        targets = sample['input_ids'][:,1:]
        sample['input_ids'] = sample['input_ids'][:,:-1]
        pred = model.forward(**sample, no_grad=True)
        
        # pred['logits'] = decode_topk(pred['topk'])
        logits =  pred['logits']
        import streamlit as st
        gt = targets[:,-logits.shape[1]:].flatten()
        pred = logits.reshape(-1, logits.size(-1))
        loss = cls.calculate_loss(pred=pred, 
                                    gt=gt)              
        
        
        st.write(loss)
        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
     

    @classmethod
    def calculate_loss( cls, pred, gt):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt)
        return loss

    def set_metrics(self, metrics=None):
        self.metrics = {}
        if metrics == None:
            self.metrics['cross_entropy'] =  torch.nn.CrossEntropyLoss()
        return metrics
    
    async def async_model_forward(self, model, *args, **kwargs):
        return self.models[model].forward(*args, **kwargs)
        
        
        
    def aggregate(self, 
                  x: List[torch.Tensor], 
                  *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = torch.mean(x, dim=0)
        x = torch.softmax(x, dim=-1)
        x = torch.log(x + 1e-10)
        
        return x
    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def forward(self, *args, **kwargs):
        
        kwargs['output_hidden_states'] = True
        jobs = []
        for model in self.models:
            job = self.async_model_forward(model=model, *args, **kwargs)
            jobs.append(job)
            
        # return 
        
        peer_outputs =  asyncio.run(asyncio.gather(*jobs))
        
        max_token_index = max([t['topk'][:,:, kwargs['topk']:].max().item() for t in peer_outputs])
        
        model_names = self.model_names
        for model_i, peer_output in enumerate(peer_outputs):
            if 'hidden_states' in peer_output:
                model_name = model_names[model_i]
                st.write(peer_output['hidden_states'].shape)
                adapter_emb = self.model_adapter[model_name](peer_output['hidden_states'])
                st.write(adapter_emb.shape) 
            if 'topk'  in peer_output:
                peer_output['logits'] = decode_topk(peer_output['topk'], vocab_size=int(max_token_index+1))
                peer_outputs[model_i] = peer_output
            else:
                peer_outputs[model_i] = peer_output
            
        # stack with different logits dimensions and aggregate
        output_dict = dict(
            peer_logits = torch.stack([x['logits'] for x in peer_outputs], dim=0),
            peer_hidden_states = torch.stack([x['hidden_states'] for x in peer_outputs], dim=0),

            peer_losses = []
        )
        
        for model_i, peer_output in enumerate(peer_outputs):
            pred =  peer_output['logits'][:,:-1]
            gt = kwargs['input_ids'][:,1:][:, -pred.shape[1]:].flatten()
            pred = pred.reshape(-1, pred.size(-1))
            peer_loss = self.calculate_loss(pred=pred, gt=gt)
            output_dict['peer_losses'].append(peer_loss)
        
        output_dict['logits'] = self.aggregate(output_dict['peer_logits'])
        st.write(tensor_dict_info(output_dict))
        
        return output_dict
    

    @property
    def device(self):
        # deepspeed has .module.device to access device
        return self.model_device

    def set_model(self, models:List[str]):
        
        self.model_adapter = nn.ModuleDict()
        
        model_adapter_class = AutoEncoder
        self.models = {}
        for model in models:
            self.models[model] = commune.connect(model)
            hidden_size = 100
            self.model_adapter[model] = model_adapter_class(in_dim=hidden_size, out_dim=self.vocab_size)
        return self.models
    

    def list_models(self):
        return list(self.models.keys())

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        from transformers import AutoTokenizer
        if isinstance(tokenizer, str):
            if tokenizer == 'bittensor':
                import bittensor
                tokenizer = bittensor.tokenizer()
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except ValueError:
                    print('resorting ot use_fast = False')
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.tokenizer = tokenizer
        
        

        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

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

    def tokenize(self, text: str = 'Whadup', input_ids_only:bool = True, device: str=None) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        device = device if device != None else self.device
        tokenizer_output = self.tokenizer(text, return_tensors='pt')
        if input_ids_only:
            return tokenizer_output.input_ids.to(self.device)
        return self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)

    
    def learn_step(self, **sample ):
        targets = sample['input_ids'][:,1:].to(self.device)
        sample['input_ids'] = sample['input_ids'][:,:-1].to(self.device)
        self.optimizer.zero_grad()
        
        
        with torch.autocast(device_type='cuda'):
            pred = self.forward(**sample, no_grad=False)
            logits =  pred['logits']
            targets = targets[:,-logits.shape[1]:]
            pred = logits.reshape(-1, logits.size(-1))
            loss = self.calculate_loss(pred=logits.reshape(-1, logits.size(-1)), 
                                        gt=targets.flatten().to(self.device))              
        

        loss.backward()
        self.optimizer.step()
    
        
        return loss.item()

    def set_stats(self, stats:dict=None): 
        if stats == None:
            stats =  dict(
                steps = 0,
                loss = 0,
            )
        self.stats = Munch(stats)
        

    @property
    def module_tag(self): 
        return self.resolve_module_tag()
    
    def resolve_module_tag(self, tag=None):
        tag = tag if tag else self.tag
        module_tag = self.model_name.replace("/", "_")
        if tag:
            module_tag +=  f'_{tag}'
        return module_tag
    
    def save(self, tag:str = None, trainable_only:bool = True):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        model_state_dict = self.models.state_dict()
        
        if trainable_only:
            model_state_dict = {k:v for k,v in model_state_dict.items() if v.requires_grad} 
    
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'stats': dict(self.stats)
        }
    
        torch.save(state_dict, path)
        
        return path
    
    def load(self, tag=None):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        if not os.path.exists(path):
            logger.warning(f'No saved model found at {path}')
            return
        loaded_state  = torch.load( path)
        state_dict = self.models.state_dict()
        for k,v in loaded_state['model'].items():
            assert k in state_dict
            state_dict[k] = v
        self.models.load_state_dict(state_dict)
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.set_stats(loaded_state['stats'])
        

    @classmethod
    def train(cls, 
                    tag:str = 'demo', 
                    num_batches:int = 200,
                    num_epochs:int = 200, 
                    dataset:str= 'BittensorDataset', **kwargs):
        model = cls(tag=tag, load=True,  **kwargs)
        dataset = cls.connect(dataset)
        
        best_loss = 10e10
        for epoch in range(num_epochs):
            total_epoch_loss = 0
            epoch_loss = 0
            if epoch > 0:
                model.load(tag=tag)
            for i in range(num_batches):
                sample = dataset.sample()
                loss = model.learn_step(**sample)
                try:
                    total_epoch_loss += loss
                except:
                    continue
                epoch_loss = total_epoch_loss/(i+1)
                info_str = f'Batch {i}/{num_batches} Epoch {epoch}/{num_epochs} CE: {loss} Epoch Loss: {epoch_loss} Best Loss: {best_loss}'
                logger.success(info_str)
                print('BROOO')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                try:
                    model.save(tag=tag)
                except TypeError:
                    continue


    # @classmethod
    # def test(cls, output_length=10, topk=4096, **kwargs):
    #     import streamlit as st
    #     # Load dataset
    #     dataset = commune.connect('dataset::bittensor')
        
    #     # Load model
    #     model = AdapterModel()
        
    #     # gets ample
    #     sample = dataset.sample()
        
    #     sample['input_ids'] = sample['input_ids'][:,:-1]
    #     targets = sample['input_ids'][:,1:]
    #     # Run model
    #     t = commune.timer()
        
        
    #     sample.update(dict(
    #         output_hidden_states=False,
    #         output_logits=False, 
    #         output_topk=True, 
    #         output_length=output_length,
    #         token_remap = False , 
    #         logit_remap = False,
    #         topk=topk
    #     ))
        
        
    #     model = commune.connect('model::gpt125m')
    #     pred = model.forward(**sample )
    #     pred['logits'] = decode_topk(pred['topk'], topk=topk)
        
    #     # what is the pred of the model? A   
                
    #     loss = cls.calculate_loss(pred=pred['logits'].reshape(-1, pred['logits'].size(-1)), gt=targets[:, -output_length: ].flatten())
    #     st.write(loss)
    #     peer_loss = []
    #     st.write(pred['peer_logits'].shape)
    #     for i, peer_logits in enumerate(pred['peer_logits']):
    #         st.write(peer_logits.shape)
    #         peer_loss += [model.calculate_loss(pred=peer_logits.reshape(-1, peer_logits.size(-1)), gt=targets.flatten())]
        
    #     metrics = {
    #         'info': tensor_dict_info(pred),
    #         'seconds': t.seconds,
    #         'loss': loss.item(),
    #         'peer_loss': peer_loss
    #     }
    #     st.write(metrics)
        
        
        
    def loss(self, logits: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:

        if not hasattr(self, 'loss_fct'):
            self.loss_fn = torch.nn.CrossEntropyLoss()
            
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return loss

if __name__ == "__main__":
    
    AdapterModel.test()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


