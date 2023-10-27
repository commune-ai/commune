# import nest_asyncio
# nest_asyncio.apply()
import commune as c
c.new_event_loop()
import bittensor
import streamlit as st
import torch
from typing import Dict, List, Union, Any
import random
from copy import deepcopy
import pandas as pd
import asyncio
from munch import Munch
from bittensor.utils.tokenizer_utils import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases

from torch import nn




class Validator(c.Model):

    def __init__(self, 
                 **kwargs
                 ):
        self.init_model()
        config = self.set_config(kwargs=kwargs)


        if config.refresh:
            self.refresh(self.tag)
            config.load = False
        if config.load:
            self.load(self.tag)

        self.set_models(models=config.models, network=config.network, update=config.update)
        self.set_dataset(config.dataset)
        self.set_tokenizer(config.tokenizer)
        self.weights = nn.Parameter(torch.ones(5))

        self.set_optimizer(config.optimizer)

    namespace_update_ts =0
    _namespace = None
    

    def set_dataset(self, dataset):
        self.dataset = c.module('dataset')

        
    def verify_signature(self, signature: Dict) -> bool:
        return True
    
            
    @classmethod
    def get_models(cls, models=None) -> List[str]:
        modules = c.servers()
        
        if models is None:
            models = [m for m in modules if m.startswith('model')]
        elif isinstance(models, str):
            models = [m for m in modules if m.startswith(models)]
        elif isinstance(models, list):
            models = [m for m in modules if m in models]
        
        return models
            
        
        return self.modules()
    
    @classmethod
    def resolve_shortcut(cls, model: str) -> str:
        model = cls.module('model.hf').shortcuts.get(model, model)
        return model

    def set_tokenizer(self, tokenizer):
        
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer
        
        tokenizer = self.resolve_shortcut(tokenizer)

        if tokenizer is None:
            tokenizer = self.model_path
            
        assert isinstance(tokenizer, str), f'tokenizer must be a string. tokenizer: {tokenizer}'

        self.config['tokenizer'] = tokenizer

        
        self.print(f'setting {tokenizer} tokenizer...')
        
        try:
            # HACK TO INCLUDE LLAMA TOKENIZER
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except ValueError:
            
            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.tokenizer = tokenizer

        self.tokenizer = prep_tokenizer(self.tokenizer)
        self.config['pad_token_id'] = self.tokenizer.pad_token_id
        self.config['vocab_size'] = self.tokenizer.vocab_size
        return self.tokenizer

    @classmethod
    def get_dataset(cls, dataset: str) -> None:
        sample = None
        datasets = c.servers(dataset)
        for dataset in datasets:
            if isinstance(dataset, str):
                dataset = c.connect(dataset)
                if callable(dataset.sample):
                    sample = dataset.sample()
                    if cls.check_input(sample):
                        break
            else:
                raise ValueError(f'Invalid dataset type: {type(dataset)}')
        
        if isinstance(dataset, str): 
            raise ValueError(f'Dataset not found {datasets}')
        return dataset
    
    def calculate_metric(self, x):
        if not hasattr(self, 'metric'):
            self.metric = torch.nn.CrossEntropyLoss()
            
        input_ids = x.get('input_ids', None).clone()
        pred = x.get('logits', None).clone()
        if input_ids != None:
            gt = input_ids[:, -(pred.shape[1]-1):].flatten()
            pred = pred[:, :-1]
            
        assert isinstance(gt, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
        assert isinstance(pred, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        metric =  self.metric(pred, gt.to(pred.device))
        
        
        return metric.item()
    
    

    def sample(self, **kwargs):
        kwargs.update(dict(
            # tokenize=True, 
            sequence_length=self.config.sequence_length,
            batch_size=self.config.batch_size
        ))
        
        sample = self.dataset.sample(**kwargs)
        return sample
   
   
    @property
    def model_keys(self):
        return list(self.models.keys())
        

    def get_sample_metatdata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        
        sample_metadata = {}
        for k, v in sample.items():
            metadata_k = {'type': type(v)}
            
            if isinstance(v, torch.Tensor):
                metadata_k.update({
                    'shape': str(v.shape),
                    'dtype': str(v.dtype),
                })
            elif type(v) in [list, set, tuple]:
                metadata_k.update({
                    'length': len(v),
                })
            elif isinstance(v, dict):
                metadata_k.update({
                    'length': len(v),
                })
            sample_metadata[k] = metadata_k

        return sample_metadata

    def resolve_model(self, model: str = None) -> Any:
        if model is None:
            model = self.random_model()
        elif isinstance(model, str):
            model = self.models[model]
        return model
    
    @classmethod
    def check_input(cls, x):
        if isinstance(x,dict):
            if 'input_ids' in x and isinstance(x['input_ids'], torch.Tensor):
                return True
        return False

    def check_output(self, x):
        if isinstance(x,dict):
            if 'topk' in x:
                return True  
        return False  
    async def async_forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                model:str=None, 
                map_tokens=False,
                train: bool = False,
                verbose:bool= False,
                output_length: bool = 16,
                topk: int = 4096,
                return_keys: List[str] = ['topk', 'stats'],
                **kwargs ):
        
        sample = self.locals2kwargs(locals())
        timer = c.timer()

        model = await self.async_connect(model, 
                                         namespace=self.namespace, 
                                         virtual=False)

        sample = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            topk= topk,
            output_length=output_length,
            return_keys=return_keys,
            train= train
            )
        
        assert self.check_input(sample)
        output = await model(fn='forward',
                             kwargs=sample, 
                             return_future=True)
        
        # check the outputs of the model
        success = self.check_output(output)
        
        if success:
            output['logits'] = self.decode_topk(output['topk'], topk=topk, vocab_size=self.config.vocab_size)
            metric = self.calculate_metric(dict(input_ids=input_ids, **output))
        else:
            output = {'error': output}
            metric = self.config.default_metric
            if verbose:
                self.print(f'forward failed: {output}', model)


        output['stats'] =  {
            'inference_time': timer.seconds,
            'metric': metric,
            'timestamp': self.time(),
            'success': success
            
        }
                
        return output
            
        
    selected_models = None
    loop = None
    
    def set_models(self, 
                   models: Union[str, List[str]] = None ,
                   network:str = None,
                   update:bool=False, ) -> List[str]:

        models = models if models != None else self.config.models
        network = network if network != None else self.config.network 
        
        self.namespace = c.namespace(network=network, update=update)
            
        if isinstance(models, list):
            for m in models:
                assert isinstance(m, str)
                assert m in self.namespace, f'{m} does not exist in namespce'
        elif isinstance(models, str):    
            models = [m for m in self.namespace.keys() if m.startswith(models)]
            
        self.models = models
        return models

    def calculate_weights(self, w):
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w)
        
        if len(w) >1:
            w = -(w - w.mean())/ (w.std()+1e-10)
            w = torch.softmax(w, dim=-1)
        else:
            w = torch.ones_like(w)
        return w
        
    def process_outputs_mixture(self, ensemble_output):
        
        w = self.calculate_weights(ensemble_output['weights'])
        ensemble_output['ranks'] =  ranks = torch.argsort(w, dim=-1, descending=True).cpu().numpy().tolist()


        logits  = torch.stack(ensemble_output['logits'])
        probs = torch.softmax(logits, dim=-1)
        
        probs = probs * w[:,None,  None, None]
        probs_unormalized = probs.sum(0)
        probs = probs_unormalized / probs_unormalized.sum(-1, keepdim=True)
        

        # convert the renormalized weights back to logits
        logits = torch.log(probs + 1e-10) 
        ensemble_output['logits'] = logits
        
        # TODO: add ensemble_output metrics
        ensemble_output['weights']  = w.tolist()
        
        # TODO: add ensemble_output metrics (defaulting to the first model)
        ensemble_output['hidden_state'] = torch.randn(logits.shape[0], logits.shape[1], self.config.hidden_size)

        return ensemble_output
        
    
    def process_outputs_best(self, ensemble_output):

        w = self.calculate_weights(ensemble_output['weights'])
        ensemble_output['ranks'] =  ranks = torch.argsort(w, dim=-1, descending=True).cpu().numpy().tolist()
        best_model_idx = ensemble_output['ranks'][0]
        ensemble_output['logits'] = logits = ensemble_output['logits'][best_model_idx]
        # convert the renormalized weights back to logits
        
        # TODO: add ensemble_output metrics
        ensemble_output['weights']  = w.tolist()
        
        # TODO: add ensemble_output metrics (defaulting to the first model)
        ensemble_output['hidden_state'] = torch.randn(logits.shape[0], logits.shape[1], self.config.hidden_size)

        return ensemble_output
        
    def process_outputs_random(self, ensemble_output):
        w = self.calculate_weights(ensemble_output['weights'])
        random_model_index = random.randint(0, len(w)-1)
        ensemble_output['logits'] = logits = ensemble_output['logits'][best_model_idx]
        ensemble_output['weights']  = w.tolist()
        ensemble_output['hidden_state'] = torch.randn(logits.shape[0], logits.shape[1], self.config.hidden_size)
        return ensemble_output
        
    def process_outputs(self, ensemble_output, ensemble_mode='best'):
        if  ensemble_mode == None:
            ensemble_mode = self.config.ensemble_mode
        return getattr(self, f'process_outputs_{ensemble_mode}')(ensemble_output)
        
        
    def get_models(self, max_models:int) -> List[str]:


        
        # shuffle to avoid overloading the first model
        
        # TODO: add a way to shuffle models
        models = self.shuffle(self.models)
        # max call size is the number of models that can be called per forward pass
        called_models = self.copy(models[:self.config.max_models_per_call])
        
        return called_models
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                output_hidden_states: bool = False,
                models:str=None, 
                threshold: float = 4.0,
                timeout = 7,
                topk: int = None,
                sequence_length:int = None,
                max_models_per_call: int = None,
                batch_size :int = 32,
                train: bool = None,
                verbose: bool = False,
                retries: int = 4,
                tag = None,
                save = True,
                return_keys = ['logits', 'hidden_state', 'topk'],
                **kwargs ):
        
        
        tag = tag if tag != None else self.tag
        
        config = self.config
        timer = self.timer()
        max_models_per_call = max_models_per_call if max_models_per_call != None else config.max_models_per_call
        topk = topk if topk != None else config.topk
        train = train if train != None else config.train
        
        if self.config.new_loop_per_forward:
            loop = self.new_event_loop()

        else:
            loop = self.get_event_loop()

        called_models = self.get_models(max_models=max_models_per_call)
        
        sequence_length = sequence_length if sequence_length else self.config.sequence_length
        batch_size = batch_size if batch_size else self.config.batch_size
        input_ids = input_ids[:batch_size, -sequence_length:]
        if verbose:
            self.print(f'forwarding to {len(called_models)} models ')

        sample = dict(input_ids=input_ids, 
                      topk=topk, 
                      timeout=timeout,
                      train=train, 
                      **kwargs)
        
        jobs = [asyncio.wait_for(self.async_forward(**sample,model=m), timeout=timeout) for m in called_models]
        model_outputs = loop.run_until_complete(asyncio.gather(*jobs))
        
        if verbose:
            self.print('RECIEVING RESPONSE FROM ',len(model_outputs), 'MODEL OUTPUTS')
        
        model_output_dict = {}
        for model_output, model_key in zip(model_outputs, called_models):
            if model_output is None:
                continue
            model_output_dict[model_key] = model_output
            
        
        stats = {
                'models_available': len(self.models),
                'models_called':len(called_models),
                'models_failed': [],
                'inference_time': timer.seconds,
                'metric': None,
                'input_schema': self.get_sample_schema(sample),
                'best_metric': None,
                          }
        
        model_stats = {}
        
        ensemble_output = self.munch({
            'weights': [],
            'logits': [],
            'metrics': [],  
            'probs': [],
            'models': [],
            'models_failed': [],
            'ranks': [],

        })
        
        for m_key, m_output in model_output_dict.items():
            m_stats = m_output['stats']
            is_success  =  m_stats['success'] and m_stats['metric'] < self.config.threshold
            
            if  is_success:
                model_stats[m_key] = m_stats
                ensemble_output['logits']+= [m_output['logits']]
                ensemble_output['metrics'] += [m_stats['metric']]
                ensemble_output['weights'] += [m_stats['metric']]
                ensemble_output['models'] += [m_key]
            else: 
                ensemble_output['models_failed'] += [m_key]
                
        ensemble_outputs = self.process_outputs(ensemble_output =ensemble_output, ensemble_mode=self.config.ensemble_mode)
        best_model_idx = ensemble_output['ranks'][0]
        # calculate the stats
        stats['best_metric'] = ensemble_output['metrics'][best_model_idx]
        stats['best_model'] = ensemble_output['models'][best_model_idx]
        stats['called'] = len(called_models)
        stats['models_failed'] = ensemble_output['models_failed']
        stats['models'] = ensemble_output['models']
        stats['success'] = len(ensemble_output['models'])
        stats['fails'] = stats['called'] - stats['success']

        for i, (mkey, mstats) in enumerate(model_stats.items()):
            if mstats == None:
                continue
            model_stats[mkey]['weight'] = ensemble_output['weights'][i]
            
            
        best_model = ensemble_output['models'][best_model_idx]
        best_model_metric = ensemble_output['metrics'][best_model_idx]
        metric = self.calculate_metric({**ensemble_output, 'input_ids': input_ids})
    
        stats.update({
            # 'model_stats': model_stats,
            'model_stats': model_stats,
            'best_metric': best_model_metric,
            'metric': metric,
            'inference_time': timer.seconds
        })
        
        ensemble_output['stats'] = stats
        
        self.set_stats(stats)
    
        if save:
            self.save(tag=tag)
        
        if 'topk' in return_keys:
            ensemble_output['topk'] = self.encode_topk(ensemble_output['logits'], topk=topk)
            
        ensemble_output = {k:ensemble_output[k] for k in return_keys}
        
        return ensemble_outputs

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

    def save(self, tag = None, verbose:bool = True, keys=['config']) -> Dict[str, Any]:
        c.Model.save(self, tag=tag, verbose=verbose, keys=keys)

    def load(self, tag = None, verbose:bool = True, keys=['config']) -> Dict[str, Any]:
        c.Model.load(self, tag=tag, verbose=verbose, keys=keys)
        
    def refresh(self, tag = None, verbose:bool = True, keys=['config']) -> Dict[str, Any]:
        c.Model.refresh(self, tag=tag, verbose=verbose, keys=keys)

    def set_stats(self, stats: Dict[str, Any] = None) -> None:
        
        self.stats = self.config.get('stats', {})
        stats  = stats if stats != None else {}

        model_stats = self.stats.get('model_stats', {})
        new_model_stats = stats.get('model_stats', {})
        for m, new_mstats in new_model_stats.items():
            past_mstats = model_stats.get(m, {})
            for k, v in new_mstats.items():
                past_v = past_mstats.get(k, v)
                new_mstats[k] = past_v*(1-self.config.alpha) + v*self.config.alpha
            new_mstats['successes'] = past_mstats.get('successes', 0) + 1
            new_mstats['fails'] = past_mstats.get('fails', 0)
            model_stats[m] = new_mstats
            
        for m in stats.get('models_failed', []):
            past_mstats = model_stats.get(m, {})
            past_mstats['fails'] = past_mstats.get('fails', 0) + 1

            model_stats[m] = past_mstats

        stats['model_stats'] = model_stats
        assert isinstance(stats, dict)
        self.stats = stats
        self.config['stats'] = stats
        
    @property
    def tag(self):
        return self.config.get('tag', 'base')
    
    @tag.setter
    def tag(self,tag):
        self.config['tag'] = tag
    
    
        
    def get_state(self, tag=None):
        return cls().load()
     
    def random_model_key(self):
        random_model_key = random.choice(self.model_keys)
        return random_model_key

    def random_model(self):
        random_model_key = self.random_model_key()
        return self.models[random_model_key]
    
    def random_model_keys(self, num_models: int = 1):
        num_models = min(num_models, len(self.model_keys))
        random_model_keys = random.choices(self.model_keys, k=num_models)
        return [k for k in random_model_keys]    
    
    def random_models(self, num_models: int = 1):
        num_models = min(num_models, len(self.model_keys))
        random_model_keys = random.choices(self.model_keys, k=num_models)
        return [self.models[k] for k in random_model_keys]
    
    @staticmethod
    def sample_check(sample):
        return bool(isinstance(sample, dict) and 'input_ids' in sample)
    
    

    
    @classmethod
    def learn(cls, *args, **kwargs):
        
        if kwargs.pop('remote', False):
            return cls.remote_fn(fn='learn', args=args, kwargs=kwargs)
        sleep_interval = kwargs.pop('sleep_interval', 3)
        stagger_interval = kwargs.pop('stagger_interval', 0)
        num_batches = kwargs.pop('num_batches', 2)
        
        self = Validator(*args, **kwargs)
        

        for _ in range(num_batches):
            
            sample = self.sample()
            if not self.sample_check(sample):
                continue
            sample['train'] = True
            output = self.forward(**sample)
            # model_stats = output.stats.pop('model_stats', None)
            c.print(self.stats['model_stats'])
            self.sleep(sleep_interval)
            
            
    @classmethod
    def ensure_defaults(params:dict, defaults:dict) -> dict:
        for k, v in defaults.items():
            if k not in params:
                params[k] = v
                
        return params
    
    def stats_table(self):
        return self.get_stats_table(self.stats)
    
    @classmethod
    def get_stats_table(cls, stats: Dict[str, Any] = None) -> pd.DataFrame:
        if isinstance(stats, str):
            stats = cls.get_stats(tag=stats)
        rows = []
        
        for model, stats in stats.get('model_stats', {}).items():
            
            row = {'model': model, **stats}
            row = {k:round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k not in ['timestamp', 'success']}
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values(by='metric')
        return df
            
    @classmethod
    def test(cls,  
             *args,
             tag='test',
             batch_size=8, 
             num_batches=2, 
             remote=False,
             **kwargs):

        kwargs = cls.locals2kwargs(locals())
        if kwargs.pop('remote', False):
            return cls.remote_fn(fn='learn', args=args, kwargs=kwargs)
        sleep_interval = kwargs.pop('sleep_interval', 1)
        stagger_interval = kwargs.pop('stagger_interval', 0)
        num_batches = kwargs.pop('num_batches', 2)
        
        self = Validator(*args, **kwargs)
        
        self.print(self.stats_table())
        
        for _ in range(num_batches):
            sample = self.sample()
            if not self.sample_check(sample):
                continue
            output = self.forward(**sample)
            # model_stats = output.stats.pop('model_stats', None)
            
            
            c.print(self.stats_table())
            self.sleep(sleep_interval)
        
    @classmethod
    def test_validation_keys(cls):
        vals = [Validator() for _ in range(10)]
        st.write([v.key.address for v in vals])
        hash = vals[0].key.hash({'hey': 'whadup'})
        sig = vals[0].key.sign(hash)
        
        assert not vals[0].key.verify(hash, signature = sig, public_key = vals[1].key.public_key )
        assert vals[0].key.verify(hash, signature = sig, public_key = vals[0].key.public_key )
        

    
    @classmethod
    def decode_topk(cls,  forward_response_tensor: torch.Tensor, topk:int=4096, vocab_size:int=50257) -> torch.Tensor:
        """ Returns full logits by decoding topk-encoding input. """
        batch_size, sequence_len, _ = forward_response_tensor.shape
        encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
        topk_values = encoded_probs[..., :topk]  # topk probs: [batch_size, sequence_len, topk]
        topk_indices = encoded_probs[..., topk:].long()  # topk probs indices: [batch_size, sequence_len, topk]

        topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
        remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
        
        remainder_floor = remainder_pmass / (vocab_size - topk)  # divide remainder: [batch_size, sequence_len]

        logits = torch.ones((batch_size, sequence_len, vocab_size), dtype=topk_values.dtype).to(topk_values.device)
        logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]

        
        # clamp max indices to topk
        topk_indices = torch.clamp(topk_indices, 0, vocab_size-1)  # [batch_size, sequence_len]
        logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

        return logits  # [batch_size, sequence_len, vocab_size]

        
    # @classmethod
    # def streamlit(cls):
    #     self =  cls(models=None, dataset='dataset.text.bittensor', load=True)
    #     timer = self.timer()
    #     for i in range(100):
    #         sample = self.sample()
    #         self.validate(sample=sample, max_models = 5 ,topk=4096, models=None)
    #         self.print(self.stats)
    #         samples_per_second = i/timer.seconds
    #         cls.print(f'samples_per_second: {samples_per_second}')
    

        
    @classmethod
    def streamlit(cls):
        
        import streamlit as st
        c.new_event_loop(nest_asyncio=True)
        # c.nest_asyncio()
        self = cls(models=None, dataset='dataset.text.bittensor', load=True)
        
        df = self.stats_table
        # print a scatter plot of the data
        import plotly.express as px
        fig = px.bar(df, x="model", y="metric", color="model")
        
        # make it vertical
        fig.update_layout(
            xaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig)
      
    @classmethod
    def run_miner(cls, remote = True, **kwargs):  
        return cls.remote_fn(fn='miner',name='miner',  kwargs=kwargs)
    
    
    @classmethod
    def mine(cls, 
               wallet='collective.0',
               model:str= 'validator',
               network = 'finney',
               netuid=3,
               port = None,
               device = None,
               prometheus_port = None,
               debug = True,
               no_set_weights = True,
               remote:bool = False,
               tag=None,
               sleep_interval = 2,
               autocast = True,
               burned_register = False,
               max_fee = 2.0,
               ):
        if tag == None:
            tag = f'{wallet}::{network}::{netuid}'
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='mine',name=f'miner::{tag}',  kwargs=kwargs)
            
        config = bittensor.neurons.core_server.neuron.config()
        # model things
        config.neuron.no_set_weights = no_set_weights
        
        # network
        subtensor = bittensor.subtensor(network=network)
        bittensor.utils.version_checking()
    
        # wallet
        coldkey, hotkey = wallet.split('.')
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        
        if wallet.is_registered(subtensor=subtensor, netuid=netuid):
            cls.print(f'wallet {wallet} is already registered')
            neuron = c.module('bittensor').get_neuron(wallet=wallet, subtensor=subtensor, netuid=netuid)
            port = neuron.axon_info.port
            prometheus_port = neuron.prometheus_info.port
        else:
            cls.ensure_registration(wallet=wallet, 
                                    subtensor=subtensor, 
                                    netuid=netuid,
                                    max_fee=max_fee,
                                    burned_register=burned_register, 
                                    sleep_interval=sleep_interval)
                        

        # enseure ports are free
        # axon port
        port = port if port is not None else cls.free_port()
        config.axon.port = port
        while cls.port_used(config.axon.port):
            config.axon.port += 1    
            
        # ensure prometheus port
        config.prometheus.port =  config.axon.port - 1000 if prometheus_port is None else prometheus_port         
        while cls.port_used(config.prometheus.port):
            config.prometheus.port += 1
            
            
        # neuron things
        config.neuron.autocast = autocast  
        if device is None:
            device = cls.most_free_gpu()


        if cls.server_exists(model):
            module =  cls.module(model)
            neuron = module.neuron(wallet=wallet, subtensor=subtensor, config=config, netuid=netuid)
        else:
            model_size = cls.get_model_size(model)
            free_gpu_memory = cls.free_gpu_memory()
            if free_gpu_memory[device] < model_size:
                device = cls.most_free_gpu()
            assert free_gpu_memory[device] > model_size, f'Not enough memory on device {device} to load model {model} of size {model_size}'
            config.neuron.device = f'cuda:{device}'
            config.neuron.model = model
        
            neuron = bittensor.neurons.core_server.neuron(wallet=wallet,
                                                        subtensor=subtensor,
                                                        config=config,
                                                        netuid=netuid)
            
        neuron.run()
                                                

 
             
             
        


    @classmethod
    def neuron(cls, config, wallet, subtensor, netuid=3, **kwargs):
       
        # server_class = bittensor.neurons.core_server.server
        server_class = cls.import_object('commune.bittensor.neuron.core_server.server')
        model = cls(**kwargs)
        server = server_class(model=model, config=config, tokenizer=model.tokenizer)
        bittensor.utils.version_checking()
        
        neuron = bittensor.neurons.core_server.neuron(model=server, 
               wallet=wallet,
               subtensor=subtensor,
               config=config,
               netuid=netuid)
        
        return neuron




if __name__ == '__main__':
    Validator.run()

        
