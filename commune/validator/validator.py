# import nest_asyncio
# nest_asyncio.apply()
import commune
commune.new_event_loop()
import bittensor
import streamlit as st
import torch
from typing import Dict, List, Union, Any
import random
from copy import deepcopy
import asyncio
from munch import Munch
from bittensor.utils.tokenizer_utils import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases

from torch import nn




class Validator(commune.Model):

    def __init__(self, 
                 **kwargs
                 ):
        self.init_model()
        config = self.set_config(kwargs=kwargs)
        self.print(config)
        
        self.set_models(models=config.models, network=config.network, update=config.update)
        self.set_max_stats_history(config.max_stats_history)
        self.set_batch_size(config.batch_size)
        self.set_sequence_length(config.sequence_length)
        self.set_dataset(config.dataset)
        self.set_stats(config.stats)
        self.set_alpha(config.alpha)
        self.set_tokenizer(config.tokenizer)
        
    namespace_update_ts =0
    _namespace = None
    @property
    def namespace(self):
        if not hasattr(self, '_namespace'):
            self._namespace = commune.namespace(network=self.config.network,update=False )
        time_since_update = self.time() - self.namespace_update_ts
        if time_since_update > self.config.namespace_update_interval:
            self.namespace_update_ts = self.time()
            self._namespace = commune.namespace(network=self.config.network,update=False )
        
        return self._namespace


    
    def set_max_stats_history(self, max_stats_history: int) -> None:
        self.max_stats_history = max_stats_history
    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
    def set_sequence_length(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def set_alpha(self, alpha: float) -> None:
        # set alpha for exponential moving average
        self.alpha = alpha
        
    def verify_signature(self, signature: Dict) -> bool:
        return True
    
            
    @classmethod
    def get_models(cls, models=None) -> List[str]:
        modules = cls.modules()
        
        if models is None:
            models = [m for m in modules if m.startswith('model')]
        elif isinstance(models, str):
            models = [m for m in modules if m.startswith(models)]
        elif isinstance(models, list):
            models = [m for m in modules if m in models]
        
        return models
            
        
        return self.modules()

    def set_tokenizer(self, tokenizer):
        
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer

        if tokenizer is None:
            tokenizer = self.model_path
            
        assert isinstance(tokenizer, str)
        assert isinstance(tokenizer, str, )

        self.config['tokenizer'] = tokenizer

        
        self.print(f'setting {tokenizer} tokenizer...')
        
        try:
            # HACK TO INCLUDE LLAMA TOKENIZER
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except ValueError:
            
            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)


        print('tokenizer loaded')

        self.tokenizer = tokenizer

        self.tokenizer = prep_tokenizer(self.tokenizer)
        self.config['pad_token_id'] = self.tokenizer.pad_token_id
        self.config['vocab_size'] = self.tokenizer.vocab_size
        self.vocab_size = self.config.get('vocab_size', 50257)
        return self.tokenizer

    

    def set_dataset(self, dataset: str) -> None:
        if isinstance(dataset, str):
            dataset = commune.connect(dataset)
        else:
            raise ValueError(f'Invalid dataset type: {type(dataset)}')
        
        self.dataset = dataset
        


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
            sequence_length=self.sequence_length,
            batch_size=self.batch_size
        ))
        
        sample = self.dataset.sample(**kwargs)
        if 'input_ids' not in sample:
            raise Exception(sample)
        
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
            
    @property
    def default_metric(self) -> float:
        return 10.0

    def resolve_model(self, model: str = None) -> Any:
        if model is None:
            model = self.random_model()
        elif isinstance(model, str):
            model = self.models[model]
        return model
    
    
    def check_input(self, x):
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
                output_length: bool = 2,
                topk: int = 4096,
                return_keys: List[str] = ['topk'],
                **kwargs ):
        
        
        
        sample = self.locals2kwargs(locals())
        timer = commune.timer()
        output = None
        # try:
        namespace = self.copy(self.namespace)
        model_name = self.copy(model)
        model = await self.async_connect(model_name, namespace=namespace, virtual=False)

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
        
        success = self.check_output(output)
        stats = {}
        if success:
            output['logits'] = self.decode_topk(output['topk'], topk=topk, vocab_size=self.vocab_size)
            metric = self.calculate_metric(dict(input_ids=input_ids, **output))
        else:
            output = {'error': output}
            metric = self.default_metric
            if verbose:
                self.print(f'forward failed: {output}', model_name)


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
                   models: Union[str, List[str]] = 'model' ,
                   network:str = None,
                   update:bool=False, ) -> List[str]:

        network = network if network != None else self.config.network 
            
        if isinstance(models, list):
            for m in models:
                assert isinstance(m, str)
                assert m in self.namespace, f'{m} does not exist in namespce'
        elif isinstance(models, str):    
            models = [m for m in self.namespace.keys() if m.startswith(models)]
            
        self.available_models = models
        return models

    

    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                output_hidden_states: bool = False,
                models:str=None, 
                threshold: float = 4.0,
                timeout = 7,
                topk: int = None,
                sequence_length:int = None,
                ratio: int = None,
                batch_size :int = None,
                train: bool = None,
                verbose: bool = True,
                retries: int = 4,
                save = True,
                **kwargs ):
        
        
        
        
        config = self.config
        timer = self.timer()
        ratio = ratio if ratio != None else config.ratio
        topk = topk if topk != None else config.topk
        train = train if train != None else config.train
        
        if self.config.new_loop_per_forward:
            loop = self.new_event_loop()

        else:
            loop = self.get_event_loop()


        available_models = self.available_models
        # shuffle to avoid overloading the first model
        available_models = self.available_models
        called_models = self.random_ratio_selection(self.copy(self.available_models), ratio=ratio)
        
        sequence_length = sequence_length if sequence_length else self.config.sequence_length
        batch_size = batch_size if batch_size else self.config.batch_size
        input_ids = input_ids[:batch_size, -sequence_length:]
        if verbose:
            self.print(f'forwarding to {len(called_models)} models ')

        sample = dict(input_ids=input_ids, 
                      topk=topk, 
                      timeout=timeout,
                      train=train, 
                      return_keys=['topk'],
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
            
        ensemble_logits = []
        
        stats = {
                'timestamp': self.time(), 
                'models_available': len(available_models),
                'models_called':len(called_models),
                
                'models_failed': [],
                
                'inference_time': timer.seconds,
                'metric': None,
                'success' : 0,
                'input_schema': self.get_sample_schema(sample),
                'best_metric': None,
                          }
        
        model_stats = {}
        weights = []
        ensemble = self.munch({
            'weights': [],
            'logits': [],
            'metrics': [],  
            'probs': [],
            'models': [],
            'models_failed': []

        })
        stats['success'] = 0
        for m_key, m_output in model_output_dict.items():
            m_stats = m_output['stats']

            if m_stats['success'] and m_stats['metric'] < self.config.threshold:
                model_stats[m_key] = m_stats
                ensemble['logits']+= [m_output['logits']]
                ensemble['metrics'] += [m_stats['metric']]
                ensemble['weights'] += [ensemble['metrics'][-1]]
                ensemble['models'] += [m_key]
            else: 
                ensemble['models_failed'] += [m_key]
                
        # calculate the stats

        stats['called'] = len(called_models)
        stats['models_failed'] = ensemble['models_failed']
        stats['models'] = ensemble['models']
        stats['success'] = len(ensemble['models'])
        stats['fails'] = stats['called'] - stats['success']


        w = torch.tensor(ensemble['weights'])
        if len(w) >1:
            w = -(w - w.mean())/ (w.std()+1e-10)
            w = torch.softmax(w, dim=-1)
        else:
            w = torch.ones_like(w)


        logits  = torch.stack(ensemble['logits'])
        
        
        probs = torch.softmax(logits, dim=-1)
        
        probs = probs * w[:,None,  None, None]
        probs_unormalized = probs.sum(0)
        probs = probs_unormalized / probs_unormalized.sum(-1, keepdim=True)
        
        # convert the renormalized weights back to logits
        logits = torch.log(probs + 1e-10) 
        
        ensemble['logits'] = logits
        # TODO: add ensemble metrics
        ensemble['weights']  = w.tolist()
        
        
        rank = torch.argsort(w, dim=-1, descending=True).cpu().numpy().tolist()
        best_model_idx = rank[0]
        
        ensemble['rank'] = rank
        ensemble['hidden_state'] = torch.randn(logits.shape[0], logits.shape[1], self.config.hidden_size)
        
        for i, (mkey, mstats) in enumerate(model_stats.items()):
            model_stats[mkey]['rank'] = ensemble['rank'][i]
            model_stats[mkey]['weights'] = ensemble['weights'][i]
            
            
        best_model = ensemble['models'][best_model_idx]
        best_model_metric = ensemble['metrics'][best_model_idx]
        metric = self.calculate_metric({**ensemble, 'input_ids': input_ids})
    
        stats.update({
            # 'model_stats': model_stats,
            # 'model_stats': model_stats,
            'best_metric': best_model_metric,
            'metric': metric,
            'inference_time': timer.seconds
        })
        
        ensemble['stats'] = stats
        
        
          
        self.set_stats(stats)
        
        print(self.config)
        if save:
            self.save()
        
        if verbose:
            self.print(stats)
        return Munch(ensemble)
    
    def validate(self, sample=None, 
                 models: str = None,
                 topk:int=512, 
                 num_batches: int = 1,
                 num_models: int = 10,
                 save: bool = True,
                 **kwargs,
                 
                 ):
        for i in range(num_batches):
            sample = self.sample() if sample == None else sample
            sample['models'] =  self.random_model_keys(num_models)
            sample['topk'] = topk
            kwargs.update(sample)
            output = self.forward(**kwargs)
        if save:
            self.save()
        
        return output

    def save(self, path: str = 'stats') -> Dict[str, Any]:
        
        self.put_json(path, self.stats)
            
    def load(self, path: str = 'stats') -> Dict[str, Any]:

        stats = self.get_json(path, default={})
           
        self.stats = stats
          
          

    def set_stats(self, stats: Dict[str, Any] = None) -> None:
        stats  = stats if stats != None else {}

        assert isinstance(stats, dict)
        self.stats = stats
        self.config['stats'] = stats
        
    @property
    def tag(self):
        return self.config.get('tag', 'base')
    
    @tag.setter
    def tag(self,tag):
        self.config['tag'] = tag
    
    def save(self, tag=None):
        tag = tag if tag else self.tag
        self.put(f'{tag}/config', self.config)
    def load(self, tag=None):
        tag = tag if tag else self.tag
        self.get(f'{tag}/config')
        
    def calculate_weights(self):
        
        
        total_weights = 0 
        weight_map = {}
        for k in self.stats.keys():
            weight_map[k] =  1 / (self.stats[k]['metric'] + 1e-8)
            total_weights = total_weights + weight_map[k]


        for k in self.stats.keys():
            weight_map[k] = weight_map[k] / total_weights
            self.stats[k]['weight'] = weight_map[k]
            
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
    
    

    
    @classmethod
    def run_train(cls, *args, **kwargs):
        
        if kwargs.pop('remote', False):
            return cls.remote_fn(fn='run_train', args=args, kwargs=kwargs)
        sleep_interval = kwargs.pop('sleep_interval', 3)
        stagger_interval = kwargs.pop('stagger_interval', 0)
        num_batches = kwargs.pop('num_batches', 2)
        
        self = Validator(*args, **kwargs)
        
        def sample_check(sample):
            return bool(isinstance(sample, dict) and 'input_ids' in sample)
        
        for _ in range(num_batches):
            sample = self.sample()
            if not sample_check(sample):
                continue
            output = self.forward(**sample)
            output.stats.pop('models', None)
            self.sleep(sleep_interval)
            
            
    @classmethod
    def ensure_defaults(params:dict, defaults:dict) -> dict:
        for k, v in defaults.items():
            if k not in params:
                params[k] = v
                
        return params
            
    @classmethod
    def test(cls,  
             *args,
             batch_size=8, 
             num_batches=2, 
             remote=False,
             **kwargs):
        
        kwargs = cls.locals2kwargs(locals())
     
        return cls.run_train(*args, **kwargs)
        
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
    
    @property
    def stats_table(self): 
        df_rows = []
        for k in self.stats.keys():
            self.stats[k]['model'] = k
            df_rows.append(self.stats[k])
            
        import pandas as pd
        df = pd.DataFrame(df_rows)
        
        return df

    @classmethod
    def streamlit(cls):
        
        import streamlit as st
        commune.new_event_loop(nest_asyncio=True)
        # commune.nest_asyncio()
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
    def miner(cls, 
               wallet='collective.0',
               network = 'finney',
               netuid=3,
               port = 9299,
               prometheus_port = 8299,
               debug = True,
               no_set_weights = True,
               remote:bool = False,
               ):
        
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='miner',name=f'miner::{wallet}',  kwargs=kwargs)
            
                
        
        config = bittensor.neurons.core_server.neuron.config()
        config.neuron.no_set_weights = no_set_weights
        config.axon.port = port 
        config.prometheus.port = config.axon.prometheus['port'] = prometheus_port if prometheus_port is not None else cls.free_port()
        config.netuid = netuid
        config.logging.debug = debug
        config.neuron.pretrained = False
        
        cls.print(config)
        
        
        model = cls()
        subtensor = bittensor.subtensor(network=network)
        server_class = commune.get_module('commune.bittensor.neuron.core_server.server')

        server = server_class(model=model, config=config)
        bittensor.utils.version_checking()
    
        coldkey, hotkey = wallet.split('.')
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        
        import time
        sleep_interval = 2
        while not wallet.is_registered(subtensor= subtensor, netuid=  netuid):
            time.sleep(sleep_interval)
            cls.print(f'Pending Registration {wallet} Waiting {sleep_interval}s ...')
            
        cls.print(f'Wallet {wallet} is registered on {network}')
             
             
        bittensor.neurons.core_server.neuron(model=server, 
               wallet=wallet,
               subtensor=subtensor,
               config=config,
               netuid=netuid).run()




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
 
if __name__ == '__main__':
    Validator.run()

        
