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


shortcuts =  {
    # 0-1B models
    'gpt125m': 'EleutherAI/gpt-neo-125m',

    # 1-3B models
    'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
    'gpt3b': 'EleutherAI/gpt-neo-2.7B',
    'opt1.3b': 'facebook/opt-1.3b',
    'opt2.7b': 'facebook/opt-2.7b',
    # 'gpt3btuning' : ''

    # 0-7B models
    'gptjt': 'togethercomputer/GPT-JT-6B-v1',
    'gptjt_mod': 'togethercomputer/GPT-JT-Moderation-6B',
    'gptj': 'EleutherAI/gpt-j-6b',
    'gptj.pyg6b': 'PygmalionAI/pygmalion-6b',
    'gpt6b': 'cerebras/Cerebras-GPT-6.7B',
    'gptj.instruct': 'nlpcloud/instruct-gpt-j-fp16',
    'gptj.codegen': 'moyix/codegen-2B-mono-gptj',
    'gptj.hivemind': 'hivemind/gpt-j-6B-8bit',
    'gptj.adventure': 'KoboldAI/GPT-J-6B-Adventure',
    'gptj.pygppo': 'TehVenom/GPT-J-Pyg_PPO-6B', 
    'gptj.alpaca.gpt4': 'vicgalle/gpt-j-6B-alpaca-gpt4',
    'gptj.alpaca': 'bertin-project/bertin-gpt-j-6B-alpaca',
    'oa.galactia.6.7b': 'OpenAssistant/galactica-6.7b-finetuned',
    'opt6.7b': 'facebook/opt-6.7b',
    'llama': 'decapoda-research/llama-7b-hf',
    'vicuna.13b': 'lmsys/vicuna-13b-delta-v0',
    'vicuna.7b': 'lmsys/vicuna-7b-delta-v0',
    'llama-trl': 'trl-lib/llama-7b-se-rl-peft',
    'opt.nerybus': 'KoboldAI/OPT-6.7B-Nerybus-Mix',
    'pygmalion-6b': 'PygmalionAI/pygmalion-6b',
    # # > 7B models
    'oa.pythia.12b': 'OpenAssistant/oasst-sft-1-pythia-12b',
    'gptneox': 'EleutherAI/gpt-neox-20b',
    'gpt20b': 'EleutherAI/gpt-neox-20b',
    'opt13b': 'facebook/opt-13b',
    'gpt13b': 'cerebras/Cerebras-GPT-13B',
    
        }



class Validator(commune.Module, nn.Module):
    shortcuts = shortcuts
    def __init__(self, 
                 **kwargs
                 ):
        
        self.default_models = [m for m,_ in commune.namespace('global', update=True).items() if m.startswith('model.')]

        loop = kwargs.pop('loop', None)
        self.set_event_loop(loop)
        
        nn.Module.__init__(self)
        self.set_config(kwargs=kwargs)
        config = self.config
        
        self.namespace = deepcopy(commune.namespace(config.network))
        self.set_max_stats_history(config.max_stats_history)
        self.set_batch_size(config.batch_size)
        self.set_sequence_length(config.sequence_length)
        self.set_dataset(config.dataset)
        self.set_metric(config.metric)
        self.set_stats(config.stats)
        self.set_alpha(config.alpha)
        self.set_tokenizer(config.tokenizer)
        
    
        
    
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
        tokenizer = self.shortcuts.get(tokenizer, tokenizer)
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
        

    def set_metric(self, metric = None) -> None:
        if metric is None:
            metric = torch.nn.CrossEntropyLoss()
        self.metric = metric
    def calculate_metric(self, x):
        
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
                topk: int = 4096,
                return_keys: List[str] = ['topk'],
                **kwargs ):
        
        
        
        sample = self.get_params(locals())
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
            return_keys=return_keys,
            train= train)
        
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
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                output_hidden_states: bool = False,
                models:str=None, 
                threshold: float = 4.0,
                timeout = 7,
                topk: int = None,
                sequence_length:int = None,
                selection_ratio: int = None,
                train: bool = None,
                verbose: bool = False,

                **kwargs ):
        config = self.config
        timer = self.timer()
        selection_ratio = selection_ratio if selection_ratio != None else config.selection_ratio
        topk = topk if topk != None else config.topk
        train = train if train != None else config.train
        
        if self.config.new_loop_per_forward:
            loop = self.new_event_loop()

        else:
            loop = self.get_event_loop()



        forwarded_models = self.random_ratio_selection(self.copy(self.default_models), ratio=selection_ratio)
        
        self.print(f'forwarding to {len(forwarded_models)} models ')
        sequence_length = sequence_length if sequence_length else self.config.sequence_length
        inputs_ids = input_ids[:, -sequence_length:]
        
        jobs = [asyncio.wait_for(self.async_forward(input_ids=input_ids, 
                                   model=model_key, 
                                   topk=topk, 
                                   timeout=timeout,
                                   train=train,
                                   **kwargs), timeout=timeout) for model_key in forwarded_models]
        
        model_outputs = loop.run_until_complete(asyncio.gather(*jobs))
        
        if verbose:
            self.print('RECIEVING RESPONSE FROM ',len(model_outputs), 'MODEL OUTPUTS')
        
        model_output_dict = {}
        for model_output, model_key in zip(model_outputs, forwarded_models):
            if model_output is None:
                continue
            model_output_dict[model_key] = model_output
            
        ensemble_logits = []
        
        stats = self.stats
        ensemble_stats = {  'passed': 0,
                          'successes': 0,
                          'failures': 0, 
                          'inference_time': 0.0,
                          'metric': 0.0, 
                          'timestamp': self.time(), 
                          'models': [],
                          'called_models':forwarded_models ,
                          'metrics': []}
        
        ensemble_stats['weights'] = []
        for model_key, output_dict in model_output_dict.items():
            
            if output_dict['stats']['success'] == False:
                ensemble_stats['failures'] += 1
                continue
            ensemble_stats['successes'] += 1
            model_stats= output_dict['stats']
            if model_stats['metric'] < threshold:
                
                ensemble_logits.append(output_dict['logits'])
                ensemble_stats['passed'] += 1
                ensemble_stats['models'] += [model_key]
                ensemble_stats['weights'] += [model_stats['metric']]
                ensemble_stats['metrics'] += [model_stats['metric']]

            else:
                model_stats['included'] = False
                
            stats[model_key] = model_stats
            

            logits = output_dict['logits']
               
               
        ensemble_weights = torch.tensor(ensemble_stats['weights'])
        ensemble_metircs = torch.tensor(ensemble_stats['metrics'])
        if len(ensemble_weights) >1:
            ensemble_weights = -(ensemble_weights - ensemble_weights.mean())/ (ensemble_weights.std())
            ensemble_weights = torch.softmax(ensemble_weights, dim=-1)
        else:
            ensemble_weights = torch.ones_like(ensemble_weights)
            
        ensemble_stats['weights']= ensemble_weights.tolist()
        ensemble_logits = torch.stack(ensemble_logits)
        ensemble_probs = torch.softmax(ensemble_logits, dim=-1)
        ensemble_probs = ensemble_probs * ensemble_weights[:,None,  None, None]
        ensemble_probs_unormalized = ensemble_probs.sum(0)
        ensemble_probs = ensemble_probs_unormalized / ensemble_probs_unormalized.sum(-1, keepdim=True)
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        output_dict = {
            'logits': ensemble_logits,
            'hidden_states': torch.randn(ensemble_logits.shape[0], ensemble_logits.shape[1], self.config.hidden_size),
        }
        output_dict['input_ids'] = input_ids
        
        ensemble_stats['metric'] = self.calculate_metric(output_dict)


        ensemble_stats['inference_time'] = timer.seconds
        stats['ensemble'] = ensemble_stats
        output_dict['stats'] = stats
        self.set_stats(stats)
        if verbose:
            self.print(stats)
        return Munch(output_dict)
    
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
        self.stats = stats if stats is not None else {}
        assert isinstance(self.stats, dict)
        
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
            stats = output.stats
            cls.print(output.stats['ensemble'])
            self.sleep(sleep_interval)
    @classmethod
    def test(cls,  *args, num_batches=2, **kwargs):
        cls.run_train(*args, num_batches=num_batches, **kwargs)
        
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
    def miner(cls, 
               wallet='collective.0',
               network = 'finney',
               netuid=3,
               port = 8091,
               debug = True,
               no_set_weights = True
               ):
                
            
    
        config = bittensor.neurons.core_server.neuron.config()
        config.neuron.no_set_weights = no_set_weights
        config.axon.port = port
        config.netuid = netuid
        config.logging.debug = debug
        config.neuron.pretrained = False
        
        
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

        
