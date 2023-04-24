
import os
import asyncio
from munch import Munch
import inspect
from typing import Optional, Union, Any , Dict, List
import torch
import random
import sys, os

from commune.utils.dict import load_yaml, save_yaml, load_json, save_json, chunk

import datasets
import inspect
from importlib import import_module
import commune

class HFDataset(commune.Module):
    def __init__(self,
                 path = None,
                **kwargs
                ): 
        
        self.config = self.set_config(kwargs=kwargs)
        self.config.path = path if path != None else self.config.path
        self.set_dataset(self.config)
        # if config.test:
        #     self.test()
        
        
        

    @property
    def default_text_feature(self):
        for k,v in self.features.items():
            if  v.dtype == 'string':
                return k
        assert False, 'No text feature found'


    
 

    def replicate(self, tag = None, **kwargs) -> None:
        '''
        Replicate the current module with a new tag.
        '''

        if isinstance(tag, list):
            for t in tag:
                self.replicate(tag=t)
        elif isinstance(tag, str) or tag is None:
            self.__class__.launch(kwargs={'config':self.config}, tag=tag,  **kwargs) 
        
        elif type(tag) in [int]:
            self.replicate(tag=str(tag))
        else:
            raise ValueError(f'Invalid tag type: {type(tag)}')
        
    def default_name(self):
        return self.available_names()[0]
        
    def set_dataset(self, config):
        
        assert isinstance(config, Munch)
        
        if hasattr(self, 'dataset'):
            self.config.update(config)


        # resolve  path and name(config)
        kwargs = {}
        path = config.path
        name = config.name
        seperator = config.get('seperator', '::')

        if len(path.split(seperator)) == 2:
            path, name = path.split(seperator)
        path = self.shortcuts.get(path, path)
        self.set_dataset_builder(path)

        
        config.name = name if name != None else self.default_name()
        config.path = path if path != None else self.default_path
        
        
        
        for k,v in config.items():
            if k in ['path', 'name', 'split', 'streaming']:
                kwargs[k] = config.get(k, None) 
                
        
        print('Loading dataset: ', kwargs)
        if not hasattr(self, 'load_dataset'):
            self.load_dataset = self.import_object('datasets.load_dataset')


        dataset = self.load_dataset(**kwargs)


        
        
        self.__dict__.update(config)
        self.dataset = dataset
        self.set_tokenizer(tokenizer=config.tokenizer)
        
        text_field = config.get('text_field', None)
        if text_field == None:
            text_field = self.default_text_feature
        self.text_field = text_field

        
        
 
        return self.dataset

    @property
    def info(self):
        return self.dataset._info.__dict__

    @property
    def features(self):
        return self.dataset._info.__dict__['features']


    def to(self, device):
        self.device = device
        return self.device

    default_receptor_path = 'bittensor.receptor.pool.module.ReceptorPoolModule'



    def set_tokenizer(self, tokenizer):
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer
            
        assert isinstance(tokenizer, str)
        self.print(f'setting {tokenizer} tokenizer...')
        assert isinstance(tokenizer, str, )
        tokenizer = self.shortcuts.get(tokenizer, tokenizer)
        self.config['tokenizer'] = tokenizer
        
        try:
            # HACK TO INCLUDE LLAMA TOKENIZER
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
        except ValueError:
            
            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)


        self.tokenizer = tokenizer
        
    
        self.std_tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast= True)
        self.std_tokenizer = prep_tokenizer(self.std_tokenizer)
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        self.token_translator = self.get_module('model.token_translator')(tokenizer=tokenizer, std_tokenizer=self.std_tokenizer)

        return self.tokenizer

    
    
    def tokenize(self, text: str = 'Whadup',
                 padding=True, 
                 truncation=True, 
                 max_length=256,
                 return_tensors='pt',
                 add_special_tokens=False,
                 device:str = None,
                 tokenizer: str = None, 
                 **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        tokenizer = tokenizer if tokenizer else self.tokenizer
        if isinstance(tokenizer, str):
            raise NotImplementedError
        sample = tokenizer(text, 
                                             padding=padding, 
                                             truncation=truncation, 
                                             max_length=max_length, 
                                             return_tensors=return_tensors,
                                             add_special_tokens=add_special_tokens, 
                                             **kwargs)  # assume tokenizer.padding_side = 'left'

        device = device if device != None else self.device
        
        sample = dict(
            input_ids= sample['input_ids'].to(device),
            attention_mask= sample['attention_mask'].to(device)
        )
        
        return sample



    @property
    def splits(self):
        available_splits =self.config['available_splits'] = self.config.get('available_splits', list(self.info['splits'].keys()))
        return available_splits
    available_splits = splits

    @property
    def split(self):
        return self.config['split']
    
    def set_split(self, split):
        assert split in self.available_splits
        self.config['split'] = split
        self.set_dataset(split=split)

    @split.setter
    def split(self, split):
        assert split in self.available_splits
        self.config['split'] = split
        self.set_dataset(split=split)

    def __len__(self):
        if not self.streaming:
            return len(self.dataset)
        else:
            return 1000


    def split_size_map(self):
        info_dict = self.info
        return {split: self.info_dict['splits'][split] for split in self.splits}


    @classmethod
    def list_datasets(cls) -> List[str]:
        # list 
        default_datasets = [
            'glue',
            'super_glue',
            'wikitext',
            'the_pile',
        
        ]
        return default_datasets 
        
    @classmethod
    def deploy_fleet(cls, datasets:List[str] = None, refresh: bool = True, **kwargs):
        datasets = datasets if datasets else cls.list_datasets()
        for dataset in datasets:
            commune.print(f'LAUNCHING {dataset} dataset', 'yellow')
            cls.launch(kwargs={'path':dataset}, name=f'dataset.text.{dataset}', refresh=refresh, **kwargs)
            
            
    @classmethod
    def deploy(cls, *datasets:List[str], 
               refresh: bool = True,
               tag_seperator:str = '::',
               **kwargs):
        for dataset in datasets:
            assert isinstance(dataset, str)
            commune.print(f'LAUNCHING {dataset} dataset', 'yellow')
            cls.launch(kwargs={'path':dataset}, name=f'dataset.{dataset}', refresh=refresh, **kwargs)
            
            
            
    def resolve_split(self, split:Optional[str]) -> str:
        if split == None:
            split = self.split
        else:
            assert split in self.splits
        return split
    
    def resolve_idx(self, idx:int):

        if isinstance(idx, int):
            assert idx >= 0 and idx < len(self)
        else:
            idx = random.randint(1,len(self)-1) 

        return idx

    def __getitem__(self, idx:Optional[int]=None, sequence_length:int=128):
        
        idx = self.resolve_idx(idx=idx)

        final_sample  = ''
        while len(final_sample.split()) < sequence_length:
            if self.streaming:
                sample = next(iter(self.dataset)).get(self.text_field)
            else:
                sample = self.dataset[idx].get(self.text_field)
            if sample == None:
                raise Exception(f'Please specify a valid text_field {list(self.dataset[idx].keys())} {self.text_field}')

            final_sample += sample if len(final_sample) == 0 else '\n' + sample
            idx = (idx + 1 ) % len(self)
        final_sample = ' '.join(final_sample.split()[:sequence_length])
        return final_sample

    def sample(self, batch_size:int=32, sequence_length:int=256, idx_list:List[int] = None, tokenize:bool= True)->dict:
        
        self.print(locals())
        if idx_list == None:
            idx_list = [self.resolve_idx(None) for i in range(batch_size)]
            
        sample_dict = {
            'text': [self.__getitem__(idx=idx ) for idx in idx_list]
        }
        
        if tokenize:
            sample_dict = self.tokenize(text=sample_dict['text'], max_length=sequence_length)

        return sample_dict
    
    forward = sample
    
    def resolve_device(self, device=None):
        if device == None:
            device = self.device
        return device

    classmethod
    def test_model_sample(cls):
        self = cls()
        batch_size=12
        seqeunce_length = 256
        x = self.sample(batch_size=batch_size,seqeunce_length=seqeunce_length )
        assert x.shape[0] == batch_size
        assert x.shape[1] == seqeunce_length


    # @property
    # def pipeline_tags_count(self):
    #     count_dict = dict(self.models_df['pipeline_tag'].value_counts())
    #     return {k:int(v) for k,v in count_dict.items()}

    @staticmethod
    def resolve_filter_fn(filter_fn):
        if filter_fn != None:
            if callable(filter_fn):
                fn = filter_fn

            if isinstance(filter_fn, str):
                filter_fn = eval(f'lambda r : {filter_fn}')
        
            assert(callable(filter_fn))
        return filter_fn

    default_path = 'glue'

    @classmethod
    def resolve_dataset_path(cls, path:str=None, **kwargs):
        path = cls.shortcuts.get(path, path)
        if path == None:
            path = cls.default_path
        return path
    @classmethod
    def get_dataset_builder( cls, path:str=None, factory_module_path:str=None):
        path = cls.resolve_dataset_path(path=path)
        if factory_module_path == None:
            
            assert isinstance(path, str)
            factory_module = datasets.load.dataset_module_factory(path)
            factory_module_path = factory_module.module_path

        dataset_builder = datasets.load.import_main_class(factory_module_path)
        return dataset_builder
    
    

    def set_dataset_builder( self, *args, **kwargs):
        self.dataset_builder = self.get_dataset_builder(*args, **kwargs)
        return self.dataset_builder

    @classmethod
    def list_configs(cls, *args, **kwargs):
        return cls.config_map(*args, **kwargs).keys()

    def available_names(self):
        return list(self.config_map.keys())
    
    list_names = available_names

    @classmethod
    def configs(cls, *args, names_only:bool = True, **kwargs):
        
        configs = cls.get_dataset_builder(*args, **kwargs).BUILDER_CONFIGS
        if names_only:
            configs = [config.name for config in configs]
        return configs
    @property
    def config_map(self):

        configs = [config.__dict__ for config in self.dataset_builder.BUILDER_CONFIGS]

        if len(configs) == 0:
            configs =  [self.dataset_builder('default').info.__dict__]
            configs[0]['name'] = 'default'

        config_map = {config['name']: config for config in configs}     

        return config_map

    @property
    def card(self) -> Dict[str,str]:
        return dict(
            module = self.module_path,
            path = self.path,
            name = self.name,
            split = self.split
        )


    @classmethod
    def test(cls, *args, **kwargs):
    
        self = cls( *args, **kwargs)
        x = self.sample()
        print(x)

    shortcuts =  {
        'pile': 'EleutherAI/the_pile',
    }
    
    tokenizer_shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6b',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
         'gpt3b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b',
        'gpt2': 'gpt2',
         }

    @classmethod
    def available_datasets(cls, prefix:str='dataset') -> List[str]:
        return [x for x in commune.servers() if x.startswith(prefix)]
    
    @classmethod
    def default_dataset(cls) -> str:
        available_datasets = cls.available_datasets()
        if len(available_datasets) == 0:
            return cls.launch(name='dataset.text.glue', kwargs=dict(path='glue'))
        return commune.connect(dataset_name)
    
    

    @classmethod
    def test_multiple(cls):
        for path in cls.list_datasets():
            cls.print(f'TESTING ({cls.module_path()}): {path}', 'yellow')
            self = cls(path=path)
            sample = self.sample(tokenize=False)
            assert 'text' in sample
            sample = self.sample(tokenize=True)
            assert 'input_ids' in sample
            cls.print(f'PASSED ({cls.module_path()}): {path}', 'green')

    @classmethod
    def sandbox(cls):
        import streamlit as st
        self = cls()
        for i in range(1000):
            self.sample()
            print(i)
        


if __name__ == '__main__':
    HFDataset.run()
    
    