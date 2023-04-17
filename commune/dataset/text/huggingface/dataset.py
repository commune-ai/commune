
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
                path: str = 'pile',
                name:str = None,
                text_field:str = None,
                split: str = 'train',
                sample_index = [0,1000],
                streaming: bool = False,
                tokenizer: str =  'gpt2',
                device: str = 'cpu',
                config: dict=None
                ):
        params = locals()
        params.pop('self')
        self.set_params(**params)

    @property
    def default_text_feature(self):
        for k,v in self.features.items():
            if  v.dtype == 'string':
                return k
        assert False, 'No text feature found'

        
 
    def set_params(self, **kwargs) -> None:
        path = kwargs.get('path')
        name = kwargs.get('name')
        split = kwargs.get('split')
        streaming = kwargs.get('streaming')
        sample_index = kwargs.get('sample_index')
        device = kwargs.get('device')
        tokenizer = kwargs.get('tokenizer')

        self.config = self.set_config(kwargs)
    
        # self.__dict__.update(self.config)
        
        self.set_tokenizer(tokenizer=tokenizer)
        self.set_dataset(path=path, 
                         name=name, 
                         split=split, 
                         streaming=streaming, 
                         sample_index = sample_index)
        
        
        self.text_field = self.config.get('text_field', self.default_text_feature)

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
    def set_dataset(self, path:str,
                    name:str=None, 
                    split:str=None,
                    streaming: bool = False,
                    sample_index : List[int] = None):
        kwargs = {}
        path = self.shortcuts.get(path, path)
        self.path = path
        if name == None:
            name = self.available_names[0]
        
        kwargs['name'] = name
        kwargs['path'] = path
        kwargs['split'] = split
        kwargs['streaming'] = streaming
        
        for k,v in kwargs.items():
            if v != None:
                self.__dict__[k] = self.config[k] = v

        if sample_index:
            assert isinstance(sample_index, list)
            assert isinstance(sample_index[0], int)
            assert isinstance(sample_index[1], int)
            kwargs['split'] = f'{split}[:{10}%]'    
        kwargs['split'] = f'{split}[:{10}%]'    
        print('Loading dataset: ', kwargs)
        if not hasattr(self, 'load_dataset'):
            self.load_dataset = self.import_object('datasets.load_dataset')

        self.dataset = self.load_dataset(**kwargs)
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


    def tokenize(self, text: str = 'Whadup',
                 padding=True, 
                 truncation=True, 
                 max_length=64,
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
        
        ]
        return default_datasets 
        
    @classmethod
    def deploy_fleet(cls, datasets:List[str] = None, refresh: bool = True, **kwargs):
        datasets = datasets if datasets else cls.list_datasets()
        for dataset in datasets:
            commune.print(f'LAUNCHING {dataset} dataset', 'yellow')
            cls.launch(kwargs={'path':dataset}, name=f'dataset.text.{dataset}', refresh=refresh, **kwargs)
            
            
    @classmethod
    def deploy(cls, *datasets:List[str], refresh: bool = True, **kwargs):
        for dataset in datasets:
            assert isinstance(dataset, str)
            commune.print(f'LAUNCHING {dataset} dataset', 'yellow')
            cls.launch(kwargs={'path':dataset}, name=f'dataset.text.{dataset}', refresh=refresh, **kwargs)
            
            
            
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

    def sample(self, batch_size:int=32, sequence_length:int=256, idx_list:List[int] = None, tokenize:bool= False)->dict:
        
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

    @staticmethod
    def set_dataset_builder( path:str=None, factory_module_path:str=None):
        if factory_module_path == None:
            assert isinstance(path, str)
            factory_module = datasets.load.dataset_module_factory(path)
            factory_module_path = factory_module.module_path

        dataset_builder = datasets.load.import_main_class(factory_module_path)
        return dataset_builder

    @staticmethod
    def set_dataset_factory( path:str):
        return datasets.load.dataset_module_factory(path)

    @property
    def dataset_factory(self):
        placeholder_name = '_dataset_factory'
        if not hasattr(self, placeholder_name):
            setattr(self, placeholder_name,self.set_dataset_factory(self.path))
        return getattr(self, placeholder_name)

    @property
    def dataset_builder(self):
        placeholder_name = '_dataset_builder'
        if not hasattr(self, placeholder_name):
        
            setattr(self, placeholder_name,self.set_dataset_builder(self.path))
        return getattr(self, placeholder_name)



    def list_configs(self):
        return self.config_map

    @property
    def available_names(self):
        available_names = self.config['available_names'] = self.config.get('available_names', list(self.config_map.keys()))
        return available_names

    def builder_configs(self):
        return self.dataset_builder.BUILDER_CONFIGS
        
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
        self = cls(*args, **kwargs)
        # self.serve()
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
    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        tokenizer = tokenizer if tokenizer else 'gpt2'
        from transformers import AutoTokenizer
        
        if isinstance(tokenizer, str):
            tokenizer = self.tokenizer_shortcuts.get(tokenizer, tokenizer)
            self.config['tokenizer'] = tokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        return self.tokenizer




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
    def test(cls):
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
    
    