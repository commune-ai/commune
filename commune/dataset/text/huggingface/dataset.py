
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
                path:str='glue',
                name:str = 'cola',
                split:str='train',
                tokenizer:'tokenizer'=None, 
                text_field: str='sentence',
                config: dict=None, 
                **kwargs):
        self.config = self.set_config(config=config)

        self.path = path
        self.name = name
        self.split = split
        self.text_field = text_field
        
        
        self.load_tokenizer(tokenizer=tokenizer)
        self.load_dataset(path=path, name=name, split=split)

    def load_tokenizer(self, tokenizer=None): 
        try:
            import bittensor
        except RuntimeError:
            commune.new_event_loop()
            import bittensor
        tokenizer = tokenizer if tokenizer else bittensor.tokenizer()
        self.tokenizer = tokenizer
        return self.tokenizer
    

    def load_dataset(self, path:str=None, name:str=None, split:str=None):
        kwargs = {}
        kwargs['path'] = path if path  else self.path
        kwargs['name'] = name if name  else self.name
        kwargs['split'] = split if split  else self.split

        self.dataset = self.import_object('datasets.load_dataset')(**kwargs)
        return self.dataset

    @property
    def info(self):
        return self.dataset._info.__dict__

    @property
    def features(self):
        return self.dataset._info.__dict__['features']


    @property
    def device(self):
        device = self.config.get('device', 'cpu')
        if 'cuda' in device:
            assert torch.cuda.is_available()
        return device

    @device.setter
    def device(self, device):
        if 'cuda' in device:
            assert torch.cuda.is_available()
        self.config['device'] = device
        return device

    def to(self, device):
        self.device = device
        return self.device

    default_receptor_path = 'bittensor.receptor.pool.module.ReceptorPoolModule'

    def tokenize(self, text, padding=True, *args, **kwargs):
        device = kwargs.pop('device', self.device)
        return torch.tensor(self.tokenizer(text=text, padding=padding)['input_ids']).to(device)

    @property
    def splits(self):
        available_splits =self.config['available_splits'] = self.config.get('available_splits', list(self.info['splits'].keys()))
        return available_splits
    available_splits = splits

    @property
    def split(self):
        return self.config['split']

    @split.setter
    def split(self, split):
        assert split in self.available_splits
        self.config['split'] = split
        self.load_dataset(split=split)

    def __len__(self):
        return len(self.dataset)

    def split_size_map(self):
        info_dict = self.info
        return {split: self.info_dict['splits'][split] for split in self.splits}



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
            sample = self.dataset[idx].get(self.text_field)
            assert sample != None, f'Please specify a valid text_field {self.dataset[idx]}'

            final_sample += sample if len(final_sample) == 0 else '\n' + sample
            idx = (idx + 1 ) % len(self)
        final_sample = ' '.join(final_sample.split()[:sequence_length])
        return final_sample

    def sample(self, batch_size:int=32, sequence_length:int=256, idx_list:List[int] = None, tokenize:bool= True)->dict:
        
        if idx_list == None:
            idx_list = [None for i in range(batch_size)]

        samples_text =  [self.__getitem__(idx=idx ) for idx in idx_list]

        sample_dict = {}

        if tokenize:
            sample_dict['input_ids'] = self.tokenizer(samples_text,   max_length=sequence_length, truncation=True, padding="max_length", return_tensors='pt')['input_ids']
        else:
            sample_dict['text'] = samples_text
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
    def load_dataset_builder( path:str=None, factory_module_path:str=None):
        if factory_module_path == None:
            assert isinstance(path, str)
            factory_module = datasets.load.dataset_module_factory(path)
            factory_module_path = factory_module.module_path

        dataset_builder = datasets.load.import_main_class(factory_module_path)
        return dataset_builder

    @staticmethod
    def load_dataset_factory( path:str):
        return datasets.load.dataset_module_factory(path)

    @property
    def dataset_factory(self):
        placeholder_name = '_dataset_factory'
        if not hasattr(self, placeholder_name):
            setattr(self, placeholder_name,self.load_dataset_factory(self.path))
        return getattr(self, placeholder_name)

    @property
    def dataset_builder(self):
        placeholder_name = '_dataset_builder'
        if not hasattr(self, placeholder_name):
            setattr(self, placeholder_name,self.load_dataset_builder(self.path))
        return getattr(self, placeholder_name)

    @property
    def path(self):
        return self.config['path']
    
    name = path

    @path.setter
    def path(self, value):
        self.config['path'] = value

    @property
    def text_field(self):
        return self.config['text_field']

    @text_field.setter
    def text_field(self, value):
        self.config['text_field'] = value

    @property
    def name(self):
        name = self.config['name'] = self.config.get('name', self.available_names[0])
        return name

    @name.setter
    def name(self, name):
        self.config['name'] = name
        self.load_dataset(name=name)

    def list_configs(self):
        return self.config_map

    @property
    def available_names(self):
        available_names = self.config['available_names'] = self.config.get('available_names', list(self.config_map.keys()))
        return available_names

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
    def test(cls):
        self = cls()
        # self.serve()
        x = self.sample()
        print(x)


if __name__ == '__main__':
    # print(commune.Module.connect('dataset.huggingface').forward())
    HFDataset.run()