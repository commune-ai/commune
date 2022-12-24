
import os
import asyncio
from munch import Munch
import inspect
from typing import Optional, Union, Any , Dict, List
import torch
from cortex.utils import load_yaml, save_yaml, load_json, save_json, chunk
import bittensor
import datasets
import inspect
from importlib import import_module

class HuggingfaceDataset:
    def __init__(self,
                path:str=None,
                name:str = None,
                split:str=None,
                tokenizer:'tokenizer'=None, 
                config: dict=None, 
                **kwargs):


        self.config = self.load_config(config=config)
        
        self.load_tokenizer(tokenizer=tokenizer)
        self.load_dataset(path=path, name=name, split=split)

    def load_tokenizer(self, tokenizer=None): 
        tokenizer = tokenizer if tokenizer else self.config['tokenizer']
        self.tokenizer = self.launch(**tokenizer)
        return self.tokenizer
    def getattr(self, k):
        return getattr(self, k)
        
    def load_dataset(self, path:str=None, name:str=None, split:str=None):
        kwargs = {}
        kwargs['path'] = path if path  else self.path
        kwargs['name'] = name if name  else self.name
        kwargs['split'] = split if split  else self.split

        self.dataset = self.launch(module='datasets.load_dataset', kwargs=kwargs)
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

    def sample(self, batch_size=10, sequence_length=16, random=True, idx_list = None, tokenize=False, padding=True)->dict:
        
        if idx_list == None:
            idx_list = [None for i in range(batch_size)]

        samples =  [self.__getitem__(idx=idx ) for idx in idx_list]

        sample_dict = {'text': samples}
        if tokenize:
            sample_dict['input_ids'] = self.tokenizer(sample_dict['text'], padding=padding,  max_length=sequence_length, truncation=True, return_tensors='pt')['input_ids']
            
        return sample_dict
    
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


    @property
    def pipeline_tags_count(self):
        count_dict = dict(self.models_df['pipeline_tag'].value_counts())
        return {k:int(v) for k,v in count_dict.items()}

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


    @classmethod
    def launch(cls, module:str, fn:str=None ,kwargs:dict={}, args=[]):
        module_class = cls.import_object(module)

        module_init_fn = fn
        module_kwargs = {**kwargs}
        module_args = [*args]
        
        if module_init_fn == None:
            module_object =  module_class(*module_args,**module_kwargs)
        else:
            module_init_fn = getattr(module_class,module_init_fn)
            module_object =  module_init_fn(*module_args, **module_kwargs)
        return module_object

    @property
    def card(self) -> Dict[str,str]:
        return dict(
            module = self.module_path,
            path = self.path,
            name = self.name,
            split = self.split
        )

    @property
    def __file__(self):
        module_path =  inspect.getmodule(self).__file__
        return module_path

    @property
    def __config_file__(self):
        return self.__file__.replace('.py', '.yaml')

    def load_config(self, config:Optional[Union[str, dict]]=None):
        if config == None:
            config = load_yaml(self.__config_file__)
        elif isinstance(config, str):
            config =  load_yaml(config)
        elif isinstance(config, dict):
            config = config
        
        return config

    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        return import_module(import_path)

    @classmethod
    def import_object(cls, key:str)-> 'Object':
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        obj =  getattr(import_module(module), object_name)
        return obj


    @property
    def module_path(self):
        local_path = self.__file__.replace(os.getenv('PWD'), '')
        local_path, local_path_ext = os.path.splitext(local_path)
        module_path ='.'.join([local_path.replace('/','.')[1:], self.__class__.__name__])
        return module_path

if __name__ == '__main__':
    DatasetModule.run()
