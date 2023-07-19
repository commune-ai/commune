import commune as c
import datasets
from datasets import load_dataset
from typing import Dict, List

class HFDataset(c.Module):
    
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.set_dataset(self.config)
    
    def __len__(self):
        return len(self.dataset)
    
        
    def sample(self, idx:str=None):
        if idx is None:
            idx = c.random_int(len(self))

        return self.dataset[idx]
    
    @classmethod
    def test(cls, *args,**kwargs):
        cls.print('Testing dataset')
        dataset = cls(*args, **kwargs)
        c.print(dir(dataset))
        sample = dataset.sample()
        print(sample)
        
        assert isinstance(sample, dict)
        return sample
    
        
    def default_name(self):
        return self.available_names()[0]
        
    def set_dataset(self, config):
        config.path = self.shortcuts.get(config.path, config.path)
        self.dataset_builder = self.get_dataset_builder(path=config.path)
        config.name = self.default_name() if config.name == None else config.name
        self.dataset = load_dataset(path=config.path,
                                     name=config.name,
                                       split=config.split, 
                                       streaming=config.streaming)
        
        self.config = config
 
        return self.dataset

    @property
    def info(self):
        return self.dataset._info.__dict__

    @property
    def features(self):
        return self.dataset._info.__dict__['features']


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
        self.set_split(split)
    
    def set_split(self, split):
        assert split in self.available_splits
        self.config['split'] = split
        self.set_dataset(split=split)

    @classmethod
    def list_datasets(cls):
        return cls.getc('datasets')    
            
    @classmethod
    def get_dataset_builder( cls, path:str=None, factory_module_path:str=None):
        path = cls.getc('shortcuts').get(path, path)
        if path == None:
            path = cls.getc('path')
        if factory_module_path == None:
            
            assert isinstance(path, str)
            factory_module = datasets.load.dataset_module_factory(path)
            factory_module_path = factory_module.module_path

        dataset_builder = datasets.load.import_main_class(factory_module_path)
        return dataset_builder
    
    

    @classmethod
    def list_configs(cls, *args, **kwargs):
        return cls.config_map(*args, **kwargs).keys()

    def available_names(self):
        return list(self.config_map().keys())
    
    list_names = available_names

    @classmethod
    def configs(cls, path = None, names_only:bool = True,):
        if path == None: 
            path = cls.getc('path')
        
        configs = cls.get_dataset_builder(path).BUILDER_CONFIGS
        if names_only:
            configs = [config.name for config in configs]
        return configs
    @classmethod
    def config_map(cls, path=None):
        if path == None:
            path = cls.getc('path') 

        dataset_builder = cls.get_dataset_builder(path)
        configs = [config.__dict__ for config in dataset_builder.BUILDER_CONFIGS]

        if len(configs) == 0:
            configs =  [dataset_builder.info.__dict__]
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
        'wiki': 'wikitext',
        'glue': 'glue'
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
    
    