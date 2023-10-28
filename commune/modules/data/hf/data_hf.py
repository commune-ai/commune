import commune as c
import datasets
from datasets import load_dataset
from typing import Dict, List

class DataHF(c.Module):
    
    def __init__(self, config = None, **kwargs):
        config = self.set_config(config, kwargs=kwargs)
        self.set_dataset(path=config.path, name=config.name, split=config.split, streaming=config.streaming)
    
    def __len__(self):
        return len(self.dataset)
    
    @property
    def n(self):
        return len(self)
    

    def random_idx(self):
        return c.random_int(len(self))
    
        
    def sample(self, idx:int=None, batch_size:int = 1):
        if batch_size > 1:
            return [self.sample() for i in range(batch_size)]
        idx = self.random_idx() if idx == None else idx
        return self.dataset[idx]
    



    def validate(self, module, num_samples=10):
        for i in range(num_samples):
            idx = self.random_idx()
            sample = self.sample(idx=idx)
            module_sample = module.sample(idx=idx)
            for key in sample.keys():
                if sample[key] != module_sample[key]:
                    return 0
        return 1

    @classmethod
    def test(cls, *args,**kwargs):
        cls.print('Testing dataset')
        dataset = cls(*args, **kwargs)
        sample = dataset.sample()
        assert isinstance(sample, dict)
        return sample
    
        
    def default_name(self):
        return self.available_names()[0]

        
    def set_dataset(self, path:str = None, name:str = None, split:str = None, streaming:bool=None):
        
        
        config = self.config

        # resolve path
        path = path if path else config.path
        path = self.config.shortcuts.get(path, path)
        self.dataset_builder = self.get_dataset_builder(path=path)

        # resolve name
        name = name if name else config.name
        name = self.default_name() if name == None else name

        # raise Exception(f'Loading dataset: {name} from {path}')

        # resolve split
        split = split if split else config.split
        streaming = streaming if streaming else config.streaming
        if isinstance(split, str):
            split = [split]

        # load dataset
        dataset_map = load_dataset(path=path,
                                     name=name,
                                       split=split, 
                                       streaming=streaming)
        
        # set attributes
        self.splits = list(dataset_map.keys())
        self.dataset = list(dataset_map.values())[0]
        self.dataset_map = dataset_map

        # update config
        self.config.update({'path': path, 'name': name, 'split': split, 'streaming': streaming, 'splits': self.splits})
        
        return self.dataset

    @property
    def data_info(self):
        return self.dataset._info.__dict__

    @property
    def features(self):
        return self.dataset._info.__dict__['features']
    
    
    def set_split(self, split):
        self.dataset = self.dataset_map[split]
        return self.dataset

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
    


    def available_names(self):
        return list(self.config_map().keys())
    
    list_names = available_names


    def configs(self,):
        configs = self.config_map()
        return list(configs.keys())
    
    def config_map(self):

        dataset_builder = self.get_dataset_builder(self.config.path)
        configs = [config.__dict__ for config in dataset_builder.BUILDER_CONFIGS]

        if len(configs) == 0:
            configs =  [dataset_builder._info.__dict__]
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

    @property
    def name_suffix(self):
        return f'{self.path}'

    @classmethod
    def serve(cls, path:str = 'truthqa', tag=None, remote:bool=True, **kwargs):
        server_name = f'data.{path}'
        kwargs = dict(path=path, **kwargs)
        c.print(f'Serving {server_name} with kwargs: {kwargs}')
        name = c.serve(module=cls.module_path(), server_name=server_name, tag=tag, kwargs=kwargs, remote=remote)['name']
        return name


    @classmethod
    def serve_category(cls, fleet:str = 'qa', remote:bool=True, tag=None,  **kwargs):
        fleet = cls.getc(f'fleet.{fleet}')

        avoid_ports = []
        for path in fleet:
            port = c.free_port(avoid_ports=avoid_ports)
            cls.serve(path=path, remote=remote, port=port, tag=tag, **kwargs)
            avoid_ports.append(port)

    @classmethod
    def fleet(cls, path:str = 'truthful_qa', n:int=5, remote:bool=True, tag=None,  **kwargs):

        avoid_ports = []
        for i in range(n):
            port = c.free_port(avoid_ports=avoid_ports)
            cls.serve(path=path, remote=remote, port=port, tag=f'{i}' if tag == None else f'{tag}.{i}', **kwargs)
            avoid_ports.append(port)


    @classmethod
    def validate(cls, module = None, ref_module=None):
        module = c.connect(module)
        ref_module = c.connect(ref_module)
        ref_idx = ref_module.random_idx()
        ref_sample = ref_module.sample(idx=ref_idx)
        reference_sample = module.sample(idx=ref_idx)

        module.sample(idx=0)
        


