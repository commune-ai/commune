import commune as c
import datasets
from datasets import load_dataset
from typing import Dict, List



class DataHF(c.Module):
    
    shortcuts = {
        'pile': 'EleutherAI/the_pile',
        'wiki': 'wikitext',
        'glue': 'glue',
        'camel_math': 'camel-ai/math',
        'mmlu': 'lukaemon/mmlu',
        'pubmed_qa': 'pubmed_qa',
        'truthqa': 'truthful_qa',
    }
    def __init__(self, 
                path: str = 'super_glue',
                name: str =  None,
                streaming: bool= False,
                split: str = None, 
                **kwargs):
        config = self.set_config(locals())
        self.set_dataset(path=config.path, name=config.name, split=config.split, streaming=config.streaming)
    

        
    def set_dataset(self, path:str, name:str = None, split:str = None, streaming:bool=False, **kwargs):
        path = self.shortcuts.get(path, path)
        c.print(f'Loading dataset: {name} from {path}')

        # resolve name
        name = list(self.config_names(path=path))[0] if name == None else name



        # raise Exception(f'Loading dataset: {name} from {path}')

        # resolve split
        if isinstance(split, str):
            split = [split]

        # update config
        self.config.update({'path': path, 'name': name, 'split': split, 'streaming': streaming})
        
        # load dataset
        dataset_map = load_dataset(path=path,
                                     name=name,
                                       split=split, 
                                       streaming=streaming)
        
        # set attributes
        self.splits = list(dataset_map.keys())
        self.dataset = list(dataset_map.values())[0]
        self.dataset_map = dataset_map

     
        return self.dataset

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
        return self.config_names()[0]


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
    def get_dataset_builder( cls, path:str, factory_module_path:str=None):
        path = cls.shortcuts.get(path, path)
        if factory_module_path == None:
            assert isinstance(path, str)
            factory_module = datasets.load.dataset_module_factory(path)
            factory_module_path = factory_module.module_path

        dataset_builder = datasets.load.import_main_class(factory_module_path)
        return dataset_builder

    @classmethod
    def config_names(self, path=None):
        return list(self.config_map(path=path).keys())
    
    list_names = config_names


    @classmethod
    def configs(cls, path=None):
        configs = cls.config_map(path=path)
        return list(configs.keys())
    
    @classmethod
    def config_map(cls, path=None):
        dataset_builder = cls.get_dataset_builder(path)
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
        datasets = list(shortcuts.keys())
        for path in datasets:
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
    def serve_category(cls, fleet:str = 'qa', remote:bool=True, tag=None,  **kwargs):
        '''
        ### Documentation
        
        #### Function: `serve_category`
        
        **Description:**
        
        This class method is responsible for serving a category of services defined in a fleet. It launches each service on a separate port, avoiding any conflicts with already used ports.
        
        **Parameters:**
        
        - `fleet`: A string representing the fleet name. Default value is `'qa'`.
        - `remote`: A boolean indicating whether the service is to be served remotely or not. Default value is `True`.
        - `tag`: An optional parameter
        '''
        fleet = cls.getc(f'fleet.{fleet}')

        avoid_ports = []
        for path in fleet:
            port = c.free_port(avoid_ports=avoid_ports)
            cls.serve(path=path, remote=remote, port=port, tag=tag, **kwargs)
            avoid_ports.append(port)

    @classmethod
    def fleet(cls, path:str = 'truthful_qa', n:int=5, remote:bool=True, tag=None,  **kwargs):
        '''
        ## Fleet Class Method
        
        ### Description
        The `fleet` method is a class method responsible for starting multiple instances of a service on different ports. This method is useful when you want to run several instances of the same service simultaneously, possibly for load balancing or high availability purposes.
        
        ### Parameters
        - `path` (str): The path to the service that needs to be served. Defaults to 'truthful_qa'.
        - `n` (int): The number of instances to be started. Defaults
        '''

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
        


