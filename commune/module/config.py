from typing import *
from munch import Munch
from copy import deepcopy
import os

class Config:
    def __init__(self, *args, **kwargs):
        self.set_config(*args, **kwargs)

    def set_config(self, 
                   config:Optional[Union[str, dict]]=None, 
                   kwargs:dict=None,
                   add_attributes: bool = False,
                   save_config:bool = False,
                   **extra_kwargs
                   ) -> Munch:
        '''
        Set the config as well as its local params
        '''
        kwargs = kwargs if kwargs != None else {}
        kwargs.update(extra_kwargs)

        # in case they passed in a locals() dict, we want to resolve the kwargs and avoid ambiguous args
        if config == None:
            config = self.load_config(config)
        
        for k,v in kwargs.items():
            config[k] = v

        if isinstance(config, dict):
            config = self.dict2munch(config)
        # get the config
        # add the config attributes to the class (via munch -> dict -> class )
        if add_attributes:
            self.__dict__.update(self.munch2dict(config))

        self.config = config 
        self.kwargs = kwargs

        if save_config:
            self.save_config(config=config)
            
        return self.config




    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = True , default=None) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''
        default = default or {}
        path = path if path else cls.config_path()
        if os.path.exists(path):
            config = cls.load_yaml(path)
        else:
            config = default

        config = config or {} 
        # convert to munch
        if to_munch:
            config =  cls.dict2munch(config)
        return config
    
    

    
    @classmethod
    def save_config(cls, config:Union[Munch, Dict]= None, path:str=None) -> Munch:

        '''
        Saves the config to a yaml file
        '''
        if config == None:
            config = cls.config()
        
        if isinstance(config, Munch):
            config = cls.munch2dict(deepcopy(config))
        elif isinstance(config, dict):
            config = deepcopy(config)
        else:
            raise ValueError(f'config must be a dict or munch, not {type(config)}')
        
        assert isinstance(config, dict), f'config must be a dict, not {config}'

        config = cls.save_yaml(data=config , path=path)

        return config
    
    
    def config_exists(self, path:str=None) -> bool:
        '''
        Returns true if the config exists
        '''
        path = path if path else self.config_path()
        return self.path_exists(path)

    

    @classmethod
    def dict2munch(cls, x:dict, recursive:bool=True)-> Munch:
        '''
        Turn dictionary into Munch
        '''
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, dict) and recursive:
                    x[k] = cls.dict2munch(v)
            x = Munch(x)
        return x 

    @classmethod
    def munch2dict(cls, x:Munch, recursive:bool=True)-> dict:
        '''
        Turn munch object  into dictionary
        '''
        if isinstance(x, Munch):
            x = dict(x)
            for k,v in x.items():
                if isinstance(v, Munch) and recursive:
                    x[k] = cls.munch2dict(v)

        return x 

    

        
    @classmethod
    def has_config(cls) -> bool:
        
        try:
            return os.path.exists(cls.config_path())
        except:
            return False
        
    
    @classmethod
    def config_path(cls) -> str:
        return os.path.abspath('./config.yaml')
