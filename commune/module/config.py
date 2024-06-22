from typing import *
from munch import Munch
from copy import deepcopy
import commune as c

class Config:

    def set_config(self, 
                   config:Optional[Union[str, dict]]=None, 
                   kwargs:dict=None,
                   to_munch: bool = True,
                   add_attributes: bool = False,
                   save_config:bool = False) -> Munch:
        '''
        Set the config as well as its local params
        '''
        kwargs = kwargs if kwargs != None else {}

        # in case they passed in a locals() dict, we want to resolve the kwargs and avoid ambiguous args
        kwargs = c.locals2kwargs(kwargs)

        if 'config' in kwargs:
            config = kwargs.pop('config')

        # get the config
        config =  self.config(config=config,kwargs=kwargs, to_munch=to_munch)

        # add the config attributes to the class (via munch -> dict -> class )
        if add_attributes:
            self.__dict__.update(self.munch2dict(config))

        self.config = config 
        self.kwargs = kwargs

        if save_config:
            self.save_config(config=config)
            
        return self.config




    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = False) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''
        if path == None: 
            path = cls.config_path()
        else:
            path = c.tree().get(path, path).replace('.py', '.yaml')
        config = cls.load_yaml(path)

        config = config or {} 
        # convert to munch
        if to_munch:
            config =  cls.dict2munch(config)
        return config
    
    
    default_config = load_config



    @classmethod
    def putc(cls, k, v, password=None) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.config()
        if password:
            v = cls.encrypt(v, password=password)
        config[k] =  v 
        cls.save_config(config=config)
        return {'success': True, 'msg': f'config({k} = {v})'}
    setc = putc
    @classmethod
    def rmc(cls, k, password=None) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.config()
        c.dict_rm(config, k)
        cls.save_config(config=config)
   
    delc = rmc



    @classmethod  
    def getc(cls, key, default= None, password=None) -> Any:
        '''
        Saves the config to a yaml file
        '''

        config = cls.config()
        data = config.get(key, default)
        
        if c.is_encrypted(data):
            if password == None:
                return data
            data = c.decrypt(data, password=password)
            
        return data




    
    @classmethod
    def save_config(cls, config:Union[Munch, Dict]= None, path:str=None) -> Munch:

        '''
        Saves the config to a yaml file
        '''
        if config == None:
            config = cls.config()
        
        path = path if path else cls.config_path()
        
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
    def config(cls, 
                   config:dict = None,
                   kwargs:dict=None, 
                   to_munch:bool = True) -> Munch:
        '''
        Set the config as well as its local params
        '''
        # THIS LOADS A YAML IF IT EXIST, OR IT USES THE INIT KWARGS IF THERE IS NO YAML
        if cls.has_config():
            default_config = cls.load_config(to_munch=False)
        else: 
            default_config = cls.init_kwargs()

        if config == None:
            config =  default_config
        elif isinstance(config, str):
            config = cls.load_config(path=config)
            assert isinstance(config, dict), f'config must be a dict, not {type(config)}'
        
        if isinstance(config, dict):
            config = {**default_config, **config}
        else:
            raise ValueError(f'config must be a dict, str or None, not {type(config)}')
                
        # SET THE CONFIG FROM THE KWARGS, FOR NESTED FIELDS USE THE DOT NOTATION, 
        # for example  model.name=bert is the same as config[model][name]=bert
        # merge kwargs with itself (CAUTION THIS DOES NOT WORK IF KWARGS WAS MEANT TO BE A VARIABLE LOL)

        config = c.locals2kwargs(config)

        if kwargs != None:
            kwargs = c.locals2kwargs(kwargs)
            for k,v in kwargs.items():
                config[k] = v 
        #  add the config after in case the config has a config attribute lol
        if to_munch:
            config = cls.dict2munch(config)


        return config
 
    cfg = get_config = config


    