

import os
import re
import sys
import yaml
import glob
from munch import Munch
from typing import List, Optional, Union, Any, Dict, Callable


class Config ( Munch ):
    """
    Implementation of the config class, which manages the config of different bittensor modules.
    """
    root = '/'.join(__file__.split('/')[:-2]) # get root from __file__ == {root}/config/config_module.py

    def __init__(self, config: Union[str, Dict, Munch]=None, *args, **kwargs,   ):
        """
        Args:
            config (Union[str, Dict, Munch]):
                - str: path to config file or directory of config files
        
        """

            
        config = config if config else {}

        if isinstance(config, str) :
            
            self.config_path = config
            config = self.load_config(path=self.config_path)
        elif config == None:
            config = {}
        self.config = config
        assert isinstance(self.config, dict) ,  f'The self.config should be a dictionary but is {type(self.config)}'

        Munch.__init__(self, self.config, *args, **kwargs)
        self.recursive_munch(self)
    @staticmethod
    def recursive_munch(config):
        if isinstance(config, dict):
            for k,v in config.items():
                if isinstance(v, dict):
                    config[k] = Config.recursive_munch(v)

            config = Munch(config)
        
        return config 


    def load(self, path:str ,override:Dict[str, Any]={}):
        self.cache = {}
        self.config = self.load_config(path=path)
        if self.config == None:
            return {}

        if isinstance(override, dict) and len(override) > 0:
            self.config = self.override_config(config=self.config, override=override)
        
        self.config = self.resolver_methods(config=self.config)
        self.config = self.recursive_munch(self.config)
        return self.config


    def save_config(self, path:str=None, config:str=None):

        config = config if config else self.config
        path = path if path else self.config_path
        
        assert isinstance(config, dict)

        with open(path, 'w') as file:
            documents = yaml.dump(config, file)
        
        return config

    
    def get_config(self, input, key_path, local_key_path=[]):
        from commune.utils.dict import dict_get

        """

        :param
            input: input string (str)
        :return:
             Regex Match
                - path of config within match
             No Regex Match
                - None (this means its not pointing to a config path)
        """

        config=input

        if isinstance(config, str):
            config_path = re.compile('^(get_config)\((.+)\)').search(input)
            # if there are any matches ()
            if config_path:
                config_path = config_path.group(2)
                config_keys =  None
                if ',' in config_path:
                    assert len(config_path.split(',')) == 2
                    config_path ,config_keys = config_path.split(',')

                config = self.load_config(config_path)
                config = self.resolve_config(config=config,root_key_path=key_path, local_key_path=key_path)

                if config_keys != None:

                    config =  dict_get(input_dict=config, keys=config_keys)

        return config

    def set_cache(self, key, value):
        self.cache[key] = value
    
    def get_cache(self, key):
        return self.cache[key]
          

    def local_copy(self, input, key_path):
        """

        :param
            input: input string (str)
        :return:
             Regex Match
                - path of config within match
             No Regex Match
                - None (this means its not pointing to a config path)
        """

        variable_object = input
        if isinstance(input, str):

            variable_path = None
            if '::' in input:
                assert len(input.split('::')) == 2
                function_name, variable_path = input.split('::')
            else:
                variable_path = re.compile('^(local_copy)\((.+)\)').search(input)
                if variable_path:
                    variable_path = variable_path.group(2)
            
            if variable_path:

                # get the object
                local_config_key_path = self.cache[list2str(key_path)]
                
                if local_config_key_path:
                    local_config = dict_get(input_dict=self.config, keys=self.cache[list2str(key_path)])
                else: 
                    local_config = self.config
                variable_object = dict_get(input_dict=local_config,
                                                    keys = variable_path)

        return variable_object


    def copy(self, input, key_path):
        """

        :param
            input: input string (str)
        :return:
             Regex Match
                - path of config within match
             No Regex Match
                - None (this means its not pointing to a config path)
        """

        variable_object = input


        if isinstance(input, str):

            variable_path = re.compile('^(copy)\((.+)\)').search(input)

            if variable_path:
                variable_path = variable_path.group(2)

                # get the object
                try:
                    variable_object = dict_get(input_dict=self.config,
                                                        keys = variable_path)
                except KeyError as e:
                    raise(e)

        
        return variable_object


    def load_config(self,
                     path=None,
                     tag='!ENV'):

        if type(path) in [dict, list]:
            return path
        assert isinstance(path, str), path
        """
        Load a yaml configuration file and resolve any environment variables
        The environment variables must have !ENV before them and be in this format
        to be parsed: ${VAR_NAME}.
        E.g.:
            client:
                host: !ENV ${HOST}
                port: !ENV ${PORT}
            app:
                log_path: !ENV '/var/${LOG_PATH}'
                something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'

        :param
            str path: the path to the yaml file
            str tag: the tag to look for

        :return
            dict the dict configuration
        """
        # pattern for global vars: look for ${word}

        with open(path) as conf_data:
            config =  yaml.load(conf_data, Loader=yaml.SafeLoader)
        
        return config
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return "\n" + yaml.dump(self.toDict())

    def to_string(self, items) -> str:
        """ Get string from items
        """
        return "\n" + yaml.dump(items.toDict())

    def update_with_kwargs( self, kwargs ):
        """ Add config to self
        """
        for key,val in kwargs.items():
            self[key] = val

    @classmethod
    def default_dict_fns(cls):
        
        from commune.config.utils import  dict_fn_local_copy, dict_fn_get_config 

        default_dict_fns = {
            'local_copy': dict_fn_local_copy,
            'get_config': dict_fn_get_config
        }
        return default_dict_fns

    def dict_fn(cls, 
                fn:Callable,
                input: Dict, 
                context:dict=None, 
                function_seperator: str='::', 
                default_dict_fns: Dict=None):
        '''
        Apply a function to a dictionary based ont he function seperator: ::
        '''
        
        from copy import deepcopy
        default_dict_fns = cls.default_dict_fns() if default_dict_fns == None else default_dict_fns
        recursive_types = [dict, list, set, tuple]
        
        # get the keys of the input
        if type(input) in [dict]:
            # get the keys
            keys = list(input.keys())
        elif type(input) in [set, list, tuple]:
            
            # the keys are the index of the list
            keys = list(range(len(input)))
            
            # Convert the set,tuple into a list
            if type(input) in [set, tuple]:
                input = list(input)
        
        for key in keys:
            if isinstance(input[key], str):
                # if the string is sperated by the function and results in 2 strings
                assert len(input[key].split(function_seperator)) == 2, \
                        f'input value must be a string with the format: function_name::input_arg, but you have {input[key]}'
                trigger_function = function_seperator in input[key]
                
                if trigger_function: 
                    function_key, input_arg =  input[key].split(function_seperator)
                    input[key] = default_dict_fns[function_key](input_arg, context=context)
            
            if type(input[key]) in [dict, list, tuple, set]:
                # functions do not apply over dictionaries
                input[key] = dict_fn(fn=fn, 
                                    input=input[key], 
                                    context=context,
                                    function_seperator=function_seperator,
                                    default_dict_fns=default_dict_fns)

    
        return input


    def override(self, override={}, config=None):
        from commune.utils.dict import dict_put
        """
        
        """
        if config == None:
            config = self.config

        for k,v in override.items():
            dict_put(input_dict=config,keys=k, value=v)

        return config
    

