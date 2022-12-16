import os
import sys
import re
import yaml
import torch
import argparse
from commune.utils import (load_yaml,
                        round_sig,
                        string_replace)

from copy import deepcopy

def parse_config(path=None, tag='!ENV'):
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
    pattern = re.compile('.*?\${(\w+)}.*?')
    loader = yaml.SafeLoader

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)
    with open(path) as conf_data:
        return yaml.load(conf_data, Loader=loader)

def resolve_devices(cfg, device):
    '''
    :param cfg: dictionary from yaml file
    :param device: string (cuda or cpu supported for now (TODO multi-gpu support))
    :return:
        cfg: dictionary from yaml file
    '''
    # Resolve Devices with Config
    if (not torch.cuda.is_available()) and \
            ('cuda' in device) :
        cfg = string_replace(cfg=cfg,
                                          old_str=device,
                                          new_str="cpu")
    return cfg




def dict_get(input_dict, keys):
    tmp_dict = deepcopy(input_dict)
    for key in keys:
        if key  in tmp_dict: 
            tmp_dict = tmp_dict[key]
    return tmp_dict




def dict_put(input_dict, keys, value, raise_error=True):
    tmp_dict = deepcopy(input_dict)
    if len(keys) == 0 :
        raise Exception(f'TOO DEEP: keys not suppose to be {len(keys)}')
    elif len(keys) == 1:
        return value
    elif len(keys)>1:
        for key in keys:
            if key in tmp_dict:         
                tmp_dict[key] = dict_put(input_dict=tmp_dict[key], keys=keys[:1])
        return tmp_dict


def dict_fn_local_copy(input,context={}):
    keys = input.split('.')
    dict_get(input_dict=context, keys=keys)


def dict_fn_get_config(input,context={}):
    keys = input.split('.')
    dict_get(input_dict=context, keys=keys)




def dict_fn_ray_get(input:str, context={}):
    
    if len(input.split('::')) == 1:
        input = input
    elif len(input.split('::')) == 2:
        namespace, actor_name = input.split('::')
    else:
        raise NotImplemented(input)

    ray.get_actor()