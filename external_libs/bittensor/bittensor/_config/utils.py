

from copy import deepcopy


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