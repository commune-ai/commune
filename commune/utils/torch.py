from typing import Dict, List, Tuple, Union, Any, Optional

import torch


def tensor_info_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    tensor_info_dict  = {}
    for k,v in input_dict.items():
        if  isinstance(v,torch.Tensor):
            tensor_info_dict[k] = { 'shape': v.shape, 'dtype': v.dtype, 'device': v.device }
        
    return tensor_info_dict
        

def tensor_dict_shape(input_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple]:
    import torch
    out_dict = {}

    """should only have tensors/np.arrays in leafs"""
    for k,v in input_dict.items():
        if isinstance(v,dict):
            out_dict[k] = tensor_dict_shape(v)
        elif type(v) in [torch.Tensor, np.ndarray]:
            out_dict[k] = v.shape

    return out_dict


def check_distributions(kwargs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    return {k: {"mean": round(v.double().mean().item(), 2), "std": round(v.double().std().item(), 2)} for k, v in
            kwargs.items() if isinstance(v, torch.Tensor)}




def confuse_gradients(model):
    """

    :param model: model
    :return:
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = torch.randn(p.grad.data.shape).to(p.grad.data.device)




def nan_check(input, key_list=[], root_key=''):
    import torch, math
    if isinstance(input, dict):
        for k, v in input.items():

            new_root_key = '.'.join([root_key, k])
            if type(v) in [dict, list]:
                nan_check(input=v,
                                    key_list=key_list,
                                    root_key=new_root_key)
            else:
                if isinstance(v, torch.Tensor):
                    if any(torch.isnan(v)):
                        key_list.append(new_root_key)
                else:
                    if math.isnan(v):
                        key_list.append(new_root_key)
    elif isinstance(input, list):
        for k, v in enumerate(input):
            new_root_key = '.'.join([root_key, str(k)])
            if type(v) in [dict, list]:
                nan_check(input=v,
                                    key_list=key_list,
                                    root_key=new_root_key)
            else:
                if isinstance(v, torch.Tensor):
                    if any(torch.isnan(v)):
                        key_list.append(new_root_key)
                else:
                    if math.isnan(v):
                        key_list.append(new_root_key)
    return key_list


def seed_everything(seed: int) -> None:
    import os, torch, np
    "seeding function for reproducibility"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    


def confuse_gradients(model):
    import torch
    """

    :param model: model
    :return:
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = torch.randn(p.grad.data.shape).to(p.grad.data.device)


def get_device_memory():
    import nvidia_smi

    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    device_map  = {}
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        name = nvidia_smi.nvmlDeviceGetName(handle)
        device_map[name] = info.__dict__
        

    nvidia_smi.nvmlShutdown()
    
def tensor_dict_info(x:Dict[str, 'torch.Tensor']) -> Dict[str, int]:
    import torch
    output_dict = {}
    for k,v in x.items():
        if not isinstance(v, torch.Tensor):
            continue
        info = dict(
           shape=v.shape,
           dtype=str(v.dtype),
           device=str(v.device),
           requires_grad= v.requires_grad
        )

        output_dict[k] = info
    return output_dict
       
    
def param_keys(model:'nn.Module' = None)->List[str]:
    return list(model.state_dict().keys())

def params_map( model, fmt='b'):
    params_map = {}
    state_dict = model.state_dict()
    for k,v in state_dict.items():
        params_map[k] = {'shape': list(v.shape) ,
                            'size': cls.get_tensor_size(v, fmt=fmt),
                            'dtype': str(v.dtype),
                            'requires_grad': v.requires_grad,
                            'device': v.device,
                            'numel': v.numel(),  
                            }
    return params_map
