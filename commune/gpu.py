    
    ### GPU LAND        import torch
import torch
import commune as c
from typing import *

class GPU(c.Module):
    def gpus(self) -> List[int]:
        available_gpus = [int(i) for i in range(torch.cuda.device_count())]
        return available_gpus
    
    def num_gpus(self):
        return len(self.gpus())
    
    def cuda_available(self) -> bool:
        return torch.cuda.is_available()
    
    
    def gpu_info_map(self, device:int = None, fmt='gb') -> Dict[int, Dict[str, float]]:
        gpu_info = {}
        for gpu_id in self.gpus():
            gpu_id = int(gpu_id)
            mem_info = torch.cuda.mem_get_info(gpu_id)
            gpu_info[gpu_id] = {
                'name': torch.cuda.get_device_name(gpu_id),
                'free': mem_info[0],
                'used': (mem_info[1]- mem_info[0]),
                'total': mem_info[1], 
                'ratio': mem_info[0]/mem_info[1],
            }
            if fmt != None:
                keys = ['free', 'used', 'total']
                for k in keys:
                    gpu_info[gpu_id][k] = c.format_data_size(gpu_info[gpu_id][k], fmt=fmt)
        if device != None:
            return gpu_info[device]

        return gpu_info

    def gpu_total_map(self) -> Dict[int, Dict[str, float]]:
        return {k:v['total'] for k,v in self.gpu_info().items()}

 
    def total_gpu_memory(self) -> int:
        total_gpu_memory = 0
        for gpu_id, gpu_info in self.gpu_info_map().items():
            total_gpu_memory += gpu_info['total']
        return total_gpu_memory
    

    def used_gpu_memory(self) -> int:
        used_gpu_memory = 0
        for gpu_id, gpu_info in self.gpu_info_map().items():
            used_gpu_memory += gpu_info['used'] 
        return used_gpu_memory

    def gpu_info(self, device:int = None) -> Dict[str, Union[int, float]]:
        '''
        Get the gpu info for a given device
        '''
        if device is None:
            device = 0
        gpu_map = self.gpu_info_map()
        if device in gpu_map:
            return gpu_map[device]
        else:
            return gpu_map
        

        

    @classmethod
    def most_free_gpu(cls, 
                      free_gpu_memory:dict = None,
                      mode : bool = 'int',
                      **kwargs) -> Union[int, Dict[str, int]]:
        """ Returns a dictionary of gpu_id to max memory for each gpu.
        Args:
            total_memory (int, optional): Total memory to allocate. Defaults to None.
            buffer_memory (int, optional): Buffer memory to leave on each gpu. Defaults to 10.
        
        Returns 
            Dict[int, str]: Dictionary of gpu_id to max memory for each gpu.
        """
        if free_gpu_memory is None:
            free_gpu_memory = cls.free_gpu_memory(**kwargs)
        assert isinstance(free_gpu_memory, dict), f'free_gpu_memory must be a dict, not {type(free_gpu_memory)}'
        most_available_gpu_tuples = sorted(free_gpu_memory.items(), key=lambda x: x[1] , reverse=True)
        if mode == 'tuple':
            return most_available_gpu_tuples[0]
        elif mode == 'dict': 
            return {most_available_gpu_tuples[0][0]: most_available_gpu_tuples[0][1]}
        elif mode == 'int':
            return most_available_gpu_tuples[0][0]
        elif mode == 'str':
            return str(most_available_gpu_tuples[0][0])
        else:
            raise ValueError(f'Invalid mode {mode}')
    
    
    @classmethod
    def most_free_gpus(cls, 
                       n:int=None,
                      free_gpu_memory:dict = None,
                      mode : str = 'dict',
                      fmt:str='b',
                      **kwargs) -> Union[int, Dict[str, int]]:
        """ Returns a dictionary of gpu_id to max memory for each gpu.
        Args:
            total_memory (int, optional): Total memory to allocate. Defaults to None.
            buffer_memory (int, optional): Buffer memory to leave on each gpu. Defaults to 10.
        
        Returns 
            Dict[int, str]: Dictionary of gpu_id to max memory for each gpu.
        """
 
        if free_gpu_memory is None:
            free_gpu_memory = cls.free_gpu_memory(**kwargs)
        assert isinstance(free_gpu_memory, dict), f'free_gpu_memory must be a dict, not {type(free_gpu_memory)}'
        most_available_gpu_tuples = sorted(free_gpu_memory.items(), key=lambda x: x[1] , reverse=True)

        if n == None:
            n = len(most_available_gpu_tuples)
        if mode == 'dict': 
            return {most_available_gpu_tuples[i][0]: c.format_data_size(most_available_gpu_tuples[i][1], fmt=fmt) for i in range(n)}
        elif mode == 'tuple':
            return [(i,c.format_data_size(most_available_gpu_tuples[i][0], fmt=fmt)) for i in range(n)]
        else:
            return [c.format_data_size(most_available_gpu_tuples[i][0], fmt=fmt) for i in range(n)]
        
    
    @classmethod
    def most_free_gpu_memory(cls, *args, **kwargs) -> int:
        gpu_id = cls.most_free_gpu()
        return cls.free_gpu_memory(*args, **kwargs)[gpu_id]
    




    @classmethod
    def resolve_device(cls, device:str = None, verbose:bool=True, find_least_used:bool = True) -> str:
        
        '''
        Resolves the device that is used the least to avoid memory overflow.
        '''
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            assert torch.cuda.is_available(), 'Cuda is not available'
            gpu_id = 0
            if find_least_used:
                gpu_id = cls.most_free_gpu()
                
            device = f'cuda:{gpu_id}'
        
            if verbose:
                device_info = cls.gpu_info(gpu_id)
                c.print(f'Using device: {device} with {device_info["free"]} GB free memory', color='yellow')
        return device  
    
    