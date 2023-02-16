import streamlit as st
import fsspec
from fsspec.implementations.local import LocalFileSystem
from copy import deepcopy
import json
import os
from typing import *
import pandas as pd
import pickle
LocalFileSystem.root_market = '/'





class LocalModule(LocalFileSystem):
    default_cfg = {
    }
    def __init__(self, config=None):
        LocalFileSystem.__init__(self)
        self.config= self.resolve_config(config)
    def ensure_path(self, path):
        """
        ensures a dir_path exists, otherwise, it will create it 
        """
        file_extension = self.get_file_extension(path)
        if os.path.isfile(path):
            dir_path = os.path.dirname(path)
        elif os.path.isdir(path):
            dir_path = path
        elif len(file_extension)>0:
            dir_path = os.path.dirname(path)
        else:
            dir_path = os.path.dirname(path)

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def get_file_extension(path):
        return os.path.splitext(path)[1].replace('.', '')

    extension2mode = {
        'pkl':'pickle',
        'pickle':'pickle',
        'json': 'json',
        'csv': 'csv',
        'yaml': 'yaml',
        'pth': 'torch.state_dict',
        'onnx': 'onnx'
    }

    supported_modes = ['pickle', 'json']


    def resolve_mode_from_path(self, path):
        mode = self.extension2mode[self.get_file_extension(path)]
        assert mode in self.supported_modes
        return mode  

    def put_json(self, path, data):
            # Directly from dictionary
        self.ensure_path(path)
        data_type = type(data)
        if data_type in [dict, list, tuple, set, float, str, int]:
            with open(path, 'w') as outfile:
                json.dump(data, outfile)
        elif data_type in [pd.DataFrame]:
            with open(path, 'w') as outfile:
                data.to_json(outfile)
        else:
            raise NotImplementedError(f"{data_type}, is not supported")


    def get_json(self, path, handle_error = False, return_type='dict', **kwargs):
        try:
            data = json.loads(self.cat(path))
        except FileNotFoundError as e:
            if handle_error:
                return None
            else:
                raise e

        if return_type in ['dict', 'json']:
            data = data
        elif return_type in ['pandas', 'pd']:
            data = pd.DataFrame(data)
        elif return_type in ['torch']:
            torch.tensor
        return data

    def put_pickle(self, path:str, data):
        with self.open(path,'wb') as f:
            pickle.dump(data, f, protocol= pickle.HIGHEST_PROTOCOL)
            
    def get_pickle(self, path, handle_error = False):
        try:
            with self.open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            if handle_error:
                return None
            else:
                raise e

    def put_object(self, path:str, data:Any, mode:str=None,**kwargs):
        if mode == None:
            mode = self.resolve_mode_from_path(path)
        return getattr(self, f'put_{mode}')(path=path,data=data, **kwargs)

    def get_object(self, path:str, mode:str=None, **kwargs):
        if mode == None:
            mode = self.resolve_mode_from_path(path)
        return getattr(self, f'get_{mode}')(path=path, **kwargs)

    @staticmethod
    def funcs(module, return_dict=True):
        fn_list = dir(module)

        final_fns = []
        if return_dict:
            final_fns = {}

        for fn in fn_list:
            if not (fn.startswith('__') and fn.endswith('__')) and not fn.startswith('_'):
                
                fn_object = getattr(module, fn)
                if callable(fn_object):
                    if return_dict:
                        final_fns[fn] = fn_object
                    else:
                        final_fns.append(fn)

        return final_fns

    def resolve_config(self,config):
        if config == None:
            config = self.default_cfg
        else:
            assert isinstance(config, dict)
        
        return config

    @classmethod
    def test_json(cls):
        self = cls()
        obj = {'bro': 1}
        dummy_path = '/tmp/commune/bro.json'
        self.put_json(path=dummy_path, data=obj)
        loaded_obj = self.get_json(path=dummy_path)
        assert json.dumps(loaded_obj) == json.dumps(obj)
        return loaded_obj
    @classmethod
    def test_pickle(cls):
        self = cls()
        obj = {'bro': 1}
        dummy_path = '/tmp/commune/bro.json'
        self.put_pickle(path=dummy_path, data=obj)
        loaded_obj = self.get_pickle(path=dummy_path)
        assert json.dumps(loaded_obj) == json.dumps(obj)
        return loaded_obj

    @classmethod
    def test(cls):
        import streamlit as st
        for attr in dir(cls):
            if attr[:len('test_')] == 'test_':
                getattr(cls, attr)()
                st.write('PASSED',attr)

    def put_json(self, path, data):
            # Directly from dictionary
        self.ensure_path(path)
        data_type = type(data)
        if data_type in [dict, list, tuple, set, float, str, int]:
            with open(path, 'w') as outfile:
                json.dump(data, outfile)
        elif data_type in [pd.DataFrame]:
            with open(path, 'w') as outfile:
                data.to_json(outfile)
        else:
            raise NotImplementedError(f"{data_type}, is not supported")

    # async stuff
    async def async_read(path, mode='r'):
        async with aiofiles.open(path, mode=mode) as f:
            data = await f.read()
        return data
    async def async_write(path, data,  mode ='w'):
        async with aiofiles.open(path, mode=mode) as f:
            await f.write(data)

    async def async_get_json(path, return_type='dict'):
        try:  
            
            data = json.loads(await async_read(path))
        except FileNotFoundError as e:
            if handle_error:
                return None
            else:
                raise e

        if return_type in ['dict', 'json']:
            data = data
        elif return_type in ['pandas', 'pd']:
            data = pd.DataFrame(data)
        elif return_type in ['torch']:
            torch.tensor
        return data

    async def async_put_json( path, data):
            # Directly from dictionary
        data_type = type(data)
        if data_type in [dict, list, tuple, set, float, str, int]:
            json_str = json.dumps(data)
        elif data_type in [pd.DataFrame]:
            json_str = json.dumps(data.to_dict())
        else:
            raise NotImplementedError(f"{data_type}, is not supported")
        
        return await async_write(path, json_str)


if __name__ == '__main__':
    import commune
    
    import asyncio
    path = '/tmp/asyncio.txt'
    data = {'bro': [1,2,4,5,5]*100}
    with commune.timer() as t:
        async_put_json(path, data)
        asyncio.run(async_get_json(path))
        st.write(sys.getsizeof(data)/t.seconds)


    # module = LocalModule()
    # st.write(module.test())
