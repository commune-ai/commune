from __future__ import annotations
from munch import Munch
import commune
from typing import List, Dict, Union 
import os

class PipelineModule(commune.Module):
    def __init__(self, modules:List[str]):
        self.build_pipeline(modules)
        
    def build_pipeline(self, modules:List[Union[str, Dict]], virtual:bool = True, default_call_fn:str = 'forward'):
        
        self.pipeline_blocks = []
        for module in modules:
            if isinstance(module, dict):
                assert 'module' in module
                
                # get the module
                get_module_kwargs = module['module'] \
                                    if isinstance(module['module'], dict) else \
                                        {'module': module['module']}
                                        
                module_obj = commune(**get_module_kwargs, virtual=True)
                
                # get the function
                module_fn_name = module.get('fn', default_call_fn)

                module_fn = getattr(module_obj, module_fn_name)
                
                module_kwargs =   module.get('kwargs', {})
                
                # map the inputs from the input node to the output node
                module_input_map =  module.get('input_map', {})
                
                block = {
                    'fn': module_fn,
                    'input_map':module_input_map,
                    'kwargs': module_kwargs
                }
                self.pipeline_blocks.append(block)
                    
        return self.pipeline_blocks

    def forward(self, **kwargs):
        from copy import deepcopy
        for block in self.pipeline_blocks:
            kwargs = deepcopy(kwargs)
            block_kwargs = block['kwargs']
            
            # add the kwargs from the block into the original kwargs
            for k,v in block_kwargs:
                assert k not in kwargs
                kwargs[k] = v
            
            # map the input kwargs from k-> v
            if len(block['input_map']) > 0:
                kwargs = {block['input_map'].get(k, k):v for k,v in kwargs.items()}
            
            output = block['function'](**kwargs)
            
            assert isinstance(output, dict), f'lets keep things simple and use an output dictionary'

            # maps the key of the output to the input of the next block in case there is a conflict
            kwargs = {block['output_map'].get(k, k):v for k,v in output.items()}
            

            
        return kwargs

    @staticmethod
    def test_sequential_pipeline():
        blocks = [
        {
            'module': 'dataset.text.huggingface',
            'fn': 'sample',
            'kwargs': {'tokenize': False},
         }
         ]

        pipeline = Pipeline(pipeline_blocks)
        st.write(pipeline.run())


if __name__ == '__main__':
    import streamlit as st

    # st.write(commune.Module.get_module_python_paths())
    # st.write(commune.Module.simple2import('commune.sandbox.paper'))

    Pipeline.test_sequential_pipeline()


        
        

