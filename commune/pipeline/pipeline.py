from __future__ import annotations
from munch import Munch
import commune
import streamlit as st
import os

# pipeline_config = commune.load_config(os.path.dirname(__file__).replace(os.getenv('PWD'), ''))




class Pipeline:
    def __init__(self, pipeline, config={}):
        self.config = Munch(config)
        self.process_block = Munch({})
        self.pipeline = pipeline if pipeline != None else self.config.pipeline
        self.build_pipeline(self.pipeline)


    def build_pipeline(self, pipeline_config):
        if isinstance(pipeline_config, list):
            keys = list(range(len(pipeline_config)))
        elif isinstance(pipeline_config, dict): 
            keys = list(pipeline_config.keys())
        
        previous_key = None
        # building the pipeline
        self.pipeline_blocks = []
        for key in keys:
            process_block = pipeline_config[key]
            path = process_block['module']

            process_block['tag'] = process_block.get('tag', None)
            process_block['name'] = process_block.get('name',  path )
            process_block['actor'] = process_block.get('actor',  False )
            launch_kwargs = dict(
                module = process_block['module'],
                fn = process_block.get('init_fn', None),
                kwargs = process_block.get('init_kwargs', {}),
                actor =  process_block['actor']
            )

            module_block = commune.launch(**launch_kwargs)
            process_block['module'] = module_block
            process_block['function'] = getattr(module_block, process_block.get('fn', process_block.get('function', '__call__' )))

            self.process_block[process_block['name']] = process_block

            if previous_key != None:
                input_modules = self.pipeline_blocks[previous_key]
                if not isinstance(input_modules, list):
                    input_modules = [input_modules]
                process_block['input_modules'] = list(map(lambda x: x['name'], input_modules ))

            previous_key = key
            self.pipeline_blocks.append(process_block)
            

    def run(self):
        input = {}
        for block in self.pipeline_blocks:
            fn = block.get('function')
            fn_args = block.get('args', [])
            fn_kwargs = block.get('kwargs', {})
            input_key_map = block.get('input_key_map', {})
            if isinstance(input, dict):
                input = {input_key_map.get(k, k):v for k,v in input.items()}
                fn_kwargs.update(input)
            else:
                fn_args = [input, *fn_args]

            output = fn(*fn_args, **fn_kwargs)
            output_key_map = block.get('output_key_map', {})
            if isinstance(output, dict):
                output = {output_key_map.get(k, k):v for k,v in output.items()}
            input = output

        return output

    @staticmethod
    def test_sequential_pipeline():
        commune.init_ray()
        pipeline_blocks = [
        {
            'module': 'commune.dataset.text.huggingface',
            'fn': 'sample',
            'kwargs': {'tokenize': False},
            'output_key_map': {'text': 'input'}
         }, 
         {
            'module': 'datasets.load_dataset',
            'fn': 'forward',
            'output_key_map': {'input': 'text'}
        }
         ]

        pipeline = Pipeline(pipeline_blocks)
        st.write(pipeline.run())


    @staticmethod
    def test_aggregator_pipeline():
        commune.init_ray()
        pipeline_blocks = [
        {
            'module': 'commune.dataset.text.huggingface',
            'fn': 'sample',
            'kwargs': {'tokenize': False},
         }, 
         
         {
            'module': 'commune.Aggregator',
            'kwargs': {'blocks': [
                                {
                                    'module': 'commune.model.transformer',
                                    'actor': {'gpus': 0.1, 'tag': f'{i}', 'wrap': True},
                                    'fn': 'forward',
                                    'kwargs': {'ray_get': True},
                                } for i in range(3)] },
        }]
        

        pipeline = Pipeline(pipeline_blocks)
    
    @staticmethod
    def dummy(a:dict) -> dict:
        return a


def get_annotations(fn:callable):
    return fn.__annotations__


if __name__ == '__main__':

    # st.write(commune.Module.get_module_python_paths())
    # st.write(commune.Module.simple2import('commune.sandbox.paper'))

    Pipeline.test_sequential_pipeline()
    Pipeline.test_aggregator_pipeline()


        
        

