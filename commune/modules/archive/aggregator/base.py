
from munch import Munch
import commune

import os
import ray

# pipeline_config = commune.load_config(os.path.dirname(__file__).replace(os.getenv('PWD'), ''))
import torch 

class BaseAggregator:
    def __init__(self, blocks:list=[], config={}):
        self.config = Munch(config)
        self.build_blocks(blocks)

    
    def build_blocks(self,blocks):
        # building the pipeline
        self.blocks = []
        for block in blocks:
            block['name'] = block.get('name',  block['module'] )
            block['actor'] = block.get('actor',  False )
            launch_kwargs = dict(
                module = block['module'],
                fn = block.get('init_fn', None),
                actor =  block['actor']
            )
            block['module'] =  commune.launch(**launch_kwargs)
            block['function'] = getattr(block['module'], block['fn'])
            self.blocks.append(block)
        return self.blocks


    @staticmethod
    def run_block(block, input={}):
        fn = block.get('function')
        fn_args = block.get('args', [])
        fn_kwargs = block.get('kwargs', {})
        key_map = block.get('key_map', {})
        input = {key_map.get(k, k):v for k,v in input.items()}
        fn_kwargs = {**input, **fn_kwargs}
        output = fn(*fn_args, **fn_kwargs)      
        return output
        
    def get_outputs(self, *args,**kwargs):
        outputs = []
        for block in self.blocks:
            output = self.run_block(block)
            outputs.append(output)
        return outputs

    @staticmethod
    def aggregate_outputs(outputs):

        if any([isinstance(o, ray._raylet.ObjectRef) for o in outputs]):
            if all([isinstance(o, ray._raylet.ObjectRef) for o in outputs]):
                outputs = ray.get(outputs)
            else:
                outputs = [ray.get(o) if isinstance(o, ray._raylet.ObjectRef) else o  for o in outputs ]
        
        aggregate_outputs = {}

        for output in outputs:
            for k,v in output.items():
                if k in aggregate_outputs:
                    aggregate_outputs[k] += [v]
                else:
                    aggregate_outputs[k] = [v]
        return aggregate_outputs

    def __call__(self, blocks:list=[], *args, **kwargs):
        if len(blocks)>0:
            self.blocks = self.build_blocks(blocks)
        outputs = self.get_outputs(*args , **kwargs)
        aggregate_outputs = self.aggregate_outputs(outputs)
        

        # stack outputs in 1st dimension
        outputs = {k:torch.stack(v) for k,v in aggregate_outputs.items()}
        outputs = {k: torch.mean(v, dim=0) for k,v in outputs.items()}
        return outputs

    @staticmethod
    def test_sequential_pipeline():
        import streamlit as st
        commune.init_ray()
        blocks = [
         {
            'module': 'model.hf',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'kwargs': {'ray_get': True},
        } for i in range(3)] 

        aggregator = BaseAggregator()
        st.write(aggregator(blocks=blocks))

if __name__ == '__main__':
    commune.Aggregator.test_sequential_pipeline()
    # st.write(commune.list_actor_names())



        
        

