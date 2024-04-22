
from munch import Munch
import commune
import streamlit as st
import os
import ray


# pipeline_config = commune.load_config(os.path.dirname(__file__).replace(os.getenv('PWD'), ''))

import torch 
class MeanAggregator(commune.Aggregator):

    def run(self, *args, **kwargs):
        outputs = self.get_outputs(*args , **kwargs)
        aggregate_outputs = self.aggregate_outputs(outputs)
        # stack outputs in 1st dimension
        outputs = {k:torch.stack(v) for k,v in aggregate_outputs.items()}
        outputs = {k: torch.mean(v, dim=0) for k,v in outputs.items()}
        return outputs

    @classmethod
    def test_sequential_pipeline(cls):
        commune.init_ray()
        blocks = [
 
         {
            'module': 'model.hf',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'kwargs': {'ray_get': True},
        }, 
         {
            'module': 'model.hf',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'kwargs': {'ray_get': True},
        },
        {
            'module': 'model.hf',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'kwargs': {'ray_get': False},
        }
        ]

        aggregator = cls(blocks)
        st.write(aggregator.run())

if __name__ == '__main__':

    MeanAggregator.test_sequential_pipeline()



        
        

