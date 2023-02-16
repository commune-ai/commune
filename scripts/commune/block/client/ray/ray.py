
import os
import sys
import json
sys.path.append(os.getcwd())
from commune import Module
import ray
import requests


class RayModule(Module):
    def __init__(
        self,
        config=None,
        **kwargs
    ):
        Module.__init__(self, config=config,  **kwargs)
        self.queue = self.launch(**self.config['servers']['queue'])
        self.object  = self.launch(**self.config['servers']['object'])



    # def load_clients(self):
    #     # load object server

    #     self.

if __name__ == '__main__':
    import streamlit as st
    module = RayModule.deploy(actor={'refresh': False, 'wrap': True})
    st.write(module.actor._ray_method_signatures['__init__'][0])




