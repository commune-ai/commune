import commune
# import streamlit as st
# import torch
# commune.launch(module='dataset.text.huggingface',  name='dataset')
# commune.get_module('model.transformer').serve_module()
# commune.get_module('dataset.text.huggingface').deploy_fleet()
# print(module.config)
# auth = module.get_auth()
# print(module.verify(auth))

# commune.restart('model.gpt125m')
module = commune.Module()

obj = module.get_auth(commune.connect('model.gpt125m').get_auth())
print(obj)
# print(commune.peer_registry())

# python3 commune/bittensor/bittensor_module.py -fn register_wallet -kwargs "{'dev_id': 1, 'wallet': 'ensemble_0.1'}"