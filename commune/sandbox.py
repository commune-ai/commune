import commune
# import streamlit as st
# import torch
# commune.launch(module='dataset.text.huggingface',  name='dataset')
# commune.get_module('model.transformer').serve_module()
# commune.get_module('dataset.text.huggingface').deploy_fleet()
module = commune.Module()
auth = module.get_auth()
print(auth)
print(module.verify(auth))
print(module.key.public_key.hex())
print(module.key.ss58_address)
print(commune.key('bro').ss58_address)

# print(commune.peer_registry())