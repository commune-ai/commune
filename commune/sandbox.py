import commune
# import streamlit as st
# import torch
# commune.launch(module='dataset.text.huggingface',  name='dataset')
# commune.get_module('model.transformer').serve_module()
# commune.get_module('dataset.text.huggingface').deploy_fleet()

print(commune.Module().get_auth())
# print(commune.peer_registry())