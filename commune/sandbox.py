import commune
import streamlit as st
import torch

# commune.get_module('dataset.text.huggingface').deploy_fleet()
print(commune.server_registry())
# st.write(commune.connect('miner.1').get_auth())