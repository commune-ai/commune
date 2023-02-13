import commune
commune.new_event_loop()
# import streamlit as st

# servers = commune.servers()

# print(commune.launch('model.dendrite', tag='B' ))
import bittensor

import torch



# commune.launch('model.transformer', name='model', tag='gptj', fn='serve_module', device=6)
# commune.launch('model.transformer', name='model', tag='gptjt', fn='serve_module', device=0)


# commune.launch('model.transformer', name='train::gptj', fn='local_train', mode='pm2', 
#                device='3', kwargs={'model': 'gptj'} ,refresh=True)