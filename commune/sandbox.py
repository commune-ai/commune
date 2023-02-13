import commune
commune.new_event_loop()
# import streamlit as st

# servers = commune.servers()

# print(commune.launch('model.dendrite', tag='B' ))
import bittensor

import torch


# commune.launch('model.transformer', name='model', tag='gptj', fn='serve_module', device='2')
# commune.launch('model.transformer', name='model', tag='gptjt', fn='serve_module', device='0')
# commune.launch('model.transformer', name='model', tag='gpt2.7b', fn='serve_module', device='4')


commune.launch('model.transformer', name='train::gptj', fn='local_train', mode='pm2', 
               device='3', kwargs={'model': 'gptj'} ,refresh=True)

commune.launch('model.transformer', name='train::gptjt', fn='local_train', mode='pm2', 
               device='5', kwargs={'model': 'gptjt'} ,refresh=True)


commune.launch('model.transformer', name='train::gpt2.7b', fn='local_train', mode='pm2', 
               device='5', kwargs={'model': 'gpt2.7b'} ,refresh=True)