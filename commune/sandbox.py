import commune
# import streamlit as st

# servers = commune.servers()

# print(commune.launch('model.dendrite', tag='B' ))
import bittensor


commune.launch('model.transformer', name='train::gptj', fn='local_train', mode='pm2', 
               device='3', kwargs={'model': 'gptj'} ,refresh=True)