import commune as c
<<<<<<< HEAD
import torch

# meta = c.module('bittensor').get_metagraph(subtensor='local')



top_uid_map = c.print(c.module('bittensor').get_top_uids())

for uid, incentive in top_uid_map.items():
    print(uid, incentive)
=======
import bittensor as bt
import streamlit as st

c.print(bt)
# # list keys

# api_key = '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'
# model = c.module('model.openai')(api=api_key)
# c.print(model)
# import json

# c.print(c.schema().keys())
# c.print(json.loads(model.forward(text='What is the best form of python?')))
# # add key
>>>>>>> origin/newton

