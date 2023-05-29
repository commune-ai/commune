import commune as c
# import bittensor as bt
import streamlit as st

# list keys

api_key = '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'
model = c.module('model.openai')(api=api_key)
c.print(model)
import json
c.print(c.schema().keys())
# c.print(json.loads(model.forward(text='What is the best form of python?')))
# add key

