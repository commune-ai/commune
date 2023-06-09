

import commune
commune.new_event_loop()
import bittensor
import streamlit as st
print(bittensor.axon(netuid=3, port=8902))
print(bittensor.subtensor())

# config = bittensor.neurons.core_server.neuron.config()
# config.neuron.no_set_weights = True
# config.neuron.no_autosave = True
# print(bittensor.neurons.core_server.neuron.config())


import commune
import streamlit as st
st.set_page_config(layout="wide")

@st.cache_data(persist=True)
def load_model_configs(model_prefix):
    commune.new_event_loop()
    models = commune.modules(model_prefix)
    model_configs = commune.call_pool(models, 'config')
    return model_configs

model_prefix = st.text_input('search for a model...','model.gptj')
model_configs = load_model_configs(model_prefix)
models = list(model_configs.keys())
import pandas as pd

models_df = pd.DataFrame([dict(model=model, **config.stats) for model, config in model_configs.items()])
st.write(models_df)



