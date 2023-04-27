
import bittensor
print(bittensor.axon())

# import commune
# import streamlit as st

# @st.cache_data(persist=True)
# def load_model_configs(model_prefix):
#     commune.new_event_loop()
#     models = commune.modules(model_prefix)
#     model_configs = commune.call_pool(models, 'config')
#     return model_configs

# model_prefix = 'model.gptj'
# model_configs = load_model_configs(model_prefix)
# models = list(model_configs.keys())

# with st.sidebar:
#     models = st.multiselect('Models', models, models)


# for model, config in model_configs.items():
#     with st.expander(model,True):
#         config = commune.munch2dict(config)
#         stats = config.pop('stats', None)
#         epoch_loss_history = stats.pop('epoch_loss_history', None)
#         st.write(epoch_loss_history)
#         config = commune.flatten_dict(config)
#         for metric, value in stats.items(): 
#             if type(value) in [float, int, str]:
#                 st.metric(label=metric, value=value)
#             else:
#                 st.write(f'{metric}')
#                 st.write(value)
