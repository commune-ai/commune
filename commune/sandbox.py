import commune 
import streamlit as st


st.write([commune.connect(m).dataset_byte_size_map for m in commune.servers('dataset')])
# # print(commune.get_module('model.transformer').launch(name='model::gpt125m', tag='1'))
# modules =  commune.pm2_list()
# selected_modules = []
# selected_modules = st.multiselect('Select a model', modules, selected_modules)

# refresh_button = st.button('Refresh')
# delete_button = st.button('Delete')

# if delete_button:
#     for module in selected_modules:
#         commune.pm2_kill(name=module)
#     selected_modules = []
# if refresh_button:
#     for module in selected_modules:
#         commune.pm2_refresh(name=module)
#     selected_modules = []

# st.write(selected_modules)


# st.write(commune.get_module('model.transformer').glob("**"))

# # print(commune.cmd('pm2 logs model::gptj::0'))