import commune
import streamlit as st
module = commune.Module()
module.set_key()
st.write(module.key.sign('broadcast'))
st.write(module.verify(module.key.sign('broadcast')))
