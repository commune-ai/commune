

import commune
import streamlit as st

bro = commune.get_module('web3.account.substrate')(seed='whadup')

st.write(bro.__dict__)

st.write(bro)