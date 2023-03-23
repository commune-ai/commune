

import commune
import streamlit as st

bro = commune.key(seed_hex = 'whadup')

st.write(bro.__dict__)

st.write(bro)