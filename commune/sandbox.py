

import commune
import streamlit as st

key = commune.key('whadup')

st.write(key.params)
st.write(commune.key(**key.params))
