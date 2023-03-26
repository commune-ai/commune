import commune
import streamlit as st
import torch

st.write(commune.connect('openai').forward(prompt='How do yo'))
# module = commune.connect('openai')
