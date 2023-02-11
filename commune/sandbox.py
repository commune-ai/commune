import commune
import streamlit as st

servers = commune.servers()

print(commune.launch('model.dendrite', tag='B' ))