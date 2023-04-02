import streamlit as st
import commune

servers = commune.servers()
print(commune.connect('162.157.13.236:50074').peer_info())