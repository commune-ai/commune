import streamlit as st
import commune

servers = commune.servers()
print(commune.key())
# print(commune.module().__str__() == 'Module')