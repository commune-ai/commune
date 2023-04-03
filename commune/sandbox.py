import streamlit as st
import commune

servers = commune.servers()
key = commune.key()
print(commune.key().encrypt('hello'))
# print(commune.module().__str__() == 'Module')