import commune
import streamlit as st
import torch

module = commune.Module()

class Demo(commune.Module):
    hello: 'world'
    
        
alice = Demo()
alice.set_key('alice')
alice.set_key('bob')
bob = Demo()
bob.set_key('bob')
bob.set_key('alice')

auth = bob.get_auth('whadup')
st.write(auth)
st.write(bob.verify(auth))
