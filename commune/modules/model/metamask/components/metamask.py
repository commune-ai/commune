import streamlit as st
from wallet_connect import wallet_connect

def metamask_connection():
    # Connect to MetaMask wallet
    connect_button = wallet_connect(label="wallet", key="wallet")
    if connect_button:
        st.session_state['address'] = connect_button
        return True
    return False