import streamlit as st
from wallet_connect import wallet_connect

def metamask_connection():
    """
    Connect to MetaMask wallet and retrieve the wallet address.
    Returns True if the connection is successful, False otherwise.
    """
    connect_button = wallet_connect(label="wallet1", key="wallet1")
    if connect_button:
        st.session_state['address'] = connect_button
        st.success(f"Connected to MetaMask wallet at address: {connect_button}")
        return True
    else:
        st.warning("Please connect to your MetaMask wallet.")
        return False

def display_wallet_info():
    """
    Display connected wallet information.
    """
    if 'address' in st.session_state:
        st.write(f"Connected Wallet Address: {st.session_state['address']}")
    else:
        st.error("No wallet connected.")