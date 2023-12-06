import commune as c
import streamlit as st
from components.metamask import metamask_connection
from components.transaction import display_transaction_ui
from utils import format_eth_address

class ModelMetamask(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call():
        main()
        
def main():
    st.title('Ethereum Transaction App with MetaMask')

    # MetaMask Connection
    if metamask_connection():
        address = st.session_state.get('address')
        st.write(f"Connected Ethereum Address: {format_eth_address(address)}")

        # Display Transaction UI
        display_transaction_ui()