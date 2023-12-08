import streamlit as st
from components.metamask import metamask_connection
from components.transaction import display_transaction_ui
from utils import format_eth_address

def main():
    st.title('Ethereum Transaction App with MetaMask')

    # MetaMask Connection
    if metamask_connection():
        address = st.session_state.get('address')
        st.write(f"Connected Ethereum Address: {format_eth_address(address)}")

        # Display Transaction UI
        display_transaction_ui()

if __name__ == "main":
    main()