import streamlit as st
from components.metamask import metamask_connection, display_wallet_info
from components.transaction import display_transaction_ui, display_trade_ui, display_history_ui

def main():
    st.title('Ethereum DApp with Uniswap Trading')
    if metamask_connection():
        display_wallet_info()

        display_transaction_ui()
        display_trade_ui()
        display_history_ui()
    else:
        st.info("Connect to MetaMask to use the application.")

if __name__ == "__main__":
    main()