import streamlit as st
import requests
import json

def display_transaction_ui():
    st.subheader("Send Ethereum Transaction")

    # Input fields for the transaction
    from_address = st.text_input("From Address")
    to_address = st.text_input("To Address")
    amount = st.number_input("Amount of Ether to Send", min_value=0.0, step=0.01, format="%.2f")
    private_key = st.text_input("Private Key", type="password")

    if st.button("Send Transaction"):
        transaction_data = {
            'from': from_address,
            'to': to_address,
            'value': amount,
            'private_key': private_key
        }

        # Backend API call to send transaction
        response = requests.post("http://localhost:5000/send_transaction", json=transaction_data)
        
        if response.status_code == 200:
            response_data = response.json()
            st.success(f"Transaction successful: {response_data.get('transaction_hash')}")
            st.json(response_data.get('receipt', {}))  # Display transaction receipt
        else:
            st.error("Failed to send transaction")

def display_trade_ui():
    st.subheader("Make a Trade on Uniswap")

    # Placeholder fields for Uniswap trade (adapt as necessary)
    token_address = st.text_input("Token Address")
    trade_amount = st.number_input("Trade Amount", min_value=0.0, step=0.01, format="%.2f")
    # Additional fields like token to swap to, slippage, etc., can be added

    if st.button("Make Trade"):
        trade_data = {
            'token_address': token_address,
            'amount': trade_amount
            # Add other necessary fields
        }

        # Backend API call to make trade
        response = requests.post("http://localhost:5000/make_trade", json=trade_data)
        
        if response.status_code == 200:
            st.success("Trade executed successfully")
        else:
            st.error("Failed to execute trade")

def display_history_ui():
    st.subheader("Transaction History")

    address = st.text_input("Enter Address to View History")
    if st.button("Fetch History"):
        # Backend API call to fetch transaction history
        response = requests.get(f"http://localhost:5000/transaction_history?address={address}")
        
        if response.status_code == 200:
            transactions = response.json().get("transactions", [])
            for tx in transactions:
                st.json(tx)  # Display each transaction in JSON format for clarity
        else:
            st.error("Failed to fetch transaction history")