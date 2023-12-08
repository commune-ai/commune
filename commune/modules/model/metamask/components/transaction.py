import streamlit as st
import requests

def display_transaction_ui():
    st.subheader("Send Ethereum Transaction")
    recipient_address = st.text_input("Recipient Address")
    amount = st.number_input("Amount of Ether to Send", min_value=0.0, format="%.5f")

    if st.button("Send Transaction"):
        transaction_data = {
            'to': recipient_address,
            'value': amount,
            # Add other necessary fields
        }
        
        response = requests.post("http://localhost:5000/send_transaction", json=transaction_data)
        st.write(response.json())