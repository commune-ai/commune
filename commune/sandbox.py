import streamlit as st
import commune

data = {'hey': 'whadup', 'hello': 'world'}
seed = 'bro'
# you can define a seed to generate the key

# if you dont provide anything, it will generate a random key


# Encryption
key = commune.key('bro')

encrypted_data = key.encrypt(data)
decrypted_data = key.decrypt(encrypted_data)
st.write(decrypted_data)


# Signing and Verification

key = commune.key()

# signatures can be strings 
signature_str = key.sign(data)
st.write(signature_str)
signature_dict = key.sign(data, return_dict=True)
st.write(signature_dict)

# you can verify the signature with the public key

assert key.verify(data, signature_str) == True

# if you pass the signature_dict, it already has the data
assert key.verify(signature_dict) == True

# you can save the key and load it later
key_state_dict = key.state_dict()
old_key_address = key.ss58_address

# you can load the key from the state dict
key.load_state_dict(key_state_dict)
st.write(old_key_address==key.ss58_address)
