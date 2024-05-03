import commune as c
import streamlit as st

class Login(c.Module):
    def __init__(self):
        self.set_config(locals())
    
    def passwords(self):
        return self.get('allowed_password', [])
    
    def add_password(self, password):
        passwords = self.passwords()
        passwords.append(str(password))
        self.put('allowed_password', passwords)

    def app(self, x:int = 1, y:int = 2) -> int:
        password = st.text_input('Password', '123456', type='password')
        self.key = c.module('key').from_password(c.hash(password))
        st.write(self.key.ss58_address)


Login.run(__name__)