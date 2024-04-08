import commune as c
import streamlit as st

class Login(c.Module):
    whitelist = []
    def __init__(self):
        self.set_config(locals())
    
    def passwords(self):
        return self.get('allowed_password', [])
    
    def add_password(self, password):
        passwords = self.passwords()
        passwords.append(str(password))
        passwords = list(set(passwords))
        return self.put('allowed_password', passwords)

    def app(self, x:int = 1, y:int = 2) -> int:
        password = st.text_input('Password', '123456', type='password')
        if c.is_ticket(password):
            key_address = c.verify_ticket(password)
            st.write(key_address)
        st.write(self.key.ss58_address)
        
    

Login.run(__name__)