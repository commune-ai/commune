import commune as c
import yaml
from yaml.loader import SafeLoader
import streamlit as st

class StreamlitAuth(c.Module):
    config_path = c.repo_name + '/data/auth_config.yaml'
    def __init__(self, config_path=None):

        self.load_config(config_path)

        self.authenticator = self.create_authenticator()
    
    def config_template_path(self):
        return self.dirpath() + '/config_template.yaml'

    def config_template(self):
        return self.load_config(self.config_template_path())

    def load_config(self, config_path=None):
        config_path = config_path if config_path != None else self.config_path
        if not c.path_exists(config_path):
            c.put_yaml(config_path, self.config_template())
        self.config = c.get_yaml(config_path)
        return self.config
    @staticmethod
    def hash_passwords(passwords):
        import streamlit_authenticator as stauth
        return stauth.Hasher(passwords).generate()

    def create_authenticator(self):
        import streamlit_authenticator as stauth
        credentials = self.config['credentials']
        cookie_config = self.config['cookie']
        preauthorized = self.config['preauthorized']
        return stauth.Authenticate(
            credentials,
            cookie_config['name'],
            cookie_config['key'],
            cookie_config['expiry_days'],
            preauthorized
        )
    

    def login(self, form_name, location):
        return self.authenticator.login(form_name, location)

    def logout(self, button_name, location):
        return self.authenticator.logout(button_name, location)

    def handle_authentication(self):
        name, authentication_status, username = self.authenticator.login('Login', 'main')
        if authentication_status:
            self.authenticator.logout('Logout', 'main')
            # Handle authenticated user
        elif authentication_status is False:
            # Handle incorrect username/password
            st.error('Incorrect username/password')
        elif authentication_status is None:
            st.info('Please log in')
            # Handle no input case

    # Additional methods to implement user privileges and other features can be added here

# Usage example:
# authenticator_manager = StreamlitAuthenticatorManager('../config.yaml')
# authenticator_manager.handle_authentication()

    def install(self):
        c.cmd("pip3 install streamlit-authenticator")
        return {"streamlit-authenticator": "installed"}
    
    @classmethod
    def dashboard(cls):
        import streamlit as st
        st.title("Streamlit Authenticator")
        st.write("This is a module for authenticating users in Streamlit applications.")
        st.write("It is based on the streamlit-authenticator package.")
        self = cls()
        self.handle_authentication()
        st.write(self.config)
        
    



