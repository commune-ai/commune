
import commune as c
from typing import Dict , Any, List
import streamlit as st
import json
class Users(c.Module):
    
    default_role = 'homie'
    default_network = 'commune'
    
    def __init__(self, 
                 config=None, 
                 **kwargs):
        self.config = self.set_config(config, kwargs=kwargs)
    
    @property
    def network(self) -> str:
        return self.config['network']
    
    @property
    def users(self):
        '''
        users : dict(network:str, name)
        '''
        return self.config['users']
    
    
    def set_user(self,user, auth):
        raise NotImplemented
    
    def set_network(self,network) -> Dict:
        assert isinstance(network, str)
        self.config['network'] = network
        return {'network': self.config['network'], 'success': True}
       
    
    def resolve_network(self, network:str = None) -> str:
        if network == None:
            network  = self.default_network
        assert isinstance(network, str)
        return network
    
    def resolve_role(self, role:str = None) -> str:
        if role == None:
            role  = self.default_role
        return role
    
    def resolve_username(self, name, seperator='::'):
        index = 0
        while name not in self.users:
            name = seperator.join(name, str(index))
            index += 1
            
        return name
    
    @property
    def user2address(self):
        return {u:u_info['address'] for u,u_info in self.users.items()}
    
    @property
    def user2role(self):
        return {u:u_info['role'] for u,u_info in self.users.items()}
    
    @property
    def address2user(self):
        return self.reverse_map(self.user2address)

    @property
    def address2role(self):
        return self.reverse_map(self.user2address)
    
    
    def get_role(self, user):
        if user in self.address2role:
            role = self.address2role[user]
        elif user in self.user2role:
            role =  self.user2role[user]
        else:
            raise NotImplementedError('you did not specify a legit user or address')
    
    def user_from_auth(self, auth):
        address = self.verify(auth)
        name = auth.pop('name')
        self.users[name] 

    def add_user(self, 
                 name = None, 
                 address = None, 
                 role = None, 
                 network=None,
                 auth = None,
                 **kwargs):
        
        # ensure name is unique
        
        if auth != None:
            self.verify(auth)
        name = self.resolve_username(name)
        network = self.resolve_network(network)
        role = self.resolve_role(role)
        self.users[name] = {
            'role':  role,
            'network': network,
            'address': address,
            **kwargs
        }
        
        return self.users[name]
    
    def rm_user(self, name:str):
        homie =self.users.pop(name, None)
        return {'msg': f'{homie} is no longer your homie'}

    def sidebar(self):
        with st.form(key='my_form'):
            st.write("Add a key")
            key = st.text_input(label='Key')
            submit_button = st.form_submit_button(label='Submit')
        
        
        
        
    @classmethod
    def st(cls):
        self = Users()

        c.st_sidebar(self.sidebar)()
        
        
        st.write(c.keys())
        
        
        # auth = key.sign('bro')
        # st.write(key.get_key('bro').__dict__)
        # verified = key.verify(auth)
        # address = auth['ss58_address']
        # st.write(key.get_signer(auth))
   
        
    def auth_data(self,
            name:str = None,
            role='homie', 
            network='commune',
            ip = None,
            **extra_field) -> Dict:
        
        role = self.resolve_role(role)
        network = self.resolve_network(network)
        ip = self.resolve_ip(ip)
        
        return {
            'name': name,
            'role': role,
            'network': network,
            'time': c.time(),
            **extra_field
        }

             
    def streamlit_sidebar(self):
        with st.sidebar:
            st.write('# Commune AI Insurance')
            self.streamlit_signin()

if __name__ == "__main__":
    Users.st()
    
    