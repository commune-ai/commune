
import commune as c
from typing import Dict , Any, List
import streamlit as st
import json
import os


st.button('Hit me')
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
    
    def save(self, path=None):
        self.save_config(self.config)
    

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
        while self.user_exists(name):
            name = seperator.join([name, str(index)])
            index += 1
            
        return name
    
    @property
    def user2address(self):
        return {u:u_info['ss58_address'] for u,u_info in self.users.items()}
    
    @property
    def user2role(self):
        return {u:u_info['role'] for u,u_info in self.users.items()}
    
    @property
    def address2user(self):
        return self.reverse_map(self.user2address)

    def address_exists(self, address:str):
        return address in self.address2user


    def user_exists(self, user:str):
        return user in self.users

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
                 name , 
                 ss58_address ,
                 role = None, 
                 network=None,
                 auth = None,
                 **kwargs):
        
        # ensure name is unique
        
        network = self.resolve_network(network)
        role = self.resolve_role(role)
        
        if self.address_exists(ss58_address):
            return {'msg': f'User with key {ss58_address} already exists', 'success': False}
        if self.user_exists(name):
            return {'msg': f'{name} is already a user', 'success': False}
        
        self.users[name] = {
            'ss58_address': ss58_address,
            'role':  role,
            'network': network,
            **kwargs
        }
        
        
        self.save()
        
        return {'msg': f'{name} is now a user', 'success': True}
    
    def rm_user(self, name:str):
        homie =self.users.pop(name, None)
        self.save()
        return {'msg': f'{homie} is no longer your homie'}

    def st_add_key(self):
        with st.form(key='my_form'):
            st.write("Add a key")
            key = st.text_input(label='Key')
            submit_button = st.form_submit_button(label='Submit')
        
    
    default_roles  = ['homie', 'admin', 'user']
    @property
    def user_roles(self):
        return self.config.get('roles', self.default_roles)
        
        
        
    def is_error(self, response):
        if isinstance(response, dict):
            return response['success']
        else:
            return False
    @classmethod
    def st(cls):
        self = Users()
        self.local_css()
        self.keys = c.keys()
        self.button = {}
        

        with st.sidebar:
            st.write('## Users')
            with st.form(key='Sign In'):
                user_info = {}
                username = st.text_input(label='Name', value='bro')
                password = st.text_input(label='Password', value=f'0x')
                
                seed = f'{password}'
                self.button['sign_in'] = st.form_submit_button(label='sign in')
                if self.button['sign_in']:
                    c.add_key(path=username,seed=seed)
                    self.key = c.get_key(username)
                    st.write(self.key.ss58_address)
                    response = self.add_user(ss58_address=self.key.ss58_address, name=username, role='admin')
                    if response['success']:
                        st.success(response['msg'])
                    else:
                        st.error(response['msg'])
                        
            with st.form(key='my_form'):
                user_info = {}
                user_info['name'] = st.text_input(label='Name', value='bro')
                user_info['ss58_address'] = st.text_input(label='Public Key', value=f'0x')

                # user_info['module_address'] = st.text_input(label='Module Address', value=f'{c.default_ip}')
                user_info['role'] = st.selectbox(label='Role', options=self.user_roles)
                # user_info['network'] = st.selectbox(label='Network', options=['commune', 'polkadot'])

                self.button['add_user'] = st.form_submit_button(label='add')
                if self.button['add_user']:
                    response = self.add_user(**user_info)
                    if response['success']:
                        st.success(response['msg'])
                    else:
                        st.error(response['msg'])
            with st.expander('Manage Users'):
                seleted_roles = st.multiselect('Select Role',self.user_roles, self.user_roles)

                users = list(self.users.keys())
                if seleted_roles:
                    selected_users = [u for u in users if self.user2role[u] in seleted_roles]
                selected_users = st.multiselect('Select User',users, selected_users)
                self.button['rm_user'] = st.button(label='rm')
                if self.button['rm_user']:
                    for user in selected_users:
                        self.rm_user(user)
                    users = []
                # self.button['rm_user'] = cols[1].button(label='rm')
            
            with st.expander('Users', expanded=True):
                st.write(self.users)
        
        self.st_sidebar()
        
        
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
        
    def st_signin(self):
        st.write('## Sign in')

             
    def st_sidebar(self):
        with st.sidebar:
            st.write('# Commune AI Insurance')
            self.st_signin()

    @staticmethod
    def local_css(file_name=os.path.dirname(__file__)+'/style.css'):
        import streamlit as st

        
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if __name__ == "__main__":
    Users.st()
    
    