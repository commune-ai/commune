        
        
    
import commune as c

import streamlit as st
import os

class UsersApp(c.Module):
    @classmethod
    def app(cls):
        self = c.module('user')()
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
                    c.add_key(path=username,suri=seed)
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
                user_info['key'] = st.text_input(label='Public Key', value=f'0x')

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
        # address = auth['key']
        # st.write(key.get_signer(auth))

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

    def st_add_key(self):
        with st.form(key='my_form'):
            st.write("Add a key")
            key = st.text_input(label='Key')
            submit_button = st.form_submit_button(label='Submit')
        
    