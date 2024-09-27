
import commune as c
from typing import Dict , Any, List
import json
import os


class User(c.Module):
    ##################################
    # USER LAND
    ##################################

    def role2users(self):
        role2users = {}
        for user,v in self.users().items():
            role = v['role']
            if role not in role2users:
                role2users[role] = []
            role2users[role].append(user)
        return role2users
    
    def add_user(self, address, role='user', name=None, **kwargs):
        assert c.valid_ss58_address(address), f'{address} is not a valid address'
        users = self.get('users', {})
        info = {'role': role, 'name': name, **kwargs}
        users[address] = info
        self.put('users', users)
        return {'success': True, 'user': address,'info':info}
    
    def users(self, role=None):
        users = self.get('users', {})
        root_key_address  = c.root_key().ss58_address
        if root_key_address not in users:
            self.add_admin(root_key_address)
        if role is not None:
            return {k:v for k,v in users.items() if v['role'] == role}
        return self.get('users', {})

    def roles(self):
        return list(set([v['role'] for k,v in self.users().items()]))

    
    
    def is_user(self, address):
        return address in self.users()

    def blacklist_user(self, address):
        blacklist = self.blacklist()
        assert c.valid_ss58_address(address), f'{address} is not a valid address'
        blacklist.append(address)
        self.put('blacklist', blacklist)
        return {'success': True, 'msg': f'blacklist {address}'}

    def whitelist_user(self, address):
        blacklist = self.blacklist()
        blacklist.remove(address)
        self.put('blacklist', blacklist)
        return {'success': True, 'msg': f'whitelisted {address}'}

    def blacklist(self):
        return self.get('blacklist', [])


    
    def get_user(self, address):
        users = self.users()
        return users.get(address, None)
    
    def update_user(self, address, **kwargs):
        info = self.get_user(address)
        info.update(kwargs)
        return self.add_user(address, **info)
    
    def get_role(self, address:str, verbose:bool=False):
        try:
            return self.get_user(address)['role']
        except Exception as e:
            c.print(e, color='red', verbose=verbose)
            return None
    
    def refresh_users(self):
        self.put('users', {})
        assert len(self.users()) == 0, 'users not refreshed'
        return {'success': True, 'msg': 'refreshed users'}
    
    def user_exists(self, address:str):
        return address in self.get('users', {})

    
    def is_root_key(self, address:str)-> str:
        return address == c.root_key().ss58_address
    
    def is_admin(self, address:str):
        return self.get_role(address) == 'admin'
    
    
    def admins(self):
        return [k for k,v in self.users().items() if v['role'] == 'admin']
    
    def add_admin(self, address):
        return  self.add_user(address, role='admin')
    
    def rm_admin(self, address):
        return  self.rm_user(address)
    
    def num_roles(self, role:str):
        return len([k for k,v in self.users().items() if v['role'] == role])
    
    
    def rm_user(self, address):
        users = self.get('users', {})
        users.pop(address)
        self.put('users', users)
        
        assert not self.user_exists(address), f'{address} still in users'
        return {'success': True, 'msg': f'removed {address} from users', 'users': self.users()}
    
    
    def df(self):
        df = []
        for k,v in self.users().items():
            v['address'] = k
            v['name'] = v.get('name', None)
            v['role'] = v.get('role', 'user')
            df.append(v)
        import pandas as pd
        df = pd.DataFrame(df)
        return df 
        
    
    def app(self):
        import streamlit as st
        st.write('### Users')
        self = self()
        users = self.users()
        

        with st.expander('Users', False):
            st.write(self.df())


        with st.expander('Add Users', True):
            
            cols = st.columns([2,1,1])
            add_user_address = cols[0].text_input('Add User Address')
            role = cols[1].selectbox('Role', ['user', 'admin'])
            [cols[2].write('\n') for i in range(2)]
            add_user = cols[2].button(f'Add {role}')
            if add_user:
                response = getattr(self, f'add_{role}')(add_user_address)
                st.write(response)

        with st.expander('Remove Users', True):
            cols = st.columns([3,1])
            user_keys = list(users.keys())
            rm_user_address = cols[0].selectbox('Remove User Address', user_keys, 0 , key='addres')
            [cols[1].write('\n') for i in range(2)]
            add_user = cols[1].button(f'Remove {rm_user_address[:4]}...')
            if add_user:
                response = getattr(self, f'rm_user')(add_user_address)
                st.write(response)

if __name__ == '__main__':
    User.run()

