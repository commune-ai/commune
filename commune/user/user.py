
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
    @classmethod
    def add_user(cls, address, role='user', name=None, **kwargs):
        assert c.valid_ss58_address(address), f'{address} is not a valid address'
        users = cls.get('users', {})
        info = {'role': role, 'name': name, **kwargs}
        users[address] = info
        cls.put('users', users)
        return {'success': True, 'user': address,'info':info}
    
    @classmethod
    def users(cls, role=None):
        users = cls.get('users', {})
        root_key_address  = c.root_key().ss58_address
        if root_key_address not in users:
            cls.add_admin(root_key_address)
        if role is not None:
            return {k:v for k,v in users.items() if v['role'] == role}
        return cls.get('users', {})

    def roles(self):
        return list(set([v['role'] for k,v in self.users().items()]))

    
    @classmethod
    def is_user(self, address):
        return address in self.users()

    def is_blacklisted(self, address):
        return address in self.blacklist()


    def blacklist_user(self, address):
        blacklist = self.blacklist()
        assert c.valid_ss58_address(address), f'{address} is not a valid address'
        blacklist.append(address)
        self.put('blacklist', blacklist)
        return {'success': True, 'msg': f'blacklisted {address}'}

    def whitelist_user(self, address):
        blacklist = self.blacklist()
        blacklist.remove(address)
        self.put('blacklist', blacklist)
        return {'success': True, 'msg': f'whitelisted {address}'}

    def blacklist(self):
        return self.get('blacklist', [])


    @classmethod
    def get_user(cls, address):
        users = cls.users()
        return users.get(address, None)
    @classmethod
    def update_user(cls, address, **kwargs):
        info = cls.get_user(address)
        info.update(kwargs)
        return cls.add_user(address, **info)
    @classmethod
    def get_role(cls, address:str, verbose:bool=False):
        try:
            return cls.get_user(address)['role']
        except Exception as e:
            c.print(e, color='red', verbose=verbose)
            return None
    @classmethod
    def refresh_users(cls):
        cls.put('users', {})
        assert len(cls.users()) == 0, 'users not refreshed'
        return {'success': True, 'msg': 'refreshed users'}
    @classmethod
    def user_exists(cls, address:str):
        return address in cls.get('users', {})

    @classmethod
    def is_root_key(cls, address:str)-> str:
        return address == c.root_key().ss58_address
    @classmethod
    def is_admin(cls, address:str):
        return cls.get_role(address) == 'admin'
    @classmethod
    def admins(cls):
        return [k for k,v in cls.users().items() if v['role'] == 'admin']
    @classmethod
    def add_admin(cls, address):
        return  cls.add_user(address, role='admin')
    @classmethod
    def rm_admin(cls, address):
        return  cls.rm_user(address)
    @classmethod
    def num_roles(cls, role:str):
        return len([k for k,v in cls.users().items() if v['role'] == role])
    
    @classmethod
    def rm_user(cls, address):
        users = cls.get('users', {})
        users.pop(address)
        cls.put('users', users)
        
        assert not cls.user_exists(address), f'{address} still in users'
        return {'success': True, 'msg': f'removed {address} from users', 'users': cls.users()}
    
    
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
        
    @classmethod
    def app(cls):
        import streamlit as st
        st.write('### Users')
        self = cls()
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
                response = getattr(cls, f'rm_user')(add_user_address)
                st.write(response)


    def test_blacklisting(self):
        blacklist = self.blacklist()
        key = c.get_key('test')
        assert key.ss58_address not in self.blacklist(), 'key already blacklisted'
        self.blacklist_user(key.ss58_address)
        assert key.ss58_address in self.blacklist(), 'key not blacklisted'
        self.whitelist_user(key.ss58_address)
        assert key.ss58_address not in self.blacklist(), 'key not whitelisted'
        return {'success': True, 'msg': 'blacklist test passed'}
        
User.run(__name__)

