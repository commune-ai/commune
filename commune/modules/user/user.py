
import commune as c
from typing import Dict , Any, List
import streamlit as st
import json
import os


class User(c.Module):
    ##################################
    # USER LAND
    ##################################
    @classmethod
    def add_user(cls, address, role='user', **kwargs):
        users = cls.get('users', {})
        info = {'role': role, **kwargs}
        users[address] = info
        cls.put('users', users)
        return {'success': True, 'user': address,'info':info}
    
    @classmethod
    def users(cls):
        users = cls.get('users', {})
        root_key_address  = c.root_key().ss58_address
        if root_key_address not in users:
            cls.add_admin(root_key_address)
        return cls.get('users', {})
    
    @classmethod
    def is_user(self, address):
        return address in self.users() or address in c.users()
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
    def user_exists(cls, address):
        return address in cls.get('users', {})

    @classmethod
    def is_root_key(cls, address:str)-> str:
        return address == c.root_key().ss58_address
    @classmethod
    def is_admin(cls, address):
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
        return {'success': True, 'msg': f'removed {address} from users'}