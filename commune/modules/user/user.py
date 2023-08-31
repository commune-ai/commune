
import commune as c
from typing import Dict , Any, List
import streamlit as st
import json
import os


class User(c.Module):
    @classmethod
    def add_user(cls, address, role='admin', **kwargs):


        users = cls.get('users', {})
        info = {'role': role, **kwargs}
        users[address] = info
        c.put('users', users)
        return {'success': True, 'user': address,'info':info}

     @classmethod
    def users(cls):
        users = cls.get('users', {})
        root_key_address  = c.root_key().ss58_address
        if root_key_address not in users:
            cls.add_admin(root_key_address)
        return cls.get('users', {})
    
    @classmethod
    def get_user(cls, address):
        users = cls.get('users', {})
        assert address in users, f'{address} not in users'
        return users[address]
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
