
import commune as c
from typing import Dict , Any, List
import streamlit as st
import json
import os


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
    def user_keys(self):
        '''
        users : dict(network:str, name)
        '''
        return list(self.get('users', {}).keys())
    
    
    def resolve_username(self, name, seperator='::'):
        index = 0
        while self.user_exists(name):
            name = seperator.join([name, str(index)])
            index += 1
            
        return name
    
    @property
    def role2address(self):
        return {u:u_info['ss58_address'] for u,u_info in self.users.items()}
    
    @property
    def user2role(self):
        return {u:u_info['role'] for u,u_info in self.users.items()}
    
    @property
    def address2user(self):
        return c.reverse_map(self.user2address)

    def address_exists(self, address:str):
        return address in self.address2user

    def user_exists(self, user:str):
        return user in self.users

    @property
    def address2role(self):
        return c.reverse_map(self.user2address)
    
    
    def get_role(self, user):
        if user in self.address2role:
            role = self.address2role[user]
        elif user in self.user2role:
            role =  self.user2role[user]
        else:
            raise NotImplementedError('you did not specify a legit user or address')
    
    def add_user(self, 
                 address : str ,
                 role = 'user', 
                 **kwargs
                 ):
        
        # ensure name is unique

        info = info or {}
        
        network = c.resolve_network(network)
        role = self.resolve_role(role)
        
        user = {
            'address': address,
            'role':  role,
            **kwargs
        }
        users = self.get('users', {})
        users[address] = user
        self.set('users', users)

        return {'msg': f'{name} is now a user', 'success': True}
    
    def rm_user(self, name:str):
        homie =self.users.pop(name, None)
        self.save()
        return {'msg': f'{homie} is no longer your homie'}

    default_roles  = ['homie', 'admin', 'user']
    @property
    def user_roles(self):
        return self.config.get('roles', self.default_roles)

