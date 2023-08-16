
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
    def users(self):
        '''
        users : dict(network:str, name)
        '''
        return list(self.config['users'].keys())
    
    def save(self, path=None):
        self.save_config(self.config)
    
    
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
                 name: str , 
                 address : str ,
                 role = 'friend', 
                 address_type: str = 'sr25519',
                 info : dict = None, 
                 refresh: bool = False
                 ):
        
        # ensure name is unique

        info = info or {}
        
        network = c.resolve_network(network)
        role = self.resolve_role(role)
        
        if self.address_exists(address):
            return {'msg': f'User with key {address} already exists', 'success': False}
        if self.user_exists(name) and refresh == False:
            return {'msg': f'{name} is already a user', 'success': False}
        
        self.users[name] = {
            'address': address,
            'role':  role,
            'network': network,
            'address_type': address_type,
            'info': info
        }
        
        self.save()
        
        c.print('bro')
        return {'msg': f'{name} is now a user', 'success': True}
    
    def rm_user(self, name:str):
        homie =self.users.pop(name, None)
        self.save()
        return {'msg': f'{homie} is no longer your homie'}

    default_roles  = ['homie', 'admin', 'user']
    @property
    def user_roles(self):
        return self.config.get('roles', self.default_roles)

        
    def auth_data(self,
            name:str = None,
            role='homie', 
            network='commune',
            ip = None,
            **extra_field) -> Dict:
        
        role = self.resolve_role(role)
        network = c.resolve_network(network)
        ip = c.ip()
        
        return {
            'name': name,
            'role': role,
            'network': network,
            'time': c.time(),
            **extra_field
        }
        
    