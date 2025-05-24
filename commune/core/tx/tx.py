import sys
import time
import sys
from typing import Any
import inspect
import commune as c
import json
from copy import deepcopy

class Tx:

    def __init__(self, 
                key:str = None, 
                path = '~/.commune/tx' , 
                serializer='serializer',
                auth = 'auth',
                roles = ['client', 'server'],
                tx_schema = {
                    'module': str,
                    'fn': str,
                    'params': dict,
                    'result': dict,
                    'time': int,
                    'time_delta': float,
                    'schema': dict,
                    'signature': str,
                    'hash': str
                },
                 version='v0'):

        self.key = c.key(key)
        self.version = version
        self.store = c.module('store')(f'{path}/{self.version}')
        self.tx_schema = tx_schema
        self.tx_features = list(self.tx_schema.keys())
        self.serializer = c.module(serializer)()
        self.auth = c.module(auth)()
        self.roles = roles


    def create_tx(self, 
                 module:str = 'module', 
                 fn:str = 'forward', 
                 params:dict = {}, 
                 result:Any = {}, 
                 schema:dict = {},
                 auths = {}
                 ):

        """ 
        create a transaction
        """

        result = self.serializer.forward(result)
        auths = auths or self.get_auths(module, fn, params, result)
        tx = {
            'module': module, # the module name (str)
            'fn': fn, # the function name (str)
            'params': params, # the params is the input to the function  (dict)
            'schema': schema, # the schema of the function (dict)
            'result': result, # the result of the function (dict)
            'client': auths['client'], # the client auth (dict)
            'server': auths['server'], # the server auth (dict)
        }
        tx['hash'] = c.hash(tx) # the hash of the transaction (str)
        assert self.verify(tx)
        self.store.put(self.tx_path(tx), tx)
        return tx

    forward = create = tx = create_tx

    def verify_tx(self, tx):
        """
        verify the transaction
        """
        auth_data = self.get_role_auth_data_map(**tx)
        for role in self.roles:
            assert self.auth.verify(tx[role], data=auth_data[role]), f'{role} auth is invalid'
        return True

    vtx = verify = verify_tx

    def paths(self):
        return self.store.paths()

    def tx_path(self, tx):
        return f'{tx["module"]}/{tx["fn"]}/{tx["hash"]}'
   
    def _rm_all(self):
        """
        DANGER: This will permanently remove all transactions from the store.
        remove the transactions
        """
        paths = self.store.paths()
        for p in self.store.paths():
            self.store.rm(p)
        new_paths = self.store.paths()
        assert len(new_paths) == 0, f'Failed to remove all transactions. Remaining paths: {new_paths}'
        return {'status': 'success', 'message': 'All transactions removed successfully', 'removed_paths': paths}

    def is_tx(self, tx):
        """
        Check if the transaction is valid
        """
        if isinstance(tx, str):
            tx = self.store.get(tx)
        if not isinstance(tx, dict):
            return False
        if not all([key in tx for key in self.tx_features]):
            return False
        return True



    def txs(self, module=None, fn=None, features = ['module', 'fn', 'time', 'params']):
        items =  self.store.items('tx')
        items = [x for x in items if self.is_tx(x)]
        df = c.df(items)
        current_time = time.time()
        if len(df) == 0:
            return df    
        df['age'] = df['time'].apply(lambda x: current_time - x)
        df['time'] = df['time'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
        df = df.sort_values(by='time', ascending=False)

        return df[features]

    def n(self):
        """
        Get the number of transactions
        """
        return len(self.store.items('tx'))

    def tx2age(self):
        return self.store.path2age()
        
    def test(self):
        """
        Test the transaction
        """
        t0 = time.time()
        tx = {
            'module': 'test',
            'fn': 'test',
            'params': {'a': 1, 'b': 2},
            'result': {'a': 1, 'b': 2},
            'schema': {
                'input': {
                    'a': 'int',
                    'b': 'int'
                },
                'output': {
                    'a': 'int',
                    'b': 'int'
                }
            },
        }

        tx = self.forward(**tx)

        assert self.verify(tx), 'Transaction is invalid'
        print('Transaction is valid')

        return { 'time': time.time() - t0, 'msg': 'Transaction test passed'}
        

    def get_auths(self, module:str, fn:str, params:dict, result:Any):
        """
        Get the auths for the transaction
        """
        auth_data = self.get_role_auth_data_map(module, fn, params, result)
        return {role : self.get_auth(auth_data[role]) for role in self.roles}

    def get_role_auth_data_map(self, module:str, fn:str, params:dict, result:Any, **_ignore_params):
        """
        Get the auth data for each role 
            client: the client auth data
            server: the server auth data
        """
        return {
                'client': {'fn': fn, 'params': params},
                'server': {'fn': fn, 'params': params, 'result': result}
                }