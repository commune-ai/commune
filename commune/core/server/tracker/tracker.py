import sys
import time
import sys
from typing import Any
import inspect
import commune as c
import json
from copy import deepcopy
import time

class Tx:

    def __init__(self, 
                tx_path = '~/.commune/server/tx' , 
                serializer='serializer',
                auth = 'auth',
                private = False,
                roles = ['client', 'server'], # the roles that need to sign the transaction
                tx_schema = {
                    'mod': str,
                    'fn': str,
                    'params': dict,
                    'result': dict,
                    'schema': dict,
                    'client': dict,  # client auth
                    'server': dict,  # server auth
                    'hash': str
                }, # the schema of the transaction
                key:str = None, 
                 version='v0'):

        self.key = c.key(key)
        self.version = version
        self.store = c.mod('store')(f'{tx_path}/{self.version}', private=private, key=self.key)
        self.tx_schema = tx_schema
        self.tx_features = list(self.tx_schema.keys())
        self.serializer = c.mod(serializer)()
        self.auth = c.mod(auth)()
        self.roles = roles

    def forward(self, 
                 mod:str = 'mod', 
                 fn:str = 'forward', 
                 bid = None, 
                 cost = None,
                 params:dict = {}, 
                 result:Any = {}, 
                 schema:dict = {},
                 auths = {},
                 key = None,
                 client= None,
                 server= None
                 ):

        """ 
        create a transaction
        """
        result = self.serializer.forward(result)

        if client is None or server is None: 
            auths = self.get_auths(mod, fn, params, result, key=key)
        else: 
            if client is not None:
                auths['client'] = client
            if server is not None:
                auths['server'] = server
            
        tx = {
            'mod': mod, # the mod name (str)
            'fn': fn, # the function name (str)
            'params': params, # the params is the input to the function  (dict)
            'result': result, # the result of the function (dict)
            'schema': schema, # the schema of the function (dict)
            'client': auths['client'], # the client auth (dict)
            'server': auths['server'], # the server auth (dict)
        }
        tx['hash'] = c.hash(tx) # the hash of the transaction (str)
        assert self.verify(tx)
        # 
        tx_path = f'{tx["client"]["key"]}/{tx["server"]["key"]}/{fn}_{auths["client"]["time"]}'
        c.print('tx', tx)
        self.store.put(tx_path, tx)
        return tx

    create_tx = create = tx = forward



    def verify(self, tx):
        """
        verify the transaction
        """
        auth_data = self.get_role_auth_data_map(**tx)
        for role in self.roles:
            assert self.auth.verify(tx[role], data=auth_data[role]), f'{role} auth is invalid'
        return True

    def paths(self, path=None):
        return self.store.paths(path=path)

    def encrypted_paths(self, path=None):
        """
        Get the encrypted paths of the transactions
        """
        return self.store.encrypted_paths(path=path)
   
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


    def transform_params(self, params):
        if len(params.get('args', {})) > 0 and len(params.get('kwargs', {})) == 0:
            return params['args']
        elif len(params.get('args', {})) == 0 and len(params.get('kwargs', {})) > 0:
            return params['kwargs']
        elif len(params.get('args', {})) > 0 and len(params.get('kwargs', {})) > 0:
            return params
        elif len(params.get('args', {})) == 0 and len(params.get('kwargs', {})) == 0:
            return params
        else:
            return params
    def txs(self, 
            search=None,
            client= None,
            server= None,
            n = None,
            max_age:float = 3600, 
            features:list = ['mod', 'fn', 'params', 'cost', 'duration',  'age', 'client', 'server', 'time', 'hash'],
            shorten_features = ['client', 'server',  'hash'],
            index = 'hash'
            ):
        path = None
        if client is not None:
            path = f'{client}/'
        if server is not None:
            path = f'*/{server}' if path is not None else f'/{server}/'
        txs = [x for x in self.store.values(path, search=search) if self.is_tx(x)] 
        txs = c.df(txs)
        current_time = time.time()
        if len(txs) == 0:
            return txs    
        df = txs
    
        df['time_start_utc'] = df['client'].apply(lambda x: float(x['time']))
        df['age'] = time.time() - df['time_start_utc']
        df = df[df['age'] <  max_age] if max_age is not None else df
        df['time_end_utc'] = df['server'].apply(lambda x: float(x['time']))
        df['duration'] = df['time_end_utc'] - df['time_start_utc']

        df['time'] = df['time_end_utc'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))

        addres2key = c.address2key()
        df['params'] = df['params'].apply(self.transform_params)
        df['cost'] = df['schema'].apply(lambda x: x['cost'] if 'cost' in x else 0)
        df['client'] = df['client'].apply(lambda x: x['key'])
        df['server'] = df['server'].apply(lambda x: x['key'])
        shorten_fn = lambda x: x if len(x) <= 10 else x[:4] + '...' + x[-4:]
        for f in shorten_features:
            if f in df.columns:
                df[f] = df[f].apply(shorten_fn)
            
        df = df.sort_values(by='time', ascending=False)
        if n is not None:
            df = df.head(n)
        df = df[features] if features is not None else df
        df = df.set_index(index, drop=True)
            
        return df

    def n(self):
        """
        Get the number of transactions
        """
        return len(self.store.items())

    def tx2age(self):
        return self.store.path2age()
        
    def test(self):
        """
        Test the transaction
        """
        t0 = time.time()
        tx = {
            'mod': 'test',
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

    def get_auths(self, mod:str, fn:str, params:dict, result:Any, key=None):
        """
        Get the auths for the transaction
        """
        auth_data = self.get_role_auth_data_map(mod, fn, params, result)
        return {role : self.auth.headers(auth_data[role], key=key) for role in self.roles}

    def get_role_auth_data_map(self, mod:str, fn:str, params:dict, result:Any, **_ignore_params):
        """
        Get the auth data for each role 
            client: the client auth data
            server: the server auth data
        """
        return {
                'client': {'fn': fn, 'params': params},
                'server': {'fn': fn, 'params': params, 'result': result}
                }




