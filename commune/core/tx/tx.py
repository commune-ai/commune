import sys
import time
import sys
from typing import Any
import inspect
import commune as c
import json
from copy import deepcopy
import time

def transform_params(params):
    if len(params.get('args', {})) > 0 and len(params.get('kwargs', {})) == 0:
        return params['args']
    elif len(params.get('args', {})) == 0 and len(params.get('kwargs', {})) > 0:
        return params['kwargs']
    elif len(params.get('args', {})) > 0 and len(params.get('kwargs', {})) > 0:
        return params
    elif len(params.get('args', {})) == 0 and len(params.get('kwargs', {})) == 0:
        return {}
    else:
        return params
class Tx:

    def __init__(self, 
                tx_path = '~/.commune/cli/tx' , 
                serializer='serializer',
                auth = 'auth',
                private = True,
                roles = ['client', 'server'],
                tx_schema = {
                    'module': str,
                    'fn': str,
                    'params': dict,
                    'result': dict,
                    'schema': dict,
                    'client': dict,  # client auth
                    'server': dict,  # server auth
                    'hash': str
                },
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


    def create_tx(self, 
                 module:str = 'module', 
                 fn:str = 'forward', 
                 params:dict = {}, 
                 result:Any = {}, 
                 schema:dict = {},
                 auths = {},
                 client= None,
                 server= None
                 ):

        """ 
        create a transaction
        """

        # if client is not None:
        #     auths['client'] = client
        # if server is not None:
        #     auths['server'] = server


        result = self.serializer.forward(result)
        if client is not None:
            auths['client'] = client
        if server is not None:
            auths['server'] = server
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
        tx_path = f'{tx["module"]}/{tx["fn"]}/{tx["hash"]}'
        self.store.put(tx_path, tx)

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
    def txs(self, 
            search=None,
            max_age:float = None, 
            features:list = ['module', 'fn', 'params', 'client', 'cost', 'time', 'duration',]):
        txs = [x for x in self.store.values() if self.is_tx(x)] 
        if search is not None:
            txs = [x for x in txs if search in x['module'] or search in x['fn'] or search in json.dumps(x['params'])]
        txs = c.df(txs)
        current_time = time.time()
        if len(txs) == 0:
            return txs    
        df = txs
    
        df['time_start_utc'] = df['client'].apply(lambda x: float(x['time']))
        df['age'] = df['time_start_utc'] - time.time()
        df = df[df['age'] > (current_time - max_age)] if max_age is not None else df
        df['time_end_utc'] = df['server'].apply(lambda x: float(x['time']))
        df['duration'] = df['time_end_utc'] - df['time_start_utc']

        df['time'] = df['time_end_utc'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))

        addres2key = c.address2key()
        df['params'] = df['params'].apply(transform_params)
        df['cost'] = df['schema'].apply(lambda x: x['cost'] if 'cost' in x else 0)
        df['client'] = df['client'].apply(lambda x: addres2key.get(x['key'], x['key']))
        df = df.sort_values(by='time', ascending=False)
        return df[features]

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
        return {role : self.auth.headers(auth_data[role]) for role in self.roles}

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




