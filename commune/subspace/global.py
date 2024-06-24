import commune as c
from typing import *

class SubspaceGlobal(c.Module):
    def __init__(self, *args, **kwargs):
        self.set_subspace(*args, **kwargs)
        c.print(self.config)


    def _get_data(self, data):
        return data

    def _set_data(self, data, value):
        data = value
        return data
    

    def global_params(self, 
                    update = False,
                    max_age = 1000,
                    timeout=30,
                    fmt:str='j', 
                    features  = None,
                    value_features = [],
                    path = f'global_params',
                    **kwargs
                    ) -> list:  
        
        features = features or self.config.global_features
        subnet_params = self.get(path, None, max_age=max_age, update=update)
        names = [self.feature2name(f) for f in features]
        future2name = {}
        name2feature = dict(zip(names, features))
        for name, feature in name2feature.items():
            c.print(f'Getting {name} for {feature}')
            query_kwargs = dict(name=feature, params=[], block=None, max_age=max_age, update=update)
            f = c.submit(self.query, kwargs=query_kwargs, timeout=timeout)
            future2name[f] = name
        
        subnet_params = {}

        for f in c.as_completed(future2name):
            result = f.result()
            subnet_params[future2name.pop(f)] = result
        for k in value_features:
            subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)
        return subnet_params



    def balance(self,
                 key: str = None ,
                 block: int = None,
                 fmt='j',
                 max_age=0,
                 update=True) -> Optional['Balance']:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        key_ss58 = self.resolve_key_ss58( key )

        result = self.query(
                module='System',
                name='Account',
                params=[key_ss58],
                block = block,
                
                update=update,
                max_age=max_age
            )

        return  self.format_amount(result['data']['free'] , fmt=fmt)
        
    get_balance = balance 



    def accounts(self, key = None, update=True, block=None, max_age=100000, **kwargs):
        key = self.resolve_key_ss58(key)
        accounts = self.query_map(
            module='System',
            name='Account',
            update=update,
            block = block,
            max_age=max_age,
            **kwargs
        )
        return accounts
    
    def balances(self,fmt:str = 'n', block: int = None, n = None, update=False , **kwargs) -> Dict[str, 'Balance']:
        accounts = self.accounts( update=update, block=block)
        balances =  {k:v['data']['free'] for k,v in accounts.items()}
        balances = {k: self.format_amount(v, fmt=fmt) for k,v in balances.items()}
        return balances
    


    def blocks_per_day(self):
        return 24*60*60/self.block_time


    
    def min_burn(self,  block=None, update=False, fmt='j'):
        min_burn = self.query('MinBurn', block=block, update=update)
        return self.format_amount(min_burn, fmt=fmt)




    def get_balances(self, 
                    keys=None,
                    search=None, 
                    batch_size = 128,
                    fmt = 'j',
                    n = 100,
                    max_trials = 3,
                    names = False,
                    **kwargs):

        key2balance  = {}

        key2address = c.key2address(search=search)
        if keys == None:
            keys = list(key2address.keys())
        if len(keys) > n:
            c.print(f'Getting balances for {len(keys)} keys > {n} keys, using batch_size {batch_size}')
            balances = self.balances(**kwargs)
            key2balance = {}
            for k,a in key2address.items():
                if a in balances:
                    key2balance[k] = balances[a]
        else:
            keys = keys[:n]
            batch_size = min(batch_size, len(keys))
            batched_keys = c.chunk(keys, batch_size)
            num_batches = len(batched_keys)
            progress = c.progress(num_batches)
            c.print(f'Getting balances for {len(keys)} keys')

            def batch_fn(batch_keys):
                substrate = self.get_substrate()
                batch_keys = [key2address.get(k, k) for k in batch_keys]
                c.print(f'Getting balances for {len(batch_keys)} keys')
                results = substrate.query_multi([ substrate.create_storage_key("System", "Account", [k]) for k in batch_keys])
                return  {k.params[0]: v['data']['free'].value for k, v in results}
            key2balance = {}
            progress = c.progress(num_batches)


            for batch_keys in batched_keys:
                fails = 0
                while fails < max_trials:
                    if fails > max_trials:
                        raise Exception(f'Error getting balances {fails}/{max_trials}')
                    try:
                        result = batch_fn(batch_keys)
                        progress.update(1)
                        break # if successful, break
                    except Exception as e:
                        fails += 1
                        c.print(f'Error getting balances {fails}/{max_trials} {e}')
                if c.is_error(result):
                    c.print(result, color='red')
                else:
                    progress.update(1)
                    key2balance.update(result)
        for k,v in key2balance.items():
            key2balance[k] = self.format_amount(v, fmt=fmt)
        if names:
            address2key = c.address2key()
            key2balance = {address2key[k]: v for k,v in key2balance.items()}
        return key2balance



    def total_balance(self, **kwargs):
        balances = self.balances(**kwargs)
        return sum(balances.values())
    

    def num_holders(self, **kwargs):
        balances = self.balances(**kwargs)
        return len(balances)
    



    def proposals(self,  block=None,  nonzero:bool=False, update:bool = False,  **kwargs):
        proposals = [v for v in self.query_map('Proposals', block=block, update=update, **kwargs)]
        return proposals



    def registrations_per_block(self,**kwargs) -> int:
        return self.query('RegistrationsPerBlock', params=[],  **kwargs)
    regsperblock = registrations_per_block
    
    def max_registrations_per_block(self, **kwargs) -> int:
        return self.query('MaxRegistrationsPerBlock', params=[], **kwargs)
 



    #################
    #### Serving ####
    #################
    def update_global(
        self,
        key: str = None,
        network : str = 'main',
        sudo:  bool = True,
        **params,
    ) -> bool:

        key = self.resolve_key(key)
        network = self.resolve_network(network)
        global_params = self.global_params(fmt='nanos')
        global_params.update(params)
        params = global_params
        for k,v in params.items():
            if isinstance(v, str):
                params[k] = v.encode('utf-8')
        # this is a sudo call
        return self.compose_call(fn='update_global',
                                     params=params, 
                                     key=key, 
                                     sudo=sudo)




    #################
    #### Serving ####
    #################
    def vote_proposal(
        self,
        proposal_id: int = None,
        key: str = None,
        network = 'main',
        nonce = None,
        netuid = 0,
        **params,

    ) -> bool:

        self.resolve_network(network)
        # remove the params that are the same as the module info
        params = {
            'proposal_id': proposal_id,
            'netuid': netuid,
        }

        response = self.compose_call(fn='add_subnet_proposal',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)

        return response



