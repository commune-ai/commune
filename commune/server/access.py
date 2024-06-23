import commune as c
from typing import *


class Access(c.Module):

    sync_time = 0
    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400, 'minute': 60, 'second': 1}

    def __init__(self, 
                module : Union[c.Module, str] = None, # the module or any python object
                network: str =  'subspace', # mainnet
                netuid: int = 'all', # subnet id
                timescale:str =  'min', # 'sec', 'min', 'hour', 'day'
                stake2rate: int =  100.0,  # 1 call per every N tokens staked per timescale
                max_rate: int =  1000.0, # 1 call per every N tokens staked per timescale
                role2rate: dict =  {}, # role to rate map, this overrides the default rate,
                state_path = f'state_path', # the path to the state
                refresh: bool = False,
                stake_from_weight = 1.0, # the weight of the staker
                max_age = 30, # max age of the state in seconds
                max_staleness: int =  60, #  1000 seconds per sync with the network

                **kwargs):
        
        self.set_config(locals())
        self.user_module = c.module("user")()
        self.state_path = self.resolve_path(state_path)
        if refresh:
            self.rm_state()
        self.last_time_synced = c.time()
        self.state = {'sync_time': 0, 
                      'stake_from': {}, 
                      'role2rate': role2rate, 
                      'fn_info': {}}
        
        self.set_module(module)
        print(c.thread, 'access fam', c.pwd())
        c.thread(self.run_loop)

    def set_module(self, module):
        if isinstance(module, str):
            module = c.module(module)()
        self.module = module
        return module

    
    def run_loop(self):
        while True:
            try:
                r = self.sync_network()
            except Exception as e:
                r = c.detailed_error(e)
            c.print(r)
            c.sleep(self.config.max_staleness)


    def sync_network(self, update=False, max_age=None, netuid=None, network=None):
        state = self.get(self.state_path, {}, max_age=self.config.max_staleness)
        netuid = netuid or self.config.netuid
        network = network or self.config.network
        staleness = c.time() - state.get('sync_time', 0)
        self.key2address = c.key2address()
        self.address2key = c.address2key()
        response = { 
                    'path': self.state_path,
                    'max_staleness':  self.config.max_staleness,
                    'network': network,
                    'netuid': netuid,
                    'staleness': int(staleness), 
                    'datetime': c.datetime()}
        
        if staleness < self.config.max_staleness:
            response['msg'] = 'synced too earlly'
            return response
        else:
            response['msg'] =  'Synced with the network'
            response['staleness'] = 0
        self.subspace = c.module('subspace')(network=network)
        max_age = max_age or self.config.max_age
        state['stakes'] = self.subspace.stakes(fmt='j', netuid=netuid, update=update, max_age=max_age)
        self.state = state
        self.put(self.state_path, self.state)
        return response

    def forward(self, fn: str = 'info' , input:dict = None, address=None) -> dict:
        """
        input : dict 
            fn : str
            address : str

        returns : dict
        """
        input = input or {}
        address = input.get('address', address)
        assert address, f'address not in input or as an argument'
        fn = input.get('fn', fn)

        # ONLY THE ADMIN CAN CALL ANY FUNCTION, THIS IS A SECURITY FEATURE
        # THE ADMIN KEYS ARE STORED IN THE CONFIG
        if c.is_admin(address):
            return {'success': True, 'msg': f'is verified admin'}
        
        assert fn in self.module.whitelist , f"Function {fn} not in whitelist={self.module.whitelist}"
        assert fn not in self.module.blacklist, f"Function {fn} is blacklisted={self.module.blacklist}" 
        
        if address in self.address2key:
            return {'success': True, 'msg': f'address {address} is a local key'}
        if fn.startswith('__') or fn.startswith('_'):
            return {'success': False, 'msg': f'Function {fn} is private'}
        if address in self.address2key:
            return {'success': True, 'msg': f'address {address} is in the whitelist'}

        if c.is_user(address):
            return {'success': True, 'msg': f'is verified user'}


        current_time = c.time()

        # sync of the state is not up to date 

        # get the rate limit for the user
        role2rate = self.state.get('role2rate', {})

        # get the role of the user
        role = self.user_module.get_role(address) or 'public'
        rate_limit = role2rate.get(role, 0)

        # stake rate limit
        stake = self.state.get('stake_from', {}).get(address, 0)
        # we want to also know if the user has been staked from
        stake_from = self.state.get('stake_from', {}).get(address, 0)
        # STEP 1:  FIRST CHECK THE WHITELIST AND BLACKLIST

        total_stake_score = stake 

        # STEP 2: CHECK THE STAKE AND CONVERT TO A RATE LIMIT
        default_fn_info = {'stake2rate': self.config.stake2rate, 'max_rate': self.config.max_rate}
        self.state['fn_info'] = self.state.get('fn_info', {})
        fn2info = self.state['fn_info'].get(fn,default_fn_info)
        stake2rate = fn2info.get('stake2rate', self.config.stake2rate)
        
        rate_limit = (total_stake_score / stake2rate) # convert the stake to a rate

        # STEP 3: CHECK THE MAX RATE
        max_rate = fn2info.get('max_rate', self.config.max_rate)
        rate_limit = min(rate_limit, max_rate) # cap the rate limit at the max rate
        
        # NOW LETS CHECK THE RATE LIMIT
        self.state['user_info'] = self.state.get('user_info', {})
        user_info = self.state['user_info'].get(address, {})

        # check if the user has exceeded the rate limit
        time_since_called = current_time - user_info.get('timestamp', 0)
        period = self.timescale_map[self.config.timescale]
        # if the time since the last call is greater than the seconds in the period, reset the requests
        if time_since_called > period:
            user_info['rate'] = 0
        try:
            assert user_info['rate'] <= rate_limit
            user_info['success'] = True
        except Exception as e:
            user_info['error'] = c.detailed_error(e)
            user_info['success'] = False

       
        # update the user info
        user_info['rate_limit'] = rate_limit
        user_info['key'] = address
        user_info['period'] = period
        user_info['role'] = role
        user_info['fn2requests'] = user_info.get('fn2requests', {})
        user_info['fn2requests'][fn] = user_info['fn2requests'].get(fn, 0) + 1
        user_info['timestamp'] = current_time
        user_info['stake'] = stake
        user_info['stake_from'] = stake_from
        user_info['rate'] = user_info.get('rate', 0) + 1
        user_info['timescale'] = self.config.timescale
        # store the user info into the state
        self.state['user_info'][address] = user_info
        # check the rate limit
        return user_info
    verify = forward

    @classmethod
    def test_whitelist(cls, key='vali::fam', base_rate=2, fn='info'):
        module = cls(module=c.module('module')(),  base_rate=base_rate)
        key = c.get_key(key)

        for i in range(base_rate*3):    
            t1 = c.time()
            result = module.forward(**{'address': key.ss58_address, 'fn': 'info'})
            t2 = c.time()
            c.print(f'🚨 {t2-t1} seconds... 🚨\033', color='yellow')
    

    
    def rm_state(self):
        self.put(self.state_path, {})
        return {'success': True, 'msg': f'removed {self.state_path}'}

    


if __name__ == '__main__':
    Access.run()

            
