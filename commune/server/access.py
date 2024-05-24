import commune as c
from typing import *


class Access(c.Module):

    sync_time = 0
    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400, 'minute': 60, 'second': 1}

    def __init__(self, 
                module : Union[c.Module, str] = None, # the module or any python object
                network: str =  'main', # mainnet
                netuid: int = 0, # subnet id
                timescale:str =  'min', # 'sec', 'min', 'hour', 'day'
                stake2rate: int =  100.0,  # 1 call per every N tokens staked per timescale
                stakfrom2rate = 100, 
                max_rate: int =  1000.0, # 1 call per every N tokens staked per timescale
                role2rate: dict =  {}, # role to rate map, this overrides the default rate,
                state_path = f'state_path', # the path to the state
                refresh: bool = False,
                stake_from_weight = 1.0, # the weight of the staker
                max_age = 30, # max age of the state in seconds
                sync_interval: int =  60, #  1000 seconds per sync with the network

                **kwargs):
        
        self.set_config(locals())
        self.user_module = c.module("user")()
        self.address2key = c.address2key()
        self.set_module(module)
        self.state_path = state_path
        if refresh:
            self.rm_state()
        self.last_time_synced = c.time()
        self.state = {'sync_time': 0, 
                      'stake_from': {}, 
                      'role2rate': role2rate, 
                      'fn_info': {}}

        c.thread(self.run_loop)

        
    def set_module(self, module: c.Module):
        module = module or c.module('module')()
        if isinstance(module, str):
            module = c.module(module)()
        self.module = module

        self.whitelist =  list(set(self.module.whitelist + c.whitelist))
        self.blacklist =  list(set(self.module.blacklist + c.blacklist))

        return {'success': True, 'msg': f'set module to {module}'}
    
    def run_loop(self):
        while True:
            try:
                r = self.sync_network()
            except Exception as e:
                r = c.detailed_error(e)
            c.sleep(self.config.sync_interval)

    def sync_network(self):
        state = self.get(self.state_path, {}, max_age=self.config.sync_interval)
        time_since_sync = c.time() - state.get('sync_time', 0)
        self.key2address = c.key2address()
        self.address2key = c.address2key()
        if time_since_sync > self.config.sync_interval:
            self.subspace = c.module('subspace')(network=self.config.network)
            state['stakes'] = self.subspace.stakes(fmt='j', netuid='all', update=False, max_age=self.config.max_age)
            self.state = state
            self.put(self.state_path, self.state)
            c.print(f'🔄 Synced {self.state_path} 🔄\033', color='yellow')

        response = {'success': True, 
                    'msg': f'synced {self.state_path}', 
                    'until_sync': int(self.config.sync_interval - time_since_sync),
                    'time_since_sync': int(time_since_sync)}
        return response

    def verify(self, 
               address='5FNBuR2yVf4A1v5nt3w5oi4ScorraGRjiSVzkXBVEsPHaGq1', 
               fn: str = 'info' ,
              input:dict = None) -> dict:
        """
        input : dict 
            fn : str
            address : str

        returns : dict
        """
        if input is not None:
            address = input.get('address', address)
            fn = input.get('fn', fn)

        # ONLY THE ADMIN CAN CALL ANY FUNCTION, THIS IS A SECURITY FEATURE
        # THE ADMIN KEYS ARE STORED IN THE CONFIG
        if c.is_admin(address):
            return {'success': True, 'msg': f'is verified admin'}

        
        assert fn in self.whitelist , f"Function {fn} not in whitelist={self.whitelist}"
        assert fn not in self.blacklist, f"Function {fn} is blacklisted={self.blacklist}" 
        
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
        self.sync_network()

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

    @classmethod
    def get_access_state(cls, module):
        access_state = cls.get(module)
        return access_state

    @classmethod
    def test_whitelist(cls, key='vali::fam', base_rate=2, fn='info'):
        module = cls(module=c.module('module')(),  base_rate=base_rate)
        key = c.get_key(key)

        for i in range(base_rate*3):    
            t1 = c.time()
            result = module.verify(**{'address': key.ss58_address, 'fn': 'info'})
            t2 = c.time()
            c.print(f'🚨 {t2-t1} seconds... 🚨\033', color='yellow')
    

    
    def rm_state(self):
        self.put(self.state_path, {})
        return {'success': True, 'msg': f'removed {self.state_path}'}

    


if __name__ == '__main__':
    Access.run()

            
