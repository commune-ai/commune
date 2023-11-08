import commune as c
from typing import *





class Access(c.Module):
    sync_time = 0
    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}

    def __init__(self, 
                module : Union[c.Module, str], # the module or any python object
                network: str =  'main', # mainnet
                netuid: int = 0, # subnet id
                sync_interval: int =  60, #  1000 seconds per sync with the network
                timescale:str =  'min', # 'sec', 'min', 'hour', 'day'
                stake2rate: int =  100,  # 1 call per every N tokens staked per timescale
                rate: int =  10,  # 1 call per timescale
                base_rate: int =  1,# base level of calls per timescale (free calls) per account
                fn2rate: dict =  {}, # function name to rate map, this overrides the default rate
                **kwargs):
        config = self.set_config(kwargs=locals())
        c.print('fam')
        self.module = module
        self.user_info = {}
        self.stakes = {}
        c.thread(self.sync_loop)
        

    def sync_loop(self):
        self.subspace = c.module('subspace')(network=self.config.network, netuid=self.config.netuid)
        while True:
            self.sync()
            c.sleep(self.config.sync_interval//2)

    def sync(self):
        # if the sync time is greater than the sync interval, we need to sync

        sync_path = f'sync_state.{self.config.network}{self.config.netuid}'
        state = self.get(sync_path, default={})

        sync_time = state.get('sync_time', 0)
        if c.time() - sync_time > self.config.sync_interval:
            state['sync_time'] = c.time()
            state['stakes'] = self.subspace.stakes(fmt='j', netuid=self.config.netuid)
            state['block'] = self.subspace.block
            self.put(sync_path, state)
        c.print({k: v for k, v in state.items() if k != 'stakes'})
        self.stakes = state['stakes']
        
    def is_module_key(self, address: str) -> bool:
        return bool(self.module.key.ss58_address == address)

    def verify(self, input:dict) -> dict:

        address = input['address']
        if c.is_admin(address) or self.module.key.ss58_address == address:
            return input
        else:
            fn = input.get('fn')
            assert fn in self.module.whitelist or fn in c.helper_whitelist, f"Function {fn} not in whitelist"
            assert fn not in self.module.blacklist, f"Function {fn} is blacklisted" 

            # RATE LIMIT CHECKING HERE
            stake = self.stakes.get(address, 0)
            # get the rate limit for the function
            if fn in self.config.fn2rate:
                rate = self.config.fn2rate[fn]
            else:
                rate = self.config.rate
            rate_limit = (stake / self.config.stake2rate)
            rate_limit = rate_limit + self.config.base_rate # add the base rate
            rate_limit =rate_limit
            rate_limit = rate_limit * rate # multiply by the rate

            default_user_info = {
                                'requests': 0, 
                                'last_time_called': 0,
                                'rate': 0,
                                'stake': stake,
                                }

            seconds_in_period = self.timescale_map[self.config.timescale]
            

            user_info = self.user_info.get(address, default_user_info)
            time_since_called = c.time() - user_info['last_time_called'] 
            periods_since_called = time_since_called // seconds_in_period
            if periods_since_called > 1:
                # reset the requests
                user_info['requests'] = 0
    
            user_info['rate_limit'] = rate_limit
            assert rate <= rate_limit, f"Rate limit too high (calls per second) {user_info}"
            # update the user info
            user_info['requests'] += 1
            user_info['last_time_called'] = c.time()
            user_info['time_since_called'] = time_since_called
            user_info['stake'] = stake
            user_info['period'] = seconds_in_period
            c.print(user_info)
            self.user_info[address] = user_info
            # check the rate limit
            return input


    @classmethod
    def test(cls, key=None):
        module = cls(module=c.module('module')())
        key = c.get_key('fam')

        for i in range(10):
            c.sleep(0.1)
            try:
                c.print(module.verify(input={'address': key.ss58_address, 'fn': 'info'}))
            except Exception as e:
                c.print(e)

            
