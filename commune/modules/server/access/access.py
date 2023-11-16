import commune as c
from typing import *





class Access(c.Module):
    sync_time = 0
    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}

    def __init__(self, 
                module : Union[c.Module, str], # the module or any python object
                network: str =  'main', # mainnet
                netuid: int = 0, # subnet id
                sync_interval: int =  30, #  1000 seconds per sync with the network
                timescale:str =  'min', # 'sec', 'min', 'hour', 'day'
                stake2rate: int =  100,  # 1 call per every N tokens staked per timescale
                rate: int =  1,  # 1 call per timescale
                base_rate: int =  100,# base level of calls per timescale (free calls) per account
                fn2rate: dict =  {}, # function name to rate map, this overrides the default rate,
                state_path = f'state_path',
                **kwargs):
        config = self.set_config(kwargs=locals())
        self.module = module
        self.user_info = {}
        self.stakes = {}
        c.thread(self.sync_loop_thread)
        

    def sync_loop_thread(self):
        while True:
            self.sync()
            c.sleep(self.config.sync_interval//2 + (c.random_int(20)-10))


    def sync(self):

        # if the sync time is greater than the sync interval, we need to sync

        
        state = self.get(self.sync_path, default={})

        time_since_sync = c.time() - state.get('sync_time', 0)
        if time_since_sync > self.config.sync_interval:
            self.subspace = c.module('subspace')(network=self.config.network)
            state['stakes'] = self.subspace.stakes(fmt='j', netuid=self.config.netuid)
            state['block'] = self.subspace.block
            state['sync_time'] = c.time()
            self.put(self.sync_path, state)

        self.stakes = state['stakes']
        until_sync = self.config.sync_interval - time_since_sync

        c.print({'block': state['block'],  
                 'until_sync': until_sync,
                 'time_since_sync': time_since_sync})
        
    def is_module_key(self, address: str) -> bool:
        return bool(self.module.key.ss58_address == address)

    def verify(self, input:dict) -> dict:


        address = input['address']
        user_info = self.user_info.get(address, {'last_time_called':0 , 'requests': 0})
        stake = self.stakes.get(address, 0)
        fn = input.get('fn')

        if c.is_admin(address) or self.module.key.ss58_address == address:
            rate_limit = 10e42
        else:
            assert fn in self.module.whitelist or fn in c.helper_whitelist, f"Function {fn} not in whitelist"
            assert fn not in self.module.blacklist, f"Function {fn} is blacklisted" 

            rate_limit = (stake / self.config.stake2rate)
            rate_limit = rate_limit + self.config.base_rate # add the base rate
            rate_limit = rate_limit * self.config.rate # multiply by the rate

        time_since_called = c.time() - user_info['last_time_called']
        seconds_in_period = self.timescale_map[self.config.timescale]

        if time_since_called > seconds_in_period:
            # reset the requests
            user_info['requests'] = 0
        passed = bool(user_info['requests'] <= rate_limit)
        # update the user info


        user_info['rate_limit'] = rate_limit
        user_info['stake'] = stake
        user_info['seconds_in_period'] = seconds_in_period
        user_info['passed'] = passed
        user_info['time_since_called'] = time_since_called
        self.user_info[address] = user_info

        assert  passed,  f"Rate limit too high (calls per second) {user_info}"

        user_info['last_time_called'] = c.time()
        user_info['requests'] +=  1
        # check the rate limit
        return user_info


    @classmethod
    def test(cls, key='vali::fam', base_rate=2):
        
        module = cls(module=c.module('module')(),  base_rate=base_rate)
        key = c.get_key(key)

        for i in range(base_rate*3):
            c.sleep(0.1)
            try:
                c.print(module.verify(input={'address': key.ss58_address, 'fn': 'info'}))
            except Exception as e:
                c.print(e)
                assert i > base_rate

            

            
