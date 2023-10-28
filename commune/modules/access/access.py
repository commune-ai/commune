import commune as c
from typing import *





class Access(c.Module):
    sync_time = 0
    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}

    def __init__(self, 
                module : Any, # the module or any python object
                network: str =  'main', # mainnet
                netuid: int = 0, # subnet id
                sync_interval: int =  1000, #  1000 seconds per sync with the network
                timescale:str =  'min', # 'sec', 'min', 'hour', 'day'
                stake2rate: int =  100,  # 1 call per every N tokens staked per timescale
                rate: int =  1,  # 1 call per timescale
                base_rate: int =  0,# base level of calls per timescale (free calls) per account
                fn2rate: dict =  {}, # function name to rate map, this overrides the default rate
                **kwargs):
        config = self.set_config(kwargs=locals())
        c.print(config)
        self.module = module
        self.user_info = {}

    def sync(self):
        sync_time  = c.time() - self.sync_time
        # if the sync time is greater than the sync interval, we need to sync
        try:
            if sync_time >  self.config.sync_interval :
                self.subspace = c.module('subspace')(network=self.config.network, netuid=self.config.netuid)
                self.stakes = self.subspace.stakes(fmt='j')
                self.sync_time = c.time()
        except Exception as e:
            c.print(f"Error syncing {e}")
            self.subspace = None
            self.stakes = {}
            return
        

    def verify(self, input:dict) -> dict:

        address = input['address']
        if c.is_admin(address):
            return input
        else:
            self.sync()
            if self.subspace == None:
                raise Exception(f"Subspace not initialized and you are not an authorized admin {input['address']}, authorized admins: {c.admins()}")
            # if not an admin address, we need to check the whitelist and blacklist
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
            rate_limit = rate_limit + self.config.base_rate

            # convert the rate limit to the correct timescale
            rate_limit = rate_limit / self.timescale_map[self.config.timescale]

            default_user_info = {
                                'requests': 0, 
                                'last_time_called': 0,
                                'rate': 0,
                                'stake': stake
                                }

            
            user_info = self.user_info.get(address, default_user_info)
            user_rate = 1 / (c.time() - user_info['last_time_called'] + 1e-10)        
            assert user_rate < rate_limit, f"Rate limit too high (calls per second) {user_rate} > {rate_limit}"
            # update the user info
            user_info['last_time_called'] = c.time()
            user_info['requests'] += 1
            user_info['rate'] = user_rate
            user_info['rate_limit'] = rate_limit

            self.user_info[address] = user_info
            
            return input


    @classmethod
    def test(cls):
        server_name = 'access_subspace.demo' 
        module = c.serve('module', server_name=server_name, wait_for_server=True)['name']

        for key in [None, 'fam']:
            client = c.connect(server_name, key=key)
            for n in range(10):
                c.print(client.info(timeout=4))
            c.kill(server_name)
            return {'name': server_name, 'module': module, 'client': client}

            


