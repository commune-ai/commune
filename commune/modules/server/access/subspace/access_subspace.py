import commune as c





class AccessSubspace(c.Module):
    sync_time = 0
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
        self.module = module
        self.sync()
        self.user_info = {}

    def sync(self):
        sync_time  = c.time() - self.sync_time
        # if the sync time is greater than the sync interval, we need to sync
        if sync_time >  self.config.sync_interval :
            self.sync_time = c.time()
            self.subspace = c.module('subspace')(network=self.config.network, netuid=self.config.netuid)
            self.stakes = self.subspace.stakes(fmt='j')
        else:
            c.print(f"Sync time {sync_time} < {self.config.sync_interval}, skipping sync")
            return
        

    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}
    def verify(self, input:dict) -> dict:

        address = input['address']
        if c.is_admin(address):

            return input
        
        if self.subspace == None:
            return {'success': False, 'error': 'subspace is not initialized, perhaps due to a network error, please check your nework'}

        # if not an admin address, we need to check the whitelist and blacklist
        fn = input.get('fn')

        
        assert fn in self.module.whitelist or fn in c.whitelist, f"Function {fn} not in whitelist"
        assert fn not in self.module.blacklist, f"Function {fn} is blacklisted" 

        # RATE LIMIT CHECKING HERE
        self.sync()

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


        # if 'fn' self.config.fn2rate:
        #     # if the function is in the weight map, we need to check the weight
        #     # get the weight of the function
        #     weight = self.fn2weight.get(fn, 1)
        #     # multiply the rate limit by the weight
        #     rate_limit = rate_limit * weight

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
        client = c.connect(server_name, key='fam')
        for n in range(10):
            c.print(client.info(timeout=4))
        c.kill(server_name)
        return {'name': server_name, 'module': module, 'client': client}

            


