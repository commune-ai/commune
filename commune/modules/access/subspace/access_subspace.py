import commune as c





class AccessSubspace(c.Module):
    sync_time = 0
    def __init__(self, module, **kwargs):
        config = self.set_config(kwargs)
        self.module = module
        self.sync()
        self.user_info = {}


    def sync(self):
        sync_time  = c.time() - self.sync_time
        if sync_time >  self.config.sync_interval :
            self.sync_time = c.time()
            self.subspace = c.module('subspace')(network=self.config.network, netuid=self.config.netuid)
            self.stakes = self.subspace.stakes(fmt='j')
        else:
            return
        

    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}
    def verify(self, input:dict) -> dict:

        address = input['address']
        if c.is_admin(address):
            return input
        # if not an admin address, we need to check the whitelist and blacklist
        fn = input.get('fn')
        assert fn in self.module.whitelist or fn in c.helper_whitelist, f"Function {fn} not in whitelist"
        assert fn not in self.module.blacklist, f"Function {fn} is blacklisted" 

        # RATE LIMIT CHECKING HERE
        self.sync()

        is_registered = bool( address in self.stakes)

        stake = self.stakes.get(address, 0)
        # get the rate limit for the function
        if fn in self.config.fn2rate:
            rate = self.config.fn2rate[fn]
        else:
            rate = self.config.rate
        rate_limit = (stake / self.config.stake2rate)
        rate_limit = rate_limit / self.timescale_map[self.config.timescale]

        if is_registered:
            rate_limit = rate_limit + self.config.rate


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
        module = c.serve('module', server_name=server_name, wait_for_server=True)
        client = c.connect(server_name, key='vali::var9')
        for n in range(10):
            c.sleep(1)
            c.print(client.info(timeout=4))
        c.kill(server_name)
        return {'name': server_name, 'module': module, 'client': client}

            


