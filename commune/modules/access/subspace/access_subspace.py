import commune as c

class AccessSubspace(c.Module):
    sync_time = 0
    def __init__(self, module, **kwargs):
        config = self.set_config(kwargs)
        self.module = module
        self.sync()
        self.requests = {}


    def sync(self):
        sync_time  = c.time() - self.sync_time
        if sync_time >  self.config.sync_interval :
            self.sync_time = c.time()
            self.subspace = c.module('subspace')(network=self.config.network, netuid=self.config.netuid)
            self.stakes = self.subspace.stakes()
        else:
            return
        

    def verify_staleness(self, input:dict) -> dict:

        # here we want to verify the data is signed with the correct key
        request_staleness = c.timestamp() - input['data'].get('timestamp', 0)
        assert request_staleness < self.config.max_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
        

    def verify(self, input:dict) -> dict:

        self.verify_staleness(input)

        address = input['address']
        fn = input.get('fn')
        if c.is_admin(address):
            requests = self.requests.get(address, 0) + 1
            return input

        # if not an admin address, we need to check the whitelist and blacklist
        assert fn in self.module.whitelist or fn in c.helper_whitelist, f"Function {fn} not in whitelist"
        assert fn not in self.module.blacklist, f"Function {fn} is blacklisted" 

        # RATE LIMIT CHECKING HERE
        self.sync()
        stake = self.stakes.get(address, 0)
        rate_limit = stake / self.config.stake2rate
        requests = self.requests.get(address, 0) + 1
        assert requests < rate_limit, f"Rate limit exceeded for {address}, {requests} > {rate_limit} with {stake} stake and stake2rate of {self.config.stake2rate}"
        self.requests[address] = requests

        return input

