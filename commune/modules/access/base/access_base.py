import commune as c

class AccessBase(c.Module):
    def __init__(self, module, **kwargs):
        
        config = self.set_config(kwargs)
        self.module = module
        self.max_staleness = config.max_staleness
        self.requests = {}

    def verify(self, input:dict) -> dict:


        # here we want to verify the data is signed with the correct key
        request_timestamp = input['data'].get('timestamp', 0)
        request_staleness = c.timestamp() - request_timestamp
        assert request_staleness < self.max_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
        address = input.get('address', None)
        fn = input.get('fn', None)

        role = c.get_role(address)
        if bool(role == 'admin'):
            # this is an admin address, so we can pass
            pass
        else:
            # if not an admin address, we need to check the whitelist and blacklist

            if fn not in self.module.whitelist or fn in self.module.blacklist:
                c.print(f"Function {fn} not in whitelist (or is blacklisted)", color='red')
                c.print(f"Whitelist: {self.module.whitelist}", color='red')
                c.print(f"Blacklist: {self.module.blacklist}", color='red')
                return False

        # RATE LIMIT CHECKING HERE
        num_requests = self.requests.get(address, 0) + 1
        rate_limit = self.config.role2rate.get(role, 0)
        if rate_limit >= 0:
            if  self.requests[address] > self.module.rate_limit:
                c.print(f"Rate limit exceeded for {address}", color='red')
                return False 
        self.reqc uests[address] = num_requests


        return True



