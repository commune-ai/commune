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
            assert fn in self.module.whitelist , f"Function {fn} not in whitelist"
            assert fn not in self.module.blacklist, f"Function {fn} is blacklisted"

        # RATE LIMIT CHECKING HERE
        num_requests = self.requests.get(address, 0) + 1
        rate_limit = self.config.role2rate.get(role, 0)
        if rate_limit >= 0:
            assert self.requests[address] < self.module.rate_limit, f"Rate limit exceeded for {address}"
        self.requests[address] = num_requests


        return input



