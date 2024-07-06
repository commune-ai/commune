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
                key2rate = {}, # the rate limit for each key
                state_path = f'state_path', # the path to the state
                refresh: bool = False,
                max_age = 30, # max age of the state in seconds
                max_staleness: int =  60, #  1000 seconds per sync with the network
                **kwargs):
        
        self.set_config(locals())
        self.user_module = c.module("user")()
        self.state_path = self.resolve_path(state_path)
        if refresh:
            self.rm_state()
        self.last_time_synced = c.time()
        if isinstance(module, str):
            module = c.module(module)()
        self.module = module
        c.thread(self.run_loop)

        self.state = {'sync_time': 0, 
                      'stake_from': {}, 
                      'fn_info': {}}
    
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
        self.address2key = c.address2key()
        response = { 
                    'path': self.state_path,
                    'max_staleness':  self.config.max_staleness,
                    'network': network,
                    'netuid': netuid,
                    'staleness': int(staleness), 
                    'datetime': c.datetime()}
        
        if staleness < self.config.max_staleness:
            response['msg'] = f'synced too earlly waiting {self.config.max_staleness - staleness} seconds'
            return response
        else:
            response['msg'] =  'Synced with the network'
            response['staleness'] = 0
        self.subspace = c.module('subspace')(network=network)
        state['stake'] = self.subspace.stakes(fmt='j', netuid=netuid, update=update, max_age=self.config.max_staleness)
        state['stake_from'] = {}
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
        assert address, f'address not in input or as an argument {input}'
        fn = input.get('fn', fn)

        # ONLY THE ADMIN CAN CALL ANY FUNCTION, THIS IS A SECURITY FEATURE
        # THE ADMIN KEYS ARE STORED IN THE CONFIG
        if c.is_admin(address):
            return {'success': True, 'msg': f'is verified admin'}
        assert fn in self.module.whitelist , f"Function {fn} not in whitelist={self.module.whitelist}"
        if address in self.address2key:
            return {'success': True, 'msg': f'address {address} is a local key'}
        if fn.startswith('__') or fn.startswith('_'):
            return {'success': False, 'msg': f'Function {fn} is private'}
        if c.is_user(address):
            return {'success': True, 'msg': f'is verified user'}
        
        # stake rate limit
        stake = self.state['stake'].get(address, 0)
        stake_from = self.state['stake_from'].get(address, 0)
        # STEP 2: CHECK THE STAKE AND CONVERT TO A RATE LIMIT
        fn_info = {'stake2rate': self.config.stake2rate, 'max_rate': self.config.max_rate}
        fn_info = self.state.get('fn_info', {}).get(fn,fn_info)
        rate_limit = (stake / fn_info['stake2rate']) # convert the stake to a rate
        rate_limit = min(rate_limit, fn_info['max_rate']) # cap the rate limit at the max rate
        rate_limit = self.config.key2rate.get(address, rate_limit)
        # NOW LETS CHECK THE RATE LIMIT
        user_info = self.state.get('user_info', {}).get(address, {})
        # check if the user has exceeded the rate limit
        og_timestamp = user_info.get('timestamp', 0)
        user_info['timestamp'] = c.time()
        user_info['time_since_called'] = user_info['timestamp'] - og_timestamp
        period = self.timescale_map[self.config.timescale]
        # update the user info
        user_info['key'] = address
        user_info['fn2requests'] = user_info.get('fn2requests', {})
        user_info['fn2requests'][fn] = user_info['fn2requests'].get(fn, 0) + 1
        user_info['stake'] = stake
        user_info['stake_from'] = stake_from
        user_info['rate'] = user_info.get('rate', 0) + 1

        # if the time since the last call is greater than the seconds in the period, reset the requests
        if user_info['time_since_called'] > period:
            user_info['rate'] = 0
        try:
            assert user_info['rate'] <= rate_limit
            user_info['success'] = True
        except Exception as e:
            user_info['error'] = c.detailed_error(e)
            user_info['success'] = False

        # store the user info into the state
        self.state['user_info'][address] = user_info
        # check the rate limit
        return user_info

    verify = forward

    def rm_state(self):
        self.put(self.state_path, {})
        return {'success': True, 'msg': f'removed {self.state_path}'}

if __name__ == '__main__':
    Access.run()

            
