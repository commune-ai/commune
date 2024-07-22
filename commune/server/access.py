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
                state_path = f'state_path', # the path to the state
                refresh: bool = False,
                stake_from_multipler: int =  1.0, # 1 call per every N tokens staked per timescale
                max_staleness: int =  60, #  1000 seconds per sync with the network
                **kwargs):
        
        self.set_config(locals())
        self.period = self.timescale_map[self.config.timescale]
        self.user_module = c.module("user")()
        self.state_path = self.resolve_path(state_path)
        if refresh:
            self.rm_state()
        if isinstance(module, str):
            module = c.module(module)()
        self.module = module
        c.thread(self.run_loop)

        self.state = {'sync_time': 0, 
                      'stake_from': {}, 
                      'fn_info': {}}


    def get_rate_limit(self, fn, address):
        # stake rate limit
        stake = self.state['stake'].get(address, 0)
        stake_from = self.state['stake_from'].get(address, 0)
        stake = (stake_from * self.config.stake_from_multipler) + stake
        fn_info = self.state.get('fn_info', {}).get(fn, {'stake2rate': self.config.stake2rate, 'max_rate': self.config.max_rate})
        rate_limit = (stake / fn_info['stake2rate']) # convert the stake to a rate

        return rate_limit

    whitelist = ['info', 'verify', 'rm_state']
    @c.endpoint(cost=1)
    def forward(self, fn: str = 'info' ,  address=None, input:dict = None) -> dict:
        """
        input:
            {
                args: list = [] # the arguments to pass to the function
                kwargs: dict = {} # the keyword arguments to pass to the function
                timestamp: int = 0 # the timestamp to use
                address: str = '' # the address to use
            }

        Rules:
        1. Admins have unlimited access to all functions, do not share your admin keys with anyone
            - Admins can add and remove other admins 
            - to check admins use the is_admin function (c.is_admin(address) or c.admins() for all admins)
            - to add an admin use the add_admin function (c.add_admin(address))
        2. Local keys have unlimited access but only to the functions in the whitelist
        returns : dict
        """
        input = input or {}
        address = input.get('address', address)
        print(f'address={address}')
        if c.is_admin(address):
            return {'success': True, 'msg': f'is verified admin'}
        assert fn in self.module.whitelist , f"Function {fn} not in whitelist={self.module.whitelist}"
        is_private_fn = bool(fn.startswith('__') or fn.startswith('_'))
        assert not is_private_fn, f'Function {fn} is private'
        # CHECK IF THE ADDRESS IS A LOCAL KEY
        is_local_key = address in self.address2key
        is_user = c.is_user(address)
        if is_local_key or is_user:
            return {'success': True, 'msg': f'address {address} is a local key or user, so it has unlimited access'}
        rate_limit = self.get_rate_limit(fn, address)
        # check if the user has exceeded the rate limit
        user_info = self.state.get('user_info', {}).get(address, {})
        user_info['timestamp'] = c.time()
        user_info['fn_info'] = user_info.get('fn_info', {})
        user_fn_info = user_info['fn_info'].get(fn, {"timestamp": c.time(), 'count': 0})
        reset_count = bool((c.time() - user_fn_info['timestamp']) > self.period)
        if reset_count:
            user_fn_info['count'] = 0
        else:
            user_fn_info['count'] += 1
        assert user_fn_info['count'] <= rate_limit, f'rate limit exceeded for {fn}'

        user_info['fn_info'][fn] = user_fn_info
        self.state['user_info'] = self.state.get('user_info', {})
        self.state['user_info'][address] = user_info
        # check the rate limit
        return user_info

    verify = forward

    def rm_state(self):
        self.put(self.state_path, {})
        return {'success': True, 'msg': f'removed {self.state_path}'}

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
        c.namespace(max_age=self.config.max_staleness)
        response = { 
                    'path': self.state_path,
                    'max_staleness':  self.config.max_staleness,
                    'network': network,
                    'netuid': netuid,
                    'staleness': int(staleness), 
                    }
        
        if staleness < self.config.max_staleness:
            response['msg'] = f'synced too earlly waiting {self.config.max_staleness - staleness} seconds'
            return response
        else:
            response['msg'] =  'Synced with the network'
            response['staleness'] = 0
        self.subspace = c.module('subspace')(network=network)
        state['stake'] = self.subspace.stakes(fmt='j', netuid=netuid, update=update, max_age=self.config.max_staleness)
        state['stake_from'] = self.subspace.stake_from(fmt='j', netuid=netuid, update=update, max_age=self.config.max_staleness)
        self.state = state
        self.put(self.state_path, self.state)
        return response


if __name__ == '__main__':
    Access.run()

            
