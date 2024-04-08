import commune as c
from typing import *


class Access(c.Module):

    sync_time = 0
    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400, 'minute': 60, 'second': 1}

    def __init__(self, 
                module : Union[c.Module, str] = None, # the module or any python object
                network: str =  'main', # mainnet
                netuid: int = 0, # subnet id
                timescale:str =  'min', # 'sec', 'min', 'hour', 'day'
                stake2rate: int =  100.0,  # 1 call per every N tokens staked per timescale
                max_rate: int =  1000.0, # 1 call per every N tokens staked per timescale
                role2rate: dict =  {}, # role to rate map, this overrides the default rate,
                state_path = f'state_path', # the path to the state
                refresh: bool = False,
                max_age = 600, # max age of the state in seconds
                sync_interval: int =  60, #  1000 seconds per sync with the network

                **kwargs):
        
        self.set_config(locals())
        self.user_module = c.module("user")()
        self.address2key = c.address2key()
        self.set_module(module)
        self.state_path = state_path
        if refresh:
            self.rm_state()
        self.last_time_synced = c.time()
        self.state = {'sync_time': 0, 
                      'stake_from': {}, 
                      'role2rate': role2rate, 
                      'fn_info': {}}
        
        c.thread(self.run_loop)


    def set_module(self, module: c.Module):
        module = module or c.module('module')()
        if isinstance(module, str):
            module = c.module(module)()
        self.module = module
        return {'success': True, 'msg': f'set module to {module}'}

    def run_loop(self):
        while True:
            try:
                r = self.sync_network()
            except Exception as e:
                r = c.detailed_error(e)
            c.print(r)
            c.sleep(self.config.sync_interval)

    def sync_network(self):
        state = self.get(self.state_path, {}, max_age=self.config.sync_interval)
        time_since_sync = c.time() - state.get('sync_time', 0)

        if time_since_sync > self.config.sync_interval:
            self.subspace = c.module('subspace')(network=self.config.network)
            state['stakes'] = self.subspace.stakes(fmt='j', netuid='all', update=False, max_age=self.config.max_age)
            state['sync_time'] = c.time()
            self.state = state
            self.put(self.state_path, self.state)
            
            c.print(f'🔄 Synced {self.state_path} at {state["sync_time"]}... 🔄\033', color='yellow')
            
            response = {'success': True, 
                        'msg': f'synced {self.state_path}', 
                        'until_sync': int(self.config.sync_interval - time_since_sync),
                        'time_since_sync': int(time_since_sync)}
        else:
            response = {'success': True, 
                        'msg': f'not syncing {self.state_path}', 
                        'until_sync': int(self.config.sync_interval - time_since_sync),
                        'time_since_sync': int(time_since_sync)}
        return response

    def verify(self, input:dict) -> dict:
        """
        input : dict 
            fn : str
            address : str

        returns : dict
        """
        fn = input['fn']
        address = input['address']
        if c.is_admin(address):
            return {'success': True, 'msg': f'is verified admin'}
        

        whitelist =  list(set(self.module.whitelist + c.whitelist))
        blacklist =  self.module.blacklist

        assert fn in whitelist , f"Function {fn} not in whitelist={whitelist}"
        assert fn not in blacklist, f"Function {fn} is blacklisted" 

        if c.is_user(address):
            return {'success': True, 'msg': f'is verified user'}

        if fn.startswith('__') or fn.startswith('_'):
            return {'success': False, 'msg': f'Function {fn} is private'}

        if address in self.address2key:
            return {'success': True, 'msg': f'address {address} is in the whitelist'}
        
        current_time = c.time()

        # sync of the state is not up to date 
        self.sync_network()

        # get the rate limit for the user
        role2rate = self.state.get('role2rate', {})

        # get the role of the user
        role = self.user_module.get_role(address) or 'public'
        rate_limit = role2rate.get(role, 0)

        # stake rate limit
        stake = self.state.get('stake_from', {}).get(address, 0)
        # we want to also know if the user has been staked from
        stake_from = self.state.get('stake_from', {}).get(address, 0)
        # STEP 1:  FIRST CHECK THE WHITELIST AND BLACKLIST

        total_stake_score = stake 

        # STEP 2: CHECK THE STAKE AND CONVERT TO A RATE LIMIT
        default_fn_info = {'stake2rate': self.config.stake2rate, 'max_rate': self.config.max_rate}
        self.state['fn_info'] = self.state.get('fn_info', {})
        fn2info = self.state['fn_info'].get(fn,default_fn_info)
        stake2rate = fn2info.get('stake2rate', self.config.stake2rate)
        
        rate_limit = (total_stake_score / stake2rate) # convert the stake to a rate


        # STEP 3: CHECK THE MAX RATE
        max_rate = fn2info.get('max_rate', self.config.max_rate)
        rate_limit = min(rate_limit, max_rate) # cap the rate limit at the max rate
        
        # NOW LETS CHECK THE RATE LIMIT
        self.state['user_info'] = self.state.get('user_info', {})
        user_info = self.state['user_info'].get(address, {})

        # check if the user has exceeded the rate limit
        time_since_called = current_time - user_info.get('timestamp', 0)
        period = self.timescale_map[self.config.timescale]
        # if the time since the last call is greater than the seconds in the period, reset the requests
        if time_since_called > period:
            user_info['rate'] = 0
        try:
            assert user_info['rate'] <= rate_limit
            user_info['success'] = True
        except Exception as e:
            user_info['error'] = c.detailed_error(e)
            user_info['success'] = False
       
        # update the user info
        user_info['rate_limit'] = rate_limit
        user_info['period'] = period
        user_info['role'] = role
        user_info['fn2requests'] = user_info.get('fn2requests', {})
        user_info['fn2requests'][fn] = user_info['fn2requests'].get(fn, 0) + 1
        user_info['timestamp'] = current_time
        user_info['stake'] = stake
        user_info['stake_from'] = stake_from
        user_info['rate'] = user_info.get('rate', 0) + 1
        user_info['timescale'] = self.config.timescale
        # store the user info into the state
        self.state['user_info'][address] = user_info
        # check the rate limit
        return user_info

    @classmethod
    def get_access_state(cls, module):
        access_state = cls.get(module)
        return access_state

    @classmethod
    def test(cls, key='vali::fam', base_rate=2):
        module = cls(module=c.module('module')(),  base_rate=base_rate)
        key = c.get_key(key)

        for i in range(base_rate*3):    
            t1 = c.time()
            c.print(module.verify(input={'address': key.ss58_address, 'fn': 'info'}))
            t2 = c.time()
            c.print(f'🚨 {t2-t1} seconds... 🚨\033', color='yellow')
    


    @classmethod
    def dashboard(cls):
        import streamlit as st
        # self = cls(module="module",  base_rate=2)
        st.title('Access')

        
        modules = c.modules()
        module = st.selectbox('module', modules)
        update = st.button('update')
        if update:
            refresh = True
        self = cls(module=module)
        state = self.state


        self.st = c.module('streamlit')()
        self.st.load_style()

        fns = self.module.fns()
        whitelist_fns = state.get('whitelist', [])
        blacklist_fns = state.get('blacklist', [])

        with st.expander('Function Whitelist/Blacklist', True):
            whitelist_fns = [fn for fn in whitelist_fns if fn in fns]
            whitelist_fns = st.multiselect('whitelist', fns, whitelist_fns )
            blacklist_fns = [fn for fn in blacklist_fns if fn in fns]
            blacklist_fns = st.multiselect('blacklist', fns, blacklist_fns )


        with st.expander('Function Rate Limiting', True):
            fn =  st.selectbox('fn', whitelist_fns,0)
            cols = st.columns([1,1])
            fn_info = state['fn_info'].get(fn, {'stake2rate': self.config.stake2rate, 'max_rate': self.config.max_rate})
            fn_info['max_rate'] = cols[1].number_input('max_rate', 0.0, 1000.0, fn_info['max_rate'])
            fn_info['stake2rate'] = cols[0].number_input('stake2rate', 0.0, fn_info['max_rate'], min(fn_info['stake2rate'], fn_info['max_rate']))
            state['fn_info'][fn] = fn_info
            state['fn_info'][fn]['stake2rate'] = fn_info['stake2rate']
            state['fn_info'] = {fn: info for fn, info in state['fn_info'].items() if fn in whitelist_fns}

            fn_info_df = []
            for fn, info in state['fn_info'].items():
                info['fn'] = fn
                fn_info_df.append(info)

            if len(fn_info_df) > 0:
                fn_info_df = c.df(fn_info_df)
                fn_info_df.set_index('fn', inplace=True)

                st.dataframe(fn_info_df, use_container_width=True)
        state['whitelist'] = whitelist_fns
        state['blacklist'] = blacklist_fns

        with st.expander('state', False):
            st.write(state)
        if st.button('save'):
            self.put(self.state_path, state)

        # with st.expander("ADD BUTTON", True):
            

        # stake_per_call_per_minute = st.slider('stake_per_call', 0, 100, 10)
        # call_weight = st.slider('call_weight', 0, 100, 10)


    
    def rm_state(self):
        self.put(self.state_path, {})
        return {'success': True, 'msg': f'removed {self.state_path}'}

    


if __name__ == '__main__':
    Access.run()

            
