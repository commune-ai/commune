import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

class Dashboard(c.Module):
    
    def __init__(self, netuid = 0, network = 'main', ): 
        st.set_page_config(layout="wide")
        self.set_config(locals())
        self.st = c.module('streamlit')()
        self.st.load_style()
        self.load_state(update=False)

    def sync(self):
        return self.load_state(update=True)
    
    def load_state(self, update:bool=False, netuid=0, network='main'):
        self.key = c.get_key()

        t = c.timer()

        self.subspace = c.module('subspace')()
        self.state = self.subspace.state_dict(update=update)
        c.print(f'Loaded State in {t.seconds} seconds')
        self.netuid = 0
        self.subnets = self.state['subnets']
        self.subnet = 'commune'

        self.subnet2info = {s['netuid']: s for s in self.subnets}
        self.subnet2netuid = {s['name']: s['netuid'] for s in self.subnets}
        self.subnet_names = [s['name'] for s in self.subnets]


        self.modules = self.state['modules'][self.netuid]
        self.name2key = {k['name']: k['key'] for k in self.modules}
        self.key2name = {k['key']: k['name'] for k in self.modules}

        self.keys  = c.keys()
        self.key2index = {k:i for i,k in enumerate(self.keys)}

        self.namespace = {m['name']: m['address'] for m in self.modules}
        self.module_names = [m['name'] for m in self.modules]
        self.block = self.state['block']
        for i, m in enumerate(self.modules):
            self.modules[i]['stake'] = self.modules[i]['stake']/1e9
            self.modules[i]['emission'] = self.modules[i]['emission']/1e9


        self.key_info = {
            'ss58_address': self.key.ss58_address,
            'balance': self.state['balances'].get(self.key.ss58_address,0),
            'stake_to': self.state['stake_to'][self.netuid].get(self.key.ss58_address,{}),
            
        }

        self.key_info['balance']  = self.key_info['balance']/1e9
        self.key_info['stake_to'] = {k:v/1e9 for k,v in self.key_info['stake_to']}
        # convert keys to names 
        for k in ['stake_to']:
            self.key_info[k] = {self.key2name.get(k, k): v for k,v in self.key_info[k].items()}

        total_balance = sum(self.state['balances'].values())
        self.subnet_info = self.state['subnets'][0]
        balances = self.state['balances']
        self.total_balance = sum(balances.values())/1e9
        for k in ['stake', 'emission', 'min_stake']:
            self.subnet_info[k] = self.subnet_info[k]/1e9
    def select_key(self,):
        with st.expander('Select Key', expanded=True):
            key = 'module'
            key = st.selectbox('Select Key', self.keys, index=self.key2index[key])
            self.key =  c.get_key(key)
            if self.key.path == None:
                self.key.path = key
            self.key_info_dict = {
                'balance': self.stats
            }

            st.write('Address: ', self.key.ss58_address)
            stake = sum([v for v in self.key_info.get('stake_to', {}).values()])
            st.write('Stake', stake )
            st.write('Balance', self.key_info.get('balance', 0))

    def create_key(self):
        with st.expander('Create Key', expanded=False):                
            new_key = st.text_input('Name of Key', '', key='create')
            create_key_button = st.button('Create Key')
            if create_key_button and len(new_key) > 0:
                c.add_key(new_key)
                key = c.get_key(new_key)

    def rename_key(self):
        with st.expander('Rename Key', expanded=False):    
            old_key = st.selectbox('Select Key', self.keys, index=self.key2index[self.key.path], key='select old rename key')           
            new_key = st.text_input('New of Key', '', key='rename')
            rename_key_button = st.button('Rename Key')
            replace_key = st.checkbox('Replace Key')
            if rename_key_button and len(new_key) > 0:
                if c.key_exists(new_key) and not replace_key:
                    st.error('Key already exists')
                c.rename_key(old_key,new_key)
                key = c.get_key(new_key)
    
    def remove_key(self):       
        with st.form(key='Remove Key'):            
            rm_keys = st.multiselect('Select Key(s) to Remove', self.keys, [], key='rm_key')
            rm_key_button = st.form_submit_button('Remove Key')
            if rm_key_button:
                c.rm_keys(rm_keys)

    def key_dashboard(self):
        # self.select_key()
        self.create_key()
        self.rename_key()
        self.remove_key()

    def subnet_management(self):
        with st.expander('Subnet', expanded=True):
        
            subnets = self.subspace.subnets()
            if len(subnets) == 0:
                subnets = [self.default_subnet]
            else:
                subnets = [n['name'] for n in subnets]
            subnet2index = {n:i for i,n in enumerate(subnets)}
            subnet = st.selectbox('Subnet', subnets, index=subnet2index['commune'])
            self.netuid = self.subspace.subnet2netuid(subnet)
            

    def select_network(self):            
        with st.expander('Network', expanded=True):
            st.write('# Network')
            key2index = {k:i for i,k in enumerate(self.keys)}
            self.subnet = st.selectbox(' ', self.subnet_names, 0, key='Select Subnet')
            self.netuid = self.subnet2netuid[self.subnet]
            sync = st.button('Sync')
            if sync:
                self.sync()

    def sidebar(self):
        with st.sidebar:
            self.select_key()
            self.select_network()
            st.write(self.subnet_info)

    def get_module_stats(self, modules):
        df = pd.DataFrame(modules)
        del_keys = ['stake_from', 'stake_to', 'key']
        for k in del_keys:
            del df[k]
        return df

    # for k in ['emission', 'stake']:
    #     df[k] = df[k].apply(lambda x: c.round_decimals(self.subspace.format_amount(x, fmt='j'), 2))

    # df.sort_values('incentive', inplace=True, ascending=False)
    # df = df[:max_rows]
    # st.write(df)
    # st.dataframe(df, width=1000)
    # # BAR OF INCENTIVES
    # options = ['emission', 'incentive', 'dividends']
    # selected_keys = st.multiselect('Select Columns', options, options, key='stats')

    # for i, k in enumerate(selected_keys):
    #     cols = st.columns(2)

    #     fig = px.line(df, y=k, x= 'name', title=f'{k.upper()} Per Module')
    #     cols[0].plotly_chart(fig)
    #     fig = px.histogram(df, y=k, title=f'{k.upper()} Distribution')
    #     cols[1].plotly_chart(fig)
    def auth(self, users):

        # Creating the authenticator object
        authenticator = stauth.Authenticate(
            users['credentials'],
            users['cookie']['name'], 
            users['cookie']['key'], 
            users['cookie']['expiry_days'],
            users['preauthorized']
        )

        name, authentication_status, username = authenticator.login('Login', 'main')
        if st.session_state.authentication_status == True:
            with st.sidebar:
                st.subheader(f'Welcome *{st.session_state.name}*')
            authenticator.logout('Logout', 'sidebar')
        if st.session_state.authentication_status == False:
            st.error('Username/password is incorrect')
        if st.session_state.authentication_status != True:
            with st.expander("Forgot password?"):
                # Creating a forgot password widget
                try:
                    username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
                    if username_forgot_pw:
                        st.success('New password sent securely')
                        st.text(random_password)
                        # Random password to be transferred to user securely
                    elif username_forgot_pw != None:
                        st.error('Username not found')
                except Exception as e:
                    st.error(e)

            with st.expander("Forgot username?"):
                # Creating a forgot username widget
                try:
                    username_forgot_username, email_forgot_username = authenticator.forgot_username('Forgot username')
                    if username_forgot_username:
                        st.success('Username sent securely')
                        # Username to be transferred to user securely
                        st.text(username_forgot_username)
                    elif username_forgot_username != None:
                        st.error('Email not found')
                except Exception as e:
                    st.error(e)

        return authenticator


    def profile(self, authenticator): 
        st.subheader(f'Your username: *{st.session_state.username}*')
        st.text(f'Your name: {st.session_state.name}')
        with st.expander("Update userdetail"): 
        # Creating an update user details widget
            try:
                if authenticator.update_user_details(st.session_state["username"], 'Update user details'):
                    st.success('Entries updated successfully')
            except Exception as e:
                st.error(e)

        with st.expander("Reset password"): 
            # Creating a password reset widgeta
            try:
                if authenticator.reset_password(st.session_state["username"], 'Reset password'):
                    st.success('Password modified successfully')
            except Exception as e:
                st.error(e)

    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        self = cls()

        current_directory = os.path.dirname(os.path.abspath(__file__))
        # Loading admin auth file
        with open(current_directory+'/admin_auth.yaml') as file:
            users = yaml.load(file, Loader=SafeLoader)

        authenticator = self.auth(users)

        if st.session_state.authentication_status:
            self.sidebar()
            
            tabs = st.tabs(['MY SPACE','GLOBAL SPACE', 'PLAYGROUND', 'CHAT', 'PROFILE']) 
            with tabs[0]:
                self.local_dashboard()
            with tabs[1]:   
                self.subspace_dashboard()
            # with tabs[2]:
            #     self.playground_dashboard()
            with tabs[4]:
                self.profile(authenticator)

        # Saving users file
        with open(current_directory + '/admin_auth.yaml', 'w') as file:
            yaml.dump(users, file, default_flow_style=False)

    def subnet_dashboard(self):
        st.write('# Subnet')
        df = pd.DataFrame(self.subnets)

        if len(df) > 0:
            fig = px.pie(df, values='stake', names='name', title='Subnet Balances')
            st.plotly_chart(fig)

        for subnet in self.subnets:
            subnet = subnet.pop('name', None)
            with st.expander(subnet, expanded=True):
                st.write(subnet)
        
        # convert into metrics
        
    def transfer_dashboard(self):
        with st.expander('Transfer', expanded=True):
            amount = st.number_input('amount', 0.0, 10000000.0, 0.0, 0.1)
            to_address = st.text_input('dest', '')
            transfer_button = st.button('Transfer')
            if transfer_button:
                kwargs = {
                    'dest': to_address,
                    'amount': amount,
                    'key': self.key,
                }
                self.subspace.transfer(**kwargs)




    def stake_dashboard(self):
        cols = st.columns(2)
        with st.expander('Stake', expanded=True):

            with st.form(key='stake'):

                amounts = st.slider('Stake Amount', 0.0,  float(self.key_info['balance']), 0.1)            
                modules = st.multiselect('Module', self.module_names, [])
                transfer_button = st.form_submit_button('STAKE')

                if transfer_button:
                    kwargs = {
                        'amounts': amounts,
                        'modules': modules,
                        'key': self.key,
                    }

                    response = self.subspace.multistake(**kwargs)
                    st.status(response)


    def unstake_dashboard(self):

        with st.expander('UnStake', expanded=True):
            module2stake_from_key = self.subspace.get_staked_modules(self.key, fmt='j')
            modules = list(self.key_info['stake_to'].keys())
            amount = st.number_input('Unstake Amount',0.0)
            modules = st.multiselect('Module', modules, [])

            unstake_button = st.button('UNSTAKE')
            if unstake_button:
                kwargs = {
                    'amounts': amount,
                    'modules': modules,
                    'key': self.key,
                }
                st.write(kwargs)
                self.subspace.multiunstake(**kwargs)


            

    def archive_dashboard(self):
        # self.register_dashboard(expanded=False)
        netuid = 0 
        df = self.get_module_stats(self.modules)
        archive_history = self.subspace.archive_history(lookback_hours=24, n=100, update=True)
        df = c.df(archive_history[1:])
        df['block'] = df['block'].astype(int)


        df['dt'] = pd.to_datetime(df['dt'])
        df.sort_values('block', inplace=True)
        df.reset_index(inplace=True)
        st.write(df)
        # df= df[df['market_cap'] < 1e9]


        fig = px.line(df, x='block', y='market_cap', title='Archive History')

        block2path= {b:df['path'][i] for i,b in enumerate(df['block'])}
        blocks = list(block2path.keys())
        paths = list(block2path.values())
        block = st.selectbox('Block', blocks, index=0)
        path = block2path[block]
        state = c.get(path)
        modules = state['modules'][netuid]
        for i in range(len(modules)):
            for k in ['stake_to', 'stake_from', 'key', 'address']:
                del modules[i][k]
            for k in ['emission', 'stake', 'balance']:
                modules[i][k] = modules[i][k]/1e9
        df = pd.DataFrame(modules)

        st.write(df)
        subnet_df = pd.DataFrame(state['subnets'])
        st.write(subnet_df)
        # st.write(state)

        st.write(fig)
        # options = ['emission', 'incentive', 'dividends', 'stake']
        # y = st.selectbox('Select Columns', options, 0)
        # # filter by stake > 1000

        # df = df[df['stake'] > 10**9]
        # histogram = px.histogram(df, x=y, title='My Modules')

        # st.write(histogram)
       
    def subspace_dashboard(self):
        # pie map of stake

        st.write(self.modules_dashboard())
        # remove the 
        st.write('# Wallet')
        self.register_dashboard()
        self.stake_dashboard()
        self.unstake_dashboard()
        self.transfer_dashboard()
        # else:
        #     # with emoji
        #     st.error('Please Register Your Key')

        fig = px.pie(values=list(self.key_info['stake_to'].values()), names=list(self.key_info['stake_to'].keys()), title='Stake To')
        st.plotly_chart(fig)

    


    def validator_dashboard(self):
        pass
    def register_dashboard(self, expanded=True, prefix= None, form = True ):


        if expanded : 
            with st.expander('Register Module', expanded=True):
                return self.register_dashboard(prefix=prefix, expanded=False)
        modules = c.modules(prefix)
        self.st.line_seperator()
        cols = st.columns([2,2,2])

        with st.form(key='register'):
            module  = cols[0].selectbox('Select A Module', modules, 0)
            tag = cols[1].text_input('tag', c.random_word(n=2), key=f'tag.register')
            stake = cols[2].number_input('stake', 0.0, 10000000.0, 0.1, key=f'stake.{prefix}.register')
            n = st.number_input('Number of Replicas', 1, 30, 1, 1, key=f'n.{prefix}.register')
            # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
            register = st.form_submit_button('Register')
            # fn = st.selectbox('Select Function', fn2index['__init__'], key=f'fn.{prefix}')
            kwargs = self.function2streamlit(module=module, fn='__init__', salt='register')
            self.st.line_seperator()

            if 'None' == tag:
                tag = None
                
                
            if 'tag' in kwargs:
                kwargs['tag'] = tag

            if register:
                try:
                    if n > 1:
                        tags = [f'{tag}.{i}' if tag != None else str(i) for i in range(n)]
                    else:
                        tags = [tag]

                    module_name = module
                    module = c.module(module)

                    for tag in tags:
                        st.write(f'Registering {module_name} with tag {tag}, {kwargs}')
                        response = module.register(tag=tag, subnet= self.subnet, stake=stake)
                        st.write(response)
                except Exception as e:
                    e = c.detailed_error(e)
                    response = {'success': False, 'message': e}
                    raise e
                if response['success']:
                    st.success('Module Registered')
                else:
                    st.error(response['message'])
                

    def serve_dashboard(self , ):

        modules = c.modules()
        self.st.line_seperator()
        cols = st.columns([2,2,1])

        with st.form(key='serve'):
            module  = cols[0].selectbox('Select A Module', modules, 0, key=f'serve.module')
            tag = cols[1].text_input('tag', '', key=f'serve.tag.{module}')
            n = cols[2].number_input('Number of Replicas', 1, 30, 1, 1, key=f'serve.n.{module}')

            # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
            tag = None if tag == '' else tag
                        
            # fn = st.selectbox('Select Function', fn2index['__init__'], key=f'fn.{prefix}')
            
            
            kwargs = self.function2streamlit(module=module, fn='__init__' )


            serve = st.form_submit_button('Serve')

            if 'None' == tag:
                tag = None
                
                
            if 'tag' in kwargs:
                kwargs['tag'] = tag

            network = 'local'
            if serve:
                for i in range(n):
                    try:
                        if tag != None:
                            s_tag = f'{tag}.{i}'
                        else:
                            s_tag = str(i)
                        response = c.module(module).serve( kwargs = kwargs, tag=s_tag, network=network)
                    except Exception as e:
                        e = c.detailed_error(e)
                        response = {'success': False, 'message': e}
        
                    if response['success']:
                        st.write(response)
                    else:
                        st.error(response)
                



    def modules_dashboard(self):
        import pandas as pd
        # search  for all of the modules with yaml files. Format of the file
        search = st.text_input('Search', '')
        df = None
        
        self.searched_modules = [m for m in self.modules if search in m['name'] or search == '']
        df = pd.DataFrame(self.searched_modules)
        if len(df) == 0:
            st.error(f'{search} does not exist {c.emoji("laughing")}')
            return
        else:
            st.success(f'{c.emoji("dank")} {len(df)} modules found with {search} in the name {c.emoji("dank")}')
            del df['stake_from']
            st.write(df)
            with st.expander('Historam'):
                key = st.selectbox('Select Key', ['incentive',  'dividends', 'emission'], 0)
                
                self.st.run(df)
                fig = px.histogram(
                    x = df[key].to_list(),
                )

                st.plotly_chart(fig)



    def local_dashboard(self):
        import pandas as pd
        # search  for all of the modules with yaml files. Format of the file
        df = None
        cols = st.columns(2)
        network = st.text_input('Network', 'local')
        with st.expander('Serve', expanded=True):
            self.serve_dashboard()



    def playground_dashboard(self):
        network = st.text_input('Network', 'local',key='playground.network')
        update = st.button('Update')
        servers_info = c.servers_info( network=network, update=update)
        server2info = {s['name']: s for s in servers_info if s != None}
        servers = list(server2info.keys())
        server = st.selectbox('Select Server', servers, 0)
        info = server2info[server]
        server_address = info['address']
        st.write('Name: ', info['name'])
        schema = info['schema']
        buttons = {}
        for fn, fn_schema in schema.items():
            with st.expander(fn, expanded=False):
                with st.form(key=f'{fn}.form'):
                    kwargs = self.function2streamlit(fn=fn, fn_schema=fn_schema)
                    buttons[fn] = st.form_submit_button(fn)
                    if buttons[fn]:
                        st.write(kwargs)

                        result = c.submit(c.call, args=[server_address, fn], timeout=10, kwargs=kwargs, return_future=False)[0]
                        st.text_area('Result', result)


        
    def remote_dashboard(self):
        st.write('# Remote')


      
    @classmethod
    def function2streamlit(cls, 
                           module = None,
                           fn:str = '__init__',
                           fn_schema = None, 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                           salt = None,
                            mode = 'pm2'):
        
        key_prefix = f'{module}.{fn}'
        if salt != None:
            key_prefix = f'{key_prefix}.{salt}'
        if module == None:
            module = cls
            
        elif isinstance(module, str):
            module = c.module(module)
        extra_defaults = {} if extra_defaults is None else extra_defaults
        kwargs = {}

        if fn_schema == None:

            fn_schema = module.schema(defaults=True, include_parents=True)[fn]
            if fn == '__init__':
                config = module.config(to_munch=False)
                extra_defaults = config
            fn_schema['default'].pop('self', None)
            fn_schema['default'].pop('cls', None)
            fn_schema['default'].update(extra_defaults)
            fn_schema['default'].pop('config', None)
            fn_schema['default'].pop('kwargs', None)
            
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        if cols == None:
            cols = [1 for i in list(range(int(len(fn_schema['input'])**0.5)))]
        if len(cols) == 0:
            return kwargs
        cols = st.columns(cols)

        for i, (k,v) in enumerate(fn_schema['default'].items()):
            
            optional = fn_schema['default'][k] != 'NA'
            fn_key = k 
            if fn_key in skip_keys:
                continue
            if k in fn_schema['input']:
                k_type = fn_schema['input'][k]
                if 'Munch' in k_type or 'Dict' in k_type:
                    k_type = 'Dict'
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            

            col_idx = col_idx % (len(cols))
            kwargs[k] = cols[col_idx].text_input(fn_key, v, key=f'{key_prefix}.{k}')
        kwargs = cls.process_kwargs(kwargs, fn_schema)       
        
        return kwargs

   
    @classmethod
    def process_kwargs(cls, kwargs:dict, fn_schema:dict):
        
        for k,v in kwargs.items():
            if v == 'None':
                v = None
            
            if isinstance(v, str):
                if v.startswith('[') and v.endswith(']'):
                    if len(v) > 2:
                        v = eval(v)
                    else:
                        v = []

                elif v.startswith('{') and v.endswith('}'):

                    if len(v) > 2:
                        v = c.jload(v)
                    else:
                        v = {}               
                elif k in fn_schema['input'] and fn_schema['input'][k] == 'str':
                    if v.startswith("f'") or v.startswith('f"'):
                        v = c.ljson(v)
                    else:
                        v = v

                elif fn_schema['input'][k] == 'float':
                    v = float(v)

                elif fn_schema['input'][k] == 'int':
                    v = int(v)

                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                elif v in ['True', 'False']:
                    v = eval(v)
                elif c.is_number(v):
                    v = eval(v)
                else:
                    v = v
            
            kwargs[k] = v

        return kwargs
    
if __name__ == '__main__':
    Dashboard.run()