import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px


class Dashboard(c.Module):
    
    def __init__(self, netuid = 0, network = 'main', ): 
        self.set_config(locals())
    
        st.set_page_config(layout="wide")
        self.st = c.module('streamlit')()
        self.st.load_style()

        # THIS IS A SPECIAL FUNCTION
        self.load_state(update=False)


    def sidebar(self):
        with st.sidebar:

            if not c.server_exists('module'):
                    c.serve(wait_for_server=True)
            self.network_dashboard(sidebar=False)
            self.servers = c.servers(network=self.network)
            self.module_name = st.selectbox('Select Server', self.servers, 0)

    

            self.module = c.connect(self.module_name, network=self.network)

            module_info_path = f'module_info/{self.module_name}'

            module_info = self.get(module_info_path, default={})
            
            if module_info == {}:
                try:
                    module_info = self.module.info(schema=True)
                except Exception as e:
                    st.error(f'Module Not Found -> {self.module_name} {e}')
                    return
            self.module_info = module_info
            self.module_schema = self.module_info['schema']
            self.put(module_info_path, self.module_info)

            self.module_functions = self.module_info['functions']
            self.module_address = self.module_info['address']



            self.fn = st.selectbox('Select Function', self.module_functions, 0)

            self.fn_path = f'{self.module_name}/{self.fn}'
            st.write(f'**address** {self.module_address}')
            with st.expander(f'{self.fn_path} playground', expanded=True):

                kwargs = self.function2streamlit(fn=self.fn, fn_schema=self.module_schema[self.fn], salt='sidebar')
                cols = st.columns([1,1])
                timeout = cols[0].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{self.fn_path}')
                call = st.button(f'Call {self.fn_path}')
                if call:
                    try:
                        response = getattr(self.module, self.fn)(**kwargs, timeout=timeout)
                    except Exception as e:
                        e = c.detailed_error(e)
                        response = {'success': False, 'message': e}
                    st.write(response)

            self.key = c.module('key.dashboard').dashboard(state=self.state)
    
            sync = st.button(f'Sync {self.network} Network'.upper(), key='sync.network')
            if sync:
                c.update_network(self.network)
                c.sync()




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


    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        self = cls()
        st.title(f'COMMUNE')
        self.sidebar()
        
        tabs = st.tabs(['CHAT', 'MODULES', 'WALLET']) 
        chat = False
        with tabs[0]:
            chat = True
        with tabs[1]: 
            self.modules_dashboard()  
        with tabs[2]:
            self.wallet_dashboard()
        if chat:
            self.chat_dashboard()

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
        with st.expander('Transfer', expanded=False):
            amount = st.number_input('amount', 0.0, 10000000.0, 0.0, 0.1)
            to_address = st.text_input('dest (s) : use , for multiple transfers', '')
            multi_transfer = False
            if ',' in to_address:
                multi_transfer = True
                to_addresses = [a.strip() for a in to_address.split(',')]

            transfer_button = st.button('Transfer')
            if transfer_button:

                if multi_transfer:
                    kwargs = {
                        'destinations': to_addresses,
                        'amounts': amount,
                        'key': self.key,
                    }
                    st.write(kwargs)
                    response = c.multitransfer(**kwargs)

                else:

                    kwargs = {
                        'dest': to_address,
                        'amount': amount,
                        'key': self.key,
                    }
                    response = c.transfer(**kwargs)

                st.write(response)





    def stake_dashboard(self):
        cols = st.columns(2)
        with st.expander('Stake', expanded=False):

            cols = st.columns(4)
            staked_modules = list(self.key_info['stake_to'].keys())
            my_staked_button = cols[3].checkbox('My Staked Modules', key='my_staked')
            search = cols[1].text_input('Search', '', key='search.stake')
            if search != '':
                staked_modules = [m for m in staked_modules if search in m]
            default_staked_modules = staked_modules if my_staked_button else []
            entire_balance = cols[2].checkbox('Entire Balance', key='entire_balance')

            



            modules = cols[2].multiselect('Module', self.module_names, default_staked_modules)


            if entire_balance:
                default_amount = c.balance(self.key.ss58_address)  / len(modules)
            else:
                default_amount = 0.0
            st.write(default_amount)
            amounts = cols[0].number_input('Stake Amount', value=default_amount,  max_value=1000000000000.0, min_value=0.0 ) # format with the value of the balance            
            stake_button = st.button('STAKE')

            if stake_button:
                kwargs = {
                    'amounts': amounts,
                    'modules': modules,
                    'key': self.key,
                }

                response = c.multistake(**kwargs)
                st.write(response)
        with st.expander('Unstake', expanded=False):
            modules = list(self.key_info['stake_to'].keys())
            cols = st.columns(3)
            cols[2].write('\n'*3)
            default_modules = [k for k,v in self.key_info['stake_to'].items() if v > amounts]
            search = cols[1].text_input('Search', '', key='search.unstake')
            amounts = cols[0].number_input('Unstake Amount',0)
            if search != '':
                modules = [m for m in modules if search in m]
            modules = cols[1].multiselect('Module', modules, default_modules)
            total_stake_amount = amounts * len(modules)
            
            st.write(f'You have {len(modules)} ready to staked for a total of {total_stake_amount} ')

            unstake_button = st.button('UNSTAKE')
            if unstake_button:
                kwargs = {
                    'amounts': amounts,
                    'modules': modules,
                    'key': self.key,
                }
                response = c.multiunstake(**kwargs)
                st.write(response)



    def archive_dashboard(self):
        # self.register_dashboard(expanded=False)
        netuid = 0 
        archive_history = c.archive_history(lookback_hours=24, n=100, update=True)
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
       
    def wallet_dashboard(self):
        # pie map of stake
        st.write('# Wallet')

        with st.expander('Key Info', expanded=False):
            st.write('ss58_address')
            st.code( self.key.ss58_address)
        
            cols = st.columns(2)
            cols[0].metric('Balance', int(self.key_info['balance']))
            cols[1].metric('Stake', int(self.key_info['stake']))
            
            cols = st.columns(2)

            values = list(self.key_info['stake_to'].values())
            labels = list(self.key_info['stake_to'].keys())

            fig = c.module('plotly').treemap(values=values, labels=labels, title='Stake To')
            # increase the width of the plot
            fig.update_layout(width=1000)
            cols[0].plotly_chart(fig)

        # bar chat of staked modules


        self.stake_dashboard()
        self.transfer_dashboard()
        self.register_dashboard()

        with st.expander('Modules', expanded=False):
            import pandas as pd
            # search  for all of the modules with yaml files. Format of the file
            search = st.text_input('Search', '')
            df = None
            self.modules = self.state['modules'][self.netuid]

        
            self.searched_modules = [m for m in self.modules if search in m['name'] or search == '']
            df = pd.DataFrame(self.searched_modules)
            if len(df) == 0:
                st.error(f'{search} does not exist {c.emoji("laughing")}')
                return
            else:
                st.success(f'{c.emoji("dank")} {len(df)} modules found with {search} in the name {c.emoji("dank")}')
                del df['stake_from']
                st.write(df)
                # with st.expander('Historam'):
                #     key = st.selectbox('Select Key', ['incentive',  'dividends', 'emission'], 0)
                    
                #     self.st.run(df)
                #     fig = px.histogram(
                #         x = df[key].to_list(),
                #     )

                #     st.plotly_chart(fig)


    def register_dashboard(self, expanded=True, prefix= None, form = True ):


        if expanded : 
            with st.expander('Register', expanded=False):
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
        
    def network_dashboard(self, sidebar=True): 

        if sidebar:
            with st.sidebar:
                self.network_dashboard(sidebar=False) 
        n = c.module('namespace')()
        self.networks = n.networks()
        network2index = {n:i for i,n in enumerate(self.networks)}
        index = network2index['local']

        cols = st.columns([1,1])
        self.network = cols[0].selectbox('Select a Network', self.networks, index=index, key='network.sidebar')



        self.subnet = 'commune'
        self.netuid = 0


    def modules_dashboard(self):
        import pandas as pd

        modules = c.modules()
        self.st.line_seperator()
        module2index = {m:i for i,m in enumerate(modules)}
        module  = st.selectbox('Select A Module', modules, module2index['agent'], key=f'serve.module')


        module = c.module(module)
        # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
                    
        with st.expander('Serve', expanded=True):

            with st.form(key='serve'):
                
                kwargs = self.function2streamlit(module=module, fn='__init__' )

                cols = st.columns([1,1,2])
                tag = cols[0].text_input('tag', 'replica', key=f'serve.tag.{module}')
                tag = None if tag == '' else tag

                n = cols[1].number_input('Number of Replicas', 1, 30, 1, 1, key=f'serve.n.{module}')

                serve = cols[2].form_submit_button('Serve')

                if serve:

                    if 'None' == tag:
                        tag = None
                    if 'tag' in kwargs:
                        kwargs['tag'] = tag
                    for i in range(n):
                        try:
                            if tag != None:
                                s_tag = f'{tag}.{i}'
                            else:
                                s_tag = str(i)
                            response = module.serve( kwargs = kwargs, tag=s_tag, network=self.network)
                        except Exception as e:
                            e = c.detailed_error(e)
                            response = {'success': False, 'message': e}
            
                        if response['success']:
                            st.write(response)
                        else:
                            st.error(response)

        with st.expander('Code', expanded=False):
            code = module.code()
            st.markdown(f"""
                        ```python
                        {code}
                        ```
                        """)

        cols = st.columns([2,2])
            
        with st.expander('Add Server'):
            address = st.text_input('Server Address', '')
            add_server = st.button('Add Server')
            if add_server:
                c.add_server(address)
        
        with st.expander('Remove Server'):
            server = st.selectbox('Module Name', self.servers, 0)
            rm_server = st.button('Remove Server')
            if rm_server:
                c.rm_server(server)



    def playground_dashboard(self):
        info = self.module_info
        network = self.network
        module_address = self.module_address
        st.write('Name: ', info['name'])
        schema = info['schema']
        buttons = {}
        for fn, fn_schema in schema.items():
            with st.expander(fn, expanded=False):
                with st.form(key=f'{fn}.form'):
                    kwargs = self.function2streamlit(fn=fn, fn_schema=fn_schema)
                    buttons[fn] = st.form_submit_button(fn)
                    if buttons[fn]:
                        kwargs['network'] = network

                        result = c.submit(c.call, args=[module_address, fn], timeout=10, kwargs=kwargs, return_future=False)[0]
                        st.write('Result', result)


        
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
            if type(v) in [float, int] or c.is_number(v):
                kwargs[k] = cols[col_idx].number_input(fn_key, v, key=f'{key_prefix}.{k}')
            elif v in ['True', 'False']:
                kwargs[k] = cols[col_idx].checkbox(fn_key, v, key=f'{key_prefix}.{k}')
            else:
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
    

    def chat_dashboard(self):
        import streamlit as st
        import random
        import time



        fn = self.fn

        module_name = self.module_name
        module = self.module
        module_info = self.module_info
        

        if fn not in module_info['schema']:
            st.error(f'{fn} not in {module_name}')

            return

        default_kwargs = module_info['schema'][fn]['default']

        with st.expander('Parameters', expanded=False):

            with st.form(key='chat'):
                chat_path : str = f'chat/{module}/defaults'
                kwargs = self.get(chat_path, default={})
                kwargs.update(default_kwargs)
                kwargs = self.function2streamlit(fn=fn, fn_schema=module_info['schema'][fn], salt='chat')
                chat_button = st.form_submit_button('set parameters')
                if chat_button:
                    response = self.put(chat_path, kwargs)
                kwargs = self.get(chat_path, default=kwargs)


        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []


        clear_history = st.button("NOTHING HAPPENED ;)")
        if clear_history:
            st.session_state.messages = []
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
 
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)


            with st.chat_message(self.module_name):
                kwargs = {k:v for i, (k,v) in enumerate(kwargs.items()) if i > 0}
                if 'history' in kwargs:
                    kwargs['history'] = st.session_state.messages
                response = getattr(module, fn)(prompt, **kwargs)
                if isinstance(response, dict):
                    for k in ['response', 'text', 'content', 'message']:
                        if k in response:
                            response = response[k]
                            break
                            
                if isinstance(response, str):
                    st.session_state.messages.append({"role": "assistant", "content": response})

                st.write(response)


            # Add user message to chat history
        

if __name__ == '__main__':
    Dashboard.run()