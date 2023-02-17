import commune
import streamlit as st

from typing import List, Dict, Union, Any 
# commune.launch('dataset.text.bittensor', mode='pm2')

# commune.new_event_loop()



class Dashboard:

    def __init__(self):
        
        self.server_registry = commune.server_registry()
        self.public_ip = commune.external_ip()
        
    @property
    def live_peers(self):
        return list(self.server_registry.keys())
    @property
    def module_list(self):
        return commune.module_list() 
    
    @property
    def module_categories(self):
        return list(set([m.split('.')[0] for m in self.module_list]))
        
    def set_peer(ip:str = None, port:int = None):
        if ip is None:
            ip = self.public_ip
        if port is None:
            port = commune.port()
        commune.set_peer(ip, port)
        st.write(f'Peer set to {ip}:{port}')

    @staticmethod
    def filter_modules_by_category(module_list:List[str], categories: List[str]):
        filtered_module_list = []
        for m in module_list:
            if any([m.startswith(c) for c in categories]):
                filtered_module_list.append(m)
        return['module'] +  filtered_module_list 
    def streamlit_launcher(self):
        st.write('# Module Launcher')
    def streamlit_module_browser(self,
                            default_categories = ['client', 'web3', 'model', 'dataset']
                            ):

        module_categories = self.module_categories
        module_list =  self.module_list
        
        with st.sidebar:
            selected_categories = st.multiselect('Categories', module_categories, default_categories )
            module_list = self.filter_modules_by_category(module_list, selected_categories)
            st.sidebar.write('# Module Land')

            selected_module = st.selectbox('Module List',module_list, 0)        
            
        



        info = dict(
            path = commune.simple2path(selected_module),
            config = commune.simple2config(selected_module),
            module = commune.simple2object(selected_module),
        )
        module_name = info['config']['module']
        
        st.write(f'## {module_name} ({selected_module})')
        
        
        # function_map =info['funciton_schema_map'] = info['object'].get_function_schema_map()
        # function_signature = info['function_signature_map'] = info['object'].get_function_signature_map()
        function_info_map = info['function_info_map'] = info['module'].get_function_info_map(include_module=False)
        
        
        init_fn_name = '__init__'
        init_fn_info = function_info_map[init_fn_name]

        init_kwarg = {}

        cols = st.columns([3,1,6])
        cols[0].write('#### Launcher')
        cols[2].write('#### Module Arguments')
        cols = st.columns([2,1,4,4])
        launch_col = cols[0]
        kwargs_cols = cols[2:]
        
        with launch_col:
       
            mode = st.selectbox('**Select Mode**', ['pm2',  'ray', 'local'] ) 
            name = st.text_input('**Name**', module_name) 
            refresh = st.checkbox('**Refresh**', False)
            serve = st.checkbox('**Serve**', True)
            launch_button = st.button('Launch Module')  
            
            
            
        # kwargs_cols[0].write('## Module Arguments')
        for i, (k,v) in enumerate(init_fn_info['default'].items()):
            
            optional = init_fn_info['default'][k] != 'NA'
            fn_key = k 
            if k in init_fn_info['schema']:
                k_type = init_fn_info['schema'][k]
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            kwargs_col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            
            kwargs_col_idx = kwargs_col_idx % (len(kwargs_cols))
            init_kwarg[k] = kwargs_cols[kwargs_col_idx].text_input(fn_key, v)
            
        if launch_button:
            kwargs = {}
            for k,v in init_kwarg.items():
                if v == 'None':
                    v = None
                elif k in init_fn_info['schema'] and init_fn_info['schema'][k] == 'str':
                    v = v
                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                else:
                    v = eval(v) 
                
                kwargs[k] = v
                
                
            launch_kwargs = dict(
                module = selected_module,
                name = name,
                tag = None,
                mode = mode,
                refresh = refresh,
                serve = serve,
                kwargs = kwargs,
            )
            cols[0].write(launch_kwargs)
            commune.launch(**launch_kwargs)
        with st.expander(f'Module Function Info', False):
            st.write(function_info_map)
   

        # # st.write(function_map['__init__'])
        # with st.expander('Module Function Schema',False):
        #     st.write(function_map)
        # with st.expander('Info'):
        #     st.write(info)
            

    
    peer_info_cache = {}
    @classmethod
    def get_peer_info(cls,peer, update: bool = False):
        
        peer_info = commune.get_peer_info(peer)
        
        
        
        with st.sidebar:
            
            cols = st.columns(3)
               
            with cols[0]:
                kill_button = st.button('Kill')
                if kill_button:
                    commune.kill_server(peer)
                    commune.servers()
            
            with cols[1]:
                start_button = cols[1].button('Start')
                
                if start_button:
                    commune.launch(peer)
                    
            refresh_button = cols[2].button('Refresh')
            if refresh_button:
                commune.pm2_kill(peer)
                commune.launch(peer)
                commune.launch() 
            
        
        # st.write(peer_info) 

    @classmethod
    def streamlit_launcher(cls):
        self = cls()
        
        server_registry = self.server_registry
        with st.sidebar:
            with st.expander('Peers', False):
                st.write(server_registry)
            peer = st.selectbox('Select Module', self.live_peers, 0)
            

        peer_info_map = {}
        peer_info = self.get_peer_info(peer)
        
        
            
            
            
            # peer_info['url'] = f'{commune.external_ip()}:{peer_info["port"]}'
            # peer_info_map[peer] = peer_info
            
            # with st.expander(f'Peer Info: {peer}'):
            #     st.write(peer_info)
                
            #     module = commune.connect(peer)
            #     st.write(module.server_stats)
            #     print(module.module_id)

        

  
            
    @classmethod
    def streamlit(cls):
        self = cls()
        st.set_page_config(layout="wide")
        self.streamlit_module_browser()
        self.streamlit_launcher()
        


        
        

if __name__ == '__main__':
    Dashboard.streamlit()