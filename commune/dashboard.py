import commune
import streamlit as st 

from typing import List, Dict, Union, Any 
# commune.launch('dataset.text.bittensor', mode='pm2')

# commune.new_event_loop()



class Dashboard:

    def __init__(self):
        self.public_ip = commune.external_ip()
        self.load_state()

    
    def load_state(self):
        self.server_registry = commune.server_registry()
        self.live_peers = list(self.server_registry.keys())
        for peer in self.live_peers:
            if peer not in self.server_registry:
                self.server_registry[peer] = commune.connect(peer).server_stats
        
        self.module_tree = commune.module_tree()
        self.module_list = list(self.module_tree.keys()) + ['module']
        self.peer_registry = commune.peer_registry()
        
    

    
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

    @property
    def default_categories(self) -> List[str]:
        default_categories = ['web3', 'model', 'dataset']
        return default_categories

    def streamlit_module_browser(self):

        module_categories = self.module_categories
        module_list =  self.module_list
        
   
        
        self.module_info = dict(
            path = commune.simple2path(self.selected_module),
            config = commune.simple2config(self.selected_module),
            module = commune.simple2object(self.selected_module),
        )
        self.module_name = self.module_info['config']['module']
        
        st.write(f'## {self.module_name} ({self.selected_module})')
        
        self.streamlit_launcher()
        
        
        with st.expander(f'Module Function Info', False):
            st.write(self.module_info['function_info_map'])
   

        # # st.write(function_map['__init__'])
        # with st.expander('Module Function Schema',False):
        #     st.write(function_map)
        # with st.expander('Info'):
        #     st.write(self.module_info)
            

    
    def streamlit_sidebar(self):
        with st.sidebar:
            st.sidebar.write('# Module Land')
            self.selected_categories = st.multiselect('Categories', self.module_categories, self.default_categories )
            self.filtered_module_list = self.filter_modules_by_category(self.module_list, self.selected_categories)
            
            self.selected_module = st.selectbox('Module List',self.filtered_module_list, 0)   
        
            self.update_button = st.button('Update', False)
            if self.update_button:
                self.load_state()
                
            st.write(self.live_peers)
                
            st.metric('Number of Module Serving', len(self.live_peers))


 
    def streamlit_peer_info(self):
        
        
        

        for ip in self.peer_registry:
            num_peers = len(self.peer_registry[ip])
            cols = st.columns(3)
            num_columns = 4 
            num_rows = num_peers // num_columns + 1
            peer_registry = self.peer_registry
            is_local =  bool(ip == self.public_ip)
            peer_names = list(self.peer_registry[ip].keys())
            peer_info = list(self.peer_registry[ip].values())

            for i, peer_name in enumerate(peer_names):
                peer = peer_info[i]
                with st.expander(f'{peer_name}', False):
                    st.write(peer)

                    kill_button = st.button(f'Kill {peer_name}')
                    if kill_button:
                        commune.kill_server(peer_name, mode='pm2')
                        st.experimental_rerun()
                        self.load_state()
    
                

        peer_info_map = {}
 
        
    def streamlit_launcher(self):
        # function_map =self.module_info['funciton_schema_map'] = self.module_info['object'].get_function_schema_map()
        # function_signature = self.module_info['function_signature_map'] = self.module_info['object'].get_function_signature_map()
        function_info_map = self.module_info['function_info_map'] = self.module_info['module'].get_function_info_map(include_module=False)
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
       
            name = st.text_input('**Name**', self.module_name) 
            refresh = st.checkbox('**Refresh**', False)
            # mode = st.selectbox('**Select Mode**', ['pm2',  'ray', 'local'] ) 
            mode = 'pm2'
            serve = True
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
                module = self.selected_module,
                name = name,
                tag = None,
                mode = mode,
                refresh = refresh,
                serve = serve,
                kwargs = kwargs,
            )
            commune.launch(**launch_kwargs)
        
    def streamlit_playground(self):
        pass
        # dataset = commune.connect('HFDataset')
        
        # st.write('## Get Example')
        
        # model = commune.connect('model::gpt125m')
        # st.write(model.forward(**dataset.sample()))
        
  
            
    @classmethod
    def streamlit(cls):
        self = cls()
        st.set_page_config(layout="wide")
        
        self.streamlit_sidebar()
        tabs = st.tabs(['Module Launcher', 'Peers', 'Playground'])
        with tabs[2]:
            self.streamlit_playground()
        with tabs[1]:
            self.streamlit_peer_info()
        with tabs[0]:
            self.streamlit_module_browser()

        


        
        

if __name__ == '__main__':
    Dashboard.streamlit()