import commune
import streamlit as st 
from typing import List, Dict, Union, Any 
import os

class Dashboard(commune.Module):

    def __init__(self):
        self.public_ip = commune.external_ip()
        self.load_state()

    
    def load_state(self):
        self.namespace = commune.namespace(update=False)
        self.servers = list(self.namespace.keys())
        self.module_tree = commune.module_tree()
        self.module_list = ['module'] + list(self.module_tree.keys())
        sorted(self.module_list)
        
    


    def streamlit_module_browser(self):

        module_list =  self.module_list
        

        
        
        st.write(f'## Module: {self.module_name}')
            
        # function_map =self.module_info['funciton_schema_map'] = self.module_info['object'].get_function_schema_map()
        # function_signature = self.module_info['function_signature_map'] = self.module_info['object'].get_function_signature_map()
        module_schema = self.module.schema(include_default=True)
        # st.write(module_schema)
        fn_name = '__init__'
        fn_info = module_schema[fn_name]
        
        init_kwarg = {}
        cols = st.columns([3,1,6])

        st.write('#### Startup Arguments')
        cols = st.columns([2,1,4,4])
        

    
        tag = None
        # tag = st.text_input('**Tag**', 'None')
        if tag == 'None' or tag == '' or tag == 'null':
            tag = None
        # refresh = st.checkbox('**Refresh**', False)
        # mode = st.selectbox('**Select Mode**', ['pm2',  'ray', 'local'] ) 
        mode = 'pm2'
        serve = True

        
        fn_info['default'].pop('self', None)
        fn_info['default'].pop('cls', None)
        
        kwargs_cols = st.columns(3)
        
        # kwargs_cols[0].write('## Module Arguments')
        for i, (k,v) in enumerate(fn_info['default'].items()):
            
            optional = fn_info['default'][k] != 'NA'
            fn_key = k 
            if k in fn_info['schema']:
                k_type = fn_info['schema'][k]
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            kwargs_col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            
            
            
            kwargs_col_idx = kwargs_col_idx % (len(kwargs_cols))
            init_kwarg[k] = kwargs_cols[kwargs_col_idx].text_input(fn_key, v)
            
        init_kwarg.pop('self', None )
        init_kwarg.pop('cls', None)
        name = st.text_input('**Name**', self.module_name) 

        launch_button = st.button('Launch Module')  

        if launch_button:
            kwargs = {}
            
            
            for k,v in init_kwarg.items():
                
                if v == 'None':
                    v = None
                elif k in fn_info['schema'] and fn_info['schema'][k] == 'str':
                    v = v
                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                else:
                    v = eval(v) 
                
                kwargs[k] = v
                
                
            launch_kwargs = dict(
                module = self.module,
                name = name,
                tag = tag,
                mode = mode,
                refresh = True,
                kwargs = kwargs,
            )
            commune.launch(**launch_kwargs)

        
        with st.expander('Config'):
            st.write(self.module_config)

        with st.expander('Schema'):
            st.write(self.module.schema())
    
            

    
    def streamlit_sidebar(self, wrapper = True):
        if wrapper:
            with st.sidebar:
                return self.streamlit_sidebar(False)
        
        
        

        st.write('## Modules')    
        self.module_name = st.selectbox('',self.module_list, 0, key='module_name')   
        self.module = commune.module(self.module_name)
        self.module_config = self.module.config()


        with st.expander('Modules'):
            modules =commune.modules()
            st.multiselect('', modules, modules)
            self.update_button = st.button('Update', False)

        if self.update_button:
            self.load_state()
            
        self.streamlit_peers()
           
    def st_root_module(self):
        self.root_module = commune.root_module()
        self.root_module_info = self.root_module.info()

        st.write('## My Address:',  self.root_module_info["address"])
 
    def streamlit_peers(self):
        
        st.write('## Peers')

        peer = st.text_input('', '0.0.0.0:8888', key='peer')
        cols = st.columns(2)
        add_peer_button = cols[0].button('add Peer')
        rm_peer_button = cols[1].button('rm peer')
        if add_peer_button :
            self.add_peer(peer)
        if rm_peer_button :
            self.rm_peer(peer)
            
        peers = commune.peers()
        st.write(peers)
            
        
            
            
        

        
 
    def streamlit_server_info(self):
        
        
        for peer_name, peer_info in self.namespace.items():
            with st.expander(peer_name, True):
                peer_info['address']=  f'{peer_info["ip"]}:{peer_info["port"]}'
                st.write(peer_info)
                
            
            
        
        
     

        peer_info_map = {}
 
    def function_call(self, module:str, fn_name: str = '__init__' ):
        
        module = commune.connect(module)
        peer_info = module.peer_info()
        
        function_info_map = self.module.function_info_map()
        fn_name = fn_name
        fn_info = function_info_map[fn_name]
        
        kwargs = {}
        cols = st.columns([3,1,6])
        cols[0].write(f'#### {name}.{fn_name}')
        cols[2].write('#### Module Arguments')
        cols = st.columns([2,1,4,4])
        launch_col = cols[0]
        kwargs_cols = cols[2:]
        st.write(function_info_map)
                
        st.write(self.module_config, 'brooo')
        
        with launch_col:
            tag = None
            # tag = st.text_input('**Tag**', 'None')
            if tag == 'None' or tag == '' or tag == 'null':
                tag = None
            # mode = st.selectbox('**Select Mode**', ['pm2',  'ray', 'local'] ) 
            mode = 'pm2'
            serve = True
            launch_button = st.button('Launch Module')  
            
        
        # kwargs_cols[0].write('## Module Arguments')
        for i, (k,v) in enumerate(fn_info['default'].items()):
            
            optional = fn_info['default'][k] != 'NA'
            fn_key = k 
            if k in fn_info['schema']:
                k_type = fn_info['schema'][k]
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            kwargs_col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            
            
            
            kwargs_col_idx = kwargs_col_idx % (len(kwargs_cols))
            init_kwarg[k] = kwargs_cols[kwargs_col_idx].text_input(fn_key, v)
            
        init_kwarg.pop('self', None )
        init_kwarg.pop('cls', None)
        


        if launch_button:
            kwargs = {}
            for k,v in init_kwarg.items():
                if v == 'None':
                    v = None
                elif k in fn_info['schema'] and fn_info['schema'][k] == 'str':
                    v = v
                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                else:
                    v = eval(v) 
                
                kwargs[k] = v
                
            launch_kwargs = dict(
                module = self.module,
                name = name,
                tag = tag,
                mode = mode,
                refresh = refresh,
                kwargs = kwargs,
            )
            commune.launch(**launch_kwargs)
            
   
    def streamlit_playground(self):
        class bro:
            def __init__(self, a, b):
                self.a = a
                self.b = b
                
        st.write(str(type(commune)) == "<class 'module'>")
        st.write()
        pass
    
    @classmethod
    def streamlit(cls):
        
        cls.local_css()
        commune.new_event_loop()

        commune.nest_asyncio()
        self = cls()
        self.st_root_module()

        
        self.streamlit_sidebar()
        
        
        tabs = st.tabs(['Modules', 'Peers', 'Users', 'Playground'])
        
        self.streamlit_module_browser()


    @staticmethod
    def local_css(file_name=os.path.dirname(__file__)+'/style.css'):
        import streamlit as st
        
        
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
        

if __name__ == '__main__':
    Dashboard.streamlit()