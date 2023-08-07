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

    def sync(self):
        self.subspace.sync()
        return self.subspace.load_state()

    @classmethod
    def function2streamlit(cls, 
                           fn_schema, 
                           extra_defaults:dict=None,
                           cols:list=None):
        if extra_defaults is None:
            extra_defaults = {}

        st.write('#### Startup Arguments')
        # refresh = st.checkbox('**Refresh**', False)
        # mode = st.selectbox('**Select Mode**', ['pm2',  'ray', 'local'] ) 
        mode = 'pm2'
        serve = True

        kwargs = {}
        fn_schema['default'].pop('self', None)
        fn_schema['default'].pop('cls', None)
        fn_schema['default'].update(extra_defaults)
        
        

        
        
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        if cols == None:
            cols = st.columns(len(fn_schema['default']))

        for i, (k,v) in enumerate(fn_schema['default'].items()):
            
            optional = fn_schema['default'][k] != 'NA'
            fn_key = k 
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
            kwargs[k] = cols[col_idx].text_input(fn_key, v)
            
        return kwargs
            
            
    @classmethod
    def process_kwargs(cls, kwargs:dict, fn_schema:dict):
        
        for k,v in kwargs.items():
            if v == 'None':
                v = None

            elif k in fn_schema['input'] and fn_schema['input'][k] == 'str':
                if v.startswith("f'") or v.startswith('f"'):
                    v = c.ljson(v)
                elif v.startswith('[') and v.endswith(']'):
                    v = v
                elif v.startswith('{') and v.endswith('}'):
                    v = v
                else:
                    v = v
                
            elif k == 'kwargs':
                continue
            elif v == 'NA':
                assert k != 'NA', f'Key {k} not in default'
            else:
                v = eval(v) 
            
        kwargs[k] = v
        return kwargs
    @classmethod
    def streamlit_module_launcher(cls, module, mode:str='pm2', fn_name='__init__'):

        module_path = module.module_path()
        st.write(f'## Module: {module.module_path()}')
            
        # function_map =self.module_info['funciton_schema_map'] = self.module_info['object'].get_schema()
        # function_signature = self.module_info['function_signature_map'] = self.module_info['object'].get_function_signature_map()
        module_schema = module.schema(defaults=True)
        
        cols = st.columns(2)
        name = cols[0].text_input('**Name**', module_path)
        tag = cols[1].text_input('**Tag**', 'None')
        config = module.config(to_munch=False)
        
        
        fn_schema = module_schema[fn_name]
        kwargs = cls.function2streamlit(fn_schema=fn_schema, extra_defaults=config )
        
        launch_button = st.button('Launch Module')  
        
        if launch_button:
            
            kwargs = cls.process_kwargs(kwargs=kwargs, fn_schema=fn_schema)
            
                
                
            launch_kwargs = dict(
                module = module,
                name = name,
                tag = tag,
                mode = mode,
                refresh = True,
                kwargs = kwargs,
            )
            commune.launch(**launch_kwargs)
            st.success(f'Launched {name} with {kwargs}')
        
        
        with st.expander('Config'):
            st.write(module.config())

        with st.expander('input'):
            st.write(module.schema())
    
            

    
    def streamlit_sidebar(self, wrapper = True):
        if wrapper:
            with st.sidebar:
                return self.streamlit_sidebar(False)
        
        
    
        st.write('## Modules')    
        self.server_name = st.selectbox('',self.module_list, 0, key='module_name')   
        self.module = commune.module(self.server_name)
        self.module_config = self.module.config(to_munch=False)


        with st.expander('Modules'):
            modules =commune.modules()
            st.multiselect('', modules, modules)


            self.update_button = st.button('Sync Network', False)

        if self.update_button:
            self.sync()
            
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
        
        self.streamlit_module_launcher(module=self.module)


    @staticmethod
    def local_css(file_name=os.path.dirname(__file__)+'/style.css'):
        import streamlit as st
        
        
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
        

if __name__ == '__main__':
    Dashboard.streamlit()