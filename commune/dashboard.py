import commune
import streamlit as st

from typing import List, Dict, Union, Any 
# commune.launch('dataset.text.bittensor', mode='pm2')

# commune.new_event_loop()



class Dashboard:

    def __init__(self):
        
        self.server_registry = commune.server_registry()
        self.public_ip = commune.external_ip()
        self.live_peers = list(self.server_registry.keys())
        
        
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
        return filtered_module_list
        
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
            
            
            with st.expander('Module List'):
                st.write(self.module_list)



        info = dict(
            path = commune.simple2path(selected_module),
            config = commune.simple2config(selected_module),
            object = commune.simple2object(selected_module),
        )
        module_name = info['config']['module']
        
        st.write(f'## {module_name} ({selected_module})')
        
        
        function_map =info['funciton_schema_map'] = info['object'].get_function_schema_map()
        function_signature = info['function_signature_map'] = info['object'].function_signature_map()
        if hasattr(info['object'], 'default_value_map'):
            default_value_map = info['default_value_map'] = info['object'].default_value_map()

                    
            st.write(default_value_map)   

        st.write(function_signature['fetch_text'])
        # st.write(function_map['__init__'])
        with st.expander('Module Function Schema',False):
            st.write(function_map)
        with st.expander('Info'):
            st.write(info)
            

    
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
        
        with st.sidebar:
            st.write('# Module Launcher')
            st.write('## Peer Info')
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

        st.write('Launcher')
        

  
            
    @classmethod
    def streamlit(cls):
        self = cls()
        self.streamlit_module_browser()
        self.streamlit_launcher()
        st.write(self.live_peers)
        


        
        

if __name__ == '__main__':
    Dashboard.streamlit()