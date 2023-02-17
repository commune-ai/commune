import commune
import streamlit as st

# commune.launch('dataset.text.bittensor', mode='pm2')

# commune.new_event_loop()



class Dashboard:

    def __init__(self):
        
        self.server_registry = commune.server_registry()
        self.public_ip = commune.external_ip()
        self.live_peers = list(self.server_registry.keys())
        
    def set_peer(ip:str = None, port:int = None):
        if ip is None:
            ip = self.public_ip
        if port is None:
            port = commune.port()
        commune.set_peer(ip, port)
        st.write(f'Peer set to {ip}:{port}')

    def streamlit_launcher(self):
        self.module_list = commune.module_list()
        selected_module = st.selectbox('Module List', self.module_list, 0)
        with st.expander('Module List'):
            st.write(self.module_list)

        info = dict(
            path = commune.simple2path(selected_module),
            config = commune.simple2config(selected_module),
            object = commune.simple2object(selected_module),
        )
        
        
        function_map = info['object'].get_function_schema_map()
        
        st.write(function_map)
        
        st.write(info)



    @classmethod
    def run(cls):
        self = cls()
        self.selected_peers = st.multiselect('Select Module', self.live_peers, self.live_peers[:1])

        peer_info_map = {}
        for peer in self.selected_peers:
            peer_info = self.server_registry[peer]
            peer_info['url'] = f'{commune.external_ip()}:{peer_info["port"]}'
            peer_info_map[peer] = peer_info
            with st.expander(f'Peer Info: {peer}'):
                st.write(peer_info)
                
                module = commune.connect(peer)
                print(module.module_id)

        st.write('Launcher')

        st.write(peer_info_map)
            


if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.streamlit_launcher()