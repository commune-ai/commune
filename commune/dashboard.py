import commune
import streamlit as st

# commune.launch('dataset.text.bittensor', mode='pm2')

# commune.new_event_loop()



class Dashboard:

    def __init__(self):
        
        self.server_registry = commune.server_registry()
        self.public_ip = commune.external_ip()
        self.live_peers = self.server_registry.keys()
        self.selected_peers = st.multiselect('Select Module', self.live_peers)
        
        
    def set_peer(ip:str = None, port:int = None):
        if ip is None:
            ip = self.public_ip
        if port is None:
            port = commune.port()
        commune.set_peer(ip, port)
        st.write(f'Peer set to {ip}:{port}

    def run(self):
        for peer in self.selected_peers:
            st.write(f'Peer: {peer}')
            st.write(f'IP: {self.server_registry[peer]}')
            st.write(f'Port: {commune.port()}')
            st.write(f'Public IP: {self.public_ip}')
            st.write(f'URL: http://{self.server_registry[peer]}:{commune.port()}')
            st.write(f'URL: http://{self.public_ip}:{commune.port()}/{peer}')