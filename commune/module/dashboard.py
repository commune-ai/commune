import commune
import streamlit as st

# commune.new_event_loop()
with st.expander('Live Servers'):
    st.write(commune.server_registry())

public_ip = commune.external_ip()
st.metric('Public IP', public_ip)
server_registry = commune.server_registry()
live_peers = server_registry.keys()

selected_peers = st.multiselect('Select Module', live_peers)

for selected_peer in selected_peers:
    
    st.write(commune.connect(selected_peer, virtual=True).functions())