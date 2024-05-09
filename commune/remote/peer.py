    
    

import commune as c 


class Peer(c.Module):


    def namespace(self, search=None, network='remote', update=False):
        namespace = {}
        if not update:
            namespace = c.get_namespace(network=network)
            return namespace
        
        peer2namespace = self.peer2namespace()
        for peer, peer_namespace in peer2namespace.items():

            for name, address in peer_namespace.items():
                if search != None and search not in name:
                    continue
                if name in namespace:
                    continue
                namespace[name + '_'+ {peer}] = address
        c.put_namespace(namespace=namespace, network=network)
        return namespace



    def peers(self, network='remote'):
        return self.servers(search='module', network=network)
    
    def peer2hash(self, search='module', network='remote'):
        return self.call('chash', search=search, network=network)


    # peers

    def peer2memory(self, min_memory:int=0, **kwargs):
        peer2hardware = self.hardware(**kwargs)
        peer2memory = {}
        for server, hardware in peer2hardware.items():
            memory_info = hardware['memory']
            if isinstance(memory_info, dict) and 'available' in memory_info:
                if memory_info['available'] >= min_memory:
                    peer2memory[server] = memory_info['available']
        return peer2memory
    
    @classmethod
    def available_peers(cls, min_memory:int=10, update: bool=False, address=False):
        peer2hardware = cls.hardware(update=update)
        peer2memory = {}
        
        for peer, hardware in peer2hardware.items():
            memory_info = hardware['memory']
        
            if isinstance(memory_info, dict) and 'available' in memory_info:
                if memory_info['available'] >= min_memory:
                    peer2memory[peer] = memory_info['available']
        
        if address :
            namespace = c.namespace(network='remote')
            peers = [namespace.get(k) for k in peer2memory]
        else :
            peers = list(peer2memory.keys())
        return peers
    

    @classmethod
    def most_available_peer(cls, min_memory:int=8, update: bool=False):
        peer2hardware = cls.hardware(update=update)
        peer2memory = {}
        
        for peer, hardware in peer2hardware.items():
            memory_info = hardware['memory']
            if isinstance(memory_info, dict) and 'available' in memory_info:
                peer2memory[peer] = memory_info['available']
        
        # get the top n peers with the most memory
        peer2memory = {k:v for k,v in sorted(peer2memory.items(), key=lambda x: x[1], reverse=True)}
        most_available_peer = list(peer2memory.keys())[0]
        most_available_peer_memory = peer2memory[most_available_peer]
        assert most_available_peer_memory >= min_memory, f'Not enough memory, {most_available_peer_memory} < {min_memory}'
        c.print(f'Most available peer: {most_available_peer} with {most_available_peer_memory} GB memory')
        return most_available_peer
    

    
    @classmethod
    def check_peers(cls, timeout=10):
        futures = []
        for m,a in c.namespace(network='remote').items():
            futures += [c.submit(c.call, args=(a,'info'),return_future=True)]
        results = c.wait(futures, timeout=timeout)
        return results
    
    
    @classmethod
    def peer2namespace(cls):
        info_paths = cls.ls('peers')
        peer2namespace = {}
        for path in info_paths:
            info = cls.get(path, {})
            peer2namespace[path] = info.get('namespace', {})
        return peer2namespace


    @classmethod
    def add_peers(cls, add_admins:bool=False, timeout=20, update=False, network='remote'):
        """
        Adds servers to the network by running `c add_peers` on each server.
        
        """
        if update:
            c.rm_namespace(network=network)
        if add_admins:
            cls.add_admin(timeout=timeout)

        namespace = c.namespace(network=network)
        address2name = {v:k for k,v in namespace.items()}
        host2_server_addresses_responses = cls.cmd('c addy', verbose=True, timeout=timeout)
        for i, (host,server_address) in enumerate(host2_server_addresses_responses.items()):
            if isinstance(server_address, str):
                server_address = server_address.split('\n')[-1]

                if server_address in address2name:
                    server_name = address2name[server_address]
                    c.print(f'{server_name} already in namespace')
                    continue
                else:
                    ip = ':'.join(server_address.split(':')[:-1])

                    server_name = 'module' + '_' +  host
                    namespace[server_name] = server_address

        c.put_namespace(network=network, namespace=namespace)

        return {'status': 'success', 'msg': f'Servers added', 'namespace': namespace}
 


    def peer_info(self, peer):
        host2ip = self.host2ip()
        peer = host2ip.get(peer, peer)
        return self.get(f'peers/{peer}', {})


    @classmethod
    def serve(cls, *args, update=False, min_memory:int=10, timeout=10, **kwargs):
        modules = cls.available_peers(update=update, min_memory=min_memory)
        c.print(f'Available modules: {modules}')
        module = c.choice(modules)
        c.print(f'Serving  on {module}')
        namespace = c.namespace(network='remote')
        address = namespace[module]
        module = c.connect(address)
        return module.serve(*args, **kwargs, timeout=timeout)
    
    @classmethod
    def servers(self,search: str ='module', network='remote'):
        return c.servers(search, network=network)
    

    @classmethod
    def fleet(cls, *args, update=False, min_memory:int=20, n=10, tag=None, **kwargs):
        responses = []
        for i in range(n):
            try:
                response = cls.serve(*args, 
                        update=update, 
                        min_memory=min_memory, 
                        tag= '' if tag == None else tag + str(i), 
                        **kwargs
                        )
            except Exception as e:
                response = c.detailed_error(e)
            c.print(response)
            responses += [response]
        return responses

    def num_servers(self):
        return len(self.servers())
    

    @classmethod
    def n_servers(self):
        return len(self.servers())


    
    @classmethod
    def peer2key(cls, search=None, network:str='remote', update=False):
        infos = c.infos(search=search, network=network, update=update)
        return {v['name']:v['ss58_address'] for v in infos if 'name' in v and 'address' in v}

    @classmethod
    def peer_addresses(cls, network:str='remote'):
        infos = c.infos(network=network)
        return {info['ss58_address'] for info in infos if 'ss58_address' in info}
    
    def check_peers(self, timeout=10):
        futures = []
        for m,a in c.namespace(network='remote').items():
            futures += [c.submit(c.call, args=(a,'info'),return_future=True)]
        results = c.wait(futures, timeout=timeout)
        return results

    def peerpath2name(self, path:str):
        return path.split('/')[-1].replace('.json', '')
    
    def peer2info(self):
        peer_infos = {}
        for path in self.ls('peers'):
            peer_name = self.peerpath2name(path)
            info = self.get(path, {})   
            peer_infos[peer_name] = info
            peer_infos[peer_name] = info
        return peer_infos
    
    def peer2lag(self, max_age=1000):
        peer2timestamp = self.peer2timestamp()
        time = c.time()
        ip2host = self.ip2host()
        return {ip2host.get(k,k):time - v for k,v in peer2timestamp.items() if time - v < max_age}

    def peer2timestamp(self):
        peer2info = self.peer2info()
        return {k:v.get('timestamp', 0) for k,v in peer2info.items()}

    def peer2hardware(self):
        info_paths = self.ls('peers')
        peer_infos = []
        for path in info_paths:
            c.print(path)
            info = self.get(path, {})
            # c.print(info)
            peer_infos += [info.get('hardware', {})]
        return peer_infos 
    

    @classmethod
    def hardware(cls, timeout=20, update= True, cache_path:str = 'hardware.json', trials=2):


        if not update:
            peer2hardware = c.get_json(cache_path, {})
            if len(peer2hardware) > 0:
                return peer2hardware
        peer2hardware = {p:None for p in peers}
        peers = cls.peers()
        for i in range(trials):
            call_modules = [p for p in peers if peer2hardware[p] == None]
            response =  cls.call('hardware', timeout=timeout, modules=call_modules)
            for peer, hardware in response.items():
                if isinstance(hardware, dict) and 'memory' in hardware:
                    c.print(f'{peer} {hardware}')
                    peer2hardware[peer] = hardware


        c.put_json(cache_path, peer2hardware)
        return peer2hardware
    
    
    @classmethod
    def addresses(self, search=None, network='remote'):
        return c.addresses(search=search, network=network)
    
    def keys(self):
        return [info.get('ss58_address', None)for info in self.infos()]
    
    @classmethod
    def infos(self, search='module',  network='remote', update=False):
        return c.infos(search=search, network=network, update=update)

    
    @classmethod
    def peer2address(self, network='remote'):
        return c.namespace(search='module', network=network)
    
    

# def peer_dashboard(self):

#     cols = st.columns(2)
#     search = cols[0].text_input('Search', 'module')
#     peer2info = self.remote.peer2info()

#     peer_info_df = []
#     for peer, info in peer2info.items():
#         memory_fields = ['available', 'total', 'used']
#         row = {'peer': peer}
#         for field in memory_fields:
#             row['memory_'+field] = info.get('hardware', {}).get('memory', {}).get(field, None)
#         # disk fields
#         disk_fields = ['total', 'used', 'free']
#         for field in disk_fields:
#             row['disk_'+field] = info.get('hardware', {}).get('disk', {}).get(field, None)
#         peer_info_df += [row]
#         row['num_modules'] = len(info.get('namespace', {}))
    
#     peer_info_df = pd.DataFrame(peer_info_df)
#     namespace = c.namespace(search=search, network='remote')
#     ip2host = self.remote.ip2host()

#     with st.expander('Peers', expanded=False):
#         for peer, info in peer2info.items():
#             cols = st.columns([1,4])
#             peer = ip2host.get(peer, peer)
#             cols[0].write('#### '+peer)

#             timestamp = info.get('timestamp', None)
#             lag = c.time() - timestamp if timestamp != None else None
#             if lag != None:
#                 lag = round(lag, 2)
#                 st.write(f'{lag} seconds ago')
#             cols[1].write(info.get('hardware', {}))
#             cols[1].write(info.get('namespace', {}))

#         if len(namespace) == 0:
#             st.error(f'No peers found with search: {search}')
#             return
#         n = cols[1].slider('Number of servers', 1, len(namespace), len(namespace))
#         module_names = list(namespace.keys())[:n]
#         module_names = st.multiselect('Modules', module_names, module_names)
#         namespace = {k:v for k,v in namespace.items() if k in module_names}
#         module_addresses = list(namespace.values())
#         module_names = list(namespace.keys())
    
#     if len(module_names) == 0:
#         st.error('No modules found')
#         return
    
#     cols = st.columns(3)
#     module_name = cols[0].selectbox('Module', module_names, index=0)
#     module_address = namespace[module_name]
#     c.print(f'Connecting to {module_name} {module_address}')
#     module = c.connect(module_address)
#     cache = cols[2].checkbox('Cache', True)

#     cache_path = f'module_info_cache/{module_address}'
#     t1 = c.time()
#     if cache:
#         module_info = self.get_json(cache_path, {})
#     else:
#         module_info = {}

#     if len(module_info) == 0:
#         st.write('Getting module info')
        
#         module_info = module.info()
#         self.put_json(cache_path, module_info)
#     fns = list(module_info['schema'].keys())
#     fn_name = st.selectbox('Function', fns, index=0)
#     fn = getattr(module, fn_name)
#     with st.expander(fn_name, expanded=False):
#         kwargs = self.function2streamlit(fn=fn_name, fn_schema=module_info['schema'][fn_name])
#     timeout = cols[1].number_input('Timeout', 1, 100, 10, key='timeout_fn')
#     run = st.button(f'Run {fn_name}')
    
#     if run:
#         future2module = {}
#         for module_address in module_addresses:
#             kwargs['fn'] = fn_name
#             future = c.submit(c.call, args=[module_address], kwargs=kwargs, return_future=True)
#             future2module[future] = module_address
        
#         futures = list(future2module.keys())
#         modules = list(future2module.values())
#         for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):
#             if not ('idx' in result and 'result' in result):
#                 continue

#             module_name = modules[result['idx']]
#             result = result['result']
            
#             with st.expander(f'{module_name}', expanded=False):

#                 st.markdown(f'### {module_name}')
#                 if c.is_error(result):
#                     result
#                     pass
#                 else:
#                     st.write(result)
    
#     t2 = c.time()
#     st.write(f'Info took {t2-t1} seconds')


