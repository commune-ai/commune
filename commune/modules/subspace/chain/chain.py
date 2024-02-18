import os
import commune as c
from typing import *


# Subspace = c.module('subspace')

class Chain(c.Module):
 
    
    chain = network = 'main'
    mode = 'docker'
    image_tag = 'subspace.librevo'
    image = f'vivonasg/subspace.librevo-2023-12-26'
    node_key_prefix = 'subspace.node'
    chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    snapshot_path = f"{chain_path}/snapshots"
    evm_chain_id = 69420
    
    def __init__(self, **kwargs):
        self.set_config(kwargs=kwargs)

    @classmethod
    def node_paths(cls, name=None, chain=chain, mode=mode) -> Dict[str, str]:
        if mode == 'docker':
            paths = c.module('docker').ps(f'{cls.node_key_prefix}.{chain}')
        elif mode == 'local':
            paths = c.pm2ls('subspace')
        else:
            raise ValueError(f"Mode {mode} not recognized. Must be 'docker' or 'local'")
        return paths


    @classmethod
    def node_info(cls, node=None, chain=chain, mode=mode) -> Dict[str, str]:
        path = cls.resolve_node_path(node=node, chain=chain)
        logs = cls.node_logs(node=node, chain=chain)
        node_key = cls.get_node_key(node=node, chain=chain)
        
        return {'path': path, 'logs': logs, 'node_key': node_key }


    @classmethod
    def node_logs(cls, node=None, chain=chain, mode=mode, tail=10) -> Dict[str, str]:
        """
        Get the logs for a node per chain and mode.
        """
        path = cls.resolve_node_path(node=node, chain=chain)
        if mode == 'docker':
            return c.dlogs(path, tail=tail) 
        elif mode == 'local':
            return c.logs(path, tail=tail)
        else:
            raise ValueError(f"Mode {mode} not recognized. Must be 'docker' or 'local'")



    @classmethod
    def node2logs(cls, node=None, chain=chain, mode=mode, verbose = True, tail=10) -> Dict[str, str]:
        """
        Get the logs for a node per chain and mode.
        """
        node2logs = {}
        for node in cls.nodes(chain=chain):
            node2logs[node] = cls.node_logs(node=node, chain=chain, mode=mode, tail=tail)
        
        if verbose:
            for k,v in node2logs.items():
                color = c.random_color()
                c.print(k, color=color)
                c.print(v, color=color)
        else:
            return node2logs
        
    n2l = node2logs

    @classmethod
    def node2cmd(cls, node=None, chain=chain, verbose:bool = True) -> Dict[str, str]:
        node_infos = cls.getc(f'chain_info.{chain}.nodes', {})
        node2cmd = {k: v['cmd'] for k,v in node_infos.items()}

        if verbose:
            for k,v in node2cmd.items():
                color = c.random_color()
                c.print(k, color=color)
                c.print(v, color=color)
        else:
            return node2cmd
        
    @classmethod
    def kill_chain(cls, chain=chain, mode=mode):
        c.print(cls.kill_nodes(chain=chain, mode=mode))
        c.print(cls.refresh_chain_info(chain=chain))

    @classmethod
    def refresh_chain_info(cls, chain=chain):
        return cls.putc(f'chain_info.{chain}', {'nodes': {}, 'boot_nodes': []})


    @classmethod
    def kill_node(cls, node=None, chain=chain, mode=mode):
        node_path = cls.resolve_node_path(node=node, chain=chain)
        if mode == 'docker':
            c.module('docker').kill(node_path)
        elif mode == 'local':
            c.kill(node_path)
        return {'success': True, 'message': f'killed {node} on {chain}'}

    @classmethod
    def chain_test(cls, chain:str = chain, verbose:bool=True, snap:bool=False ):

        c.cmd('cargo test', cwd=cls.chain_path, verbose=verbose) 


    @classmethod
    def kill_nodes(cls, chain=chain, verbose=True, mode=mode):

        kill_node_paths = []
        for node_path in cls.node_paths(chain=chain):
            if verbose:
                c.print(f'killing {node_path}',color='red')
            if mode == 'local':
                c.pm2_kill(node_path)
            elif mode == 'docker':
                c.module('docker').kill(node_path)

            kill_node_paths.append(node_path)

        return {
                'success': True, 
                'message': f'killed all nodes on {chain}', 
                'killed_nodes': kill_node_paths,
                'nodes': cls.node_paths(chain=chain)
                }



    @classmethod
    def get_node_id(cls,  node='alice',
                    chain=chain, 
                    max_trials=10, 
                    sleep_interval=1,
                     mode=mode, 
                     verbose=True
                     ):
        node2path = cls.node2path(chain=chain)
        node_path = node2path[node]
        node_id = None
        node_logs = ''
        indicator = 'Local node identity is: '

        while indicator not in node_logs and max_trials > 0:
            if mode == 'docker':
                node_path = node2path[node]
                node_logs = c.module('docker').logs(node_path, tail=400)
            elif mode == 'local':
                node_logs = c.logs(node_path, start_line = 0 , end_line=400, mode='local')
            else:
                raise Exception(f'Invalid mode {mode}')

            if indicator in node_logs:
                break
            max_trials -= 1
            c.sleep(sleep_interval)
        for line in node_logs.split('\n'):
            # c.print(line)
            if 'Local node identity is: ' in line:
                node_id = line.split('Local node identity is: ')[1].strip()
                break

        if node_id == None:
            raise Exception(f'Could not find node_id for {node} on {chain}')

        return node_id
        

    @classmethod
    def node_help(cls, mode=mode):
        chain_release_path = cls.chain_release_path(mode=mode)
        cmd = f'{chain_release_path} --help'
        if mode == 'docker':
            cmd = f'docker run {cls.image} {cmd}'
        elif mode == 'local':
            cmd = f'{cmd}'

        c.cmd(cmd, verbose=True)  


    @classmethod
    def install_rust(cls, sudo=True):
        c.cmd(f'chmod +x scripts/install_rust_env.sh',  cwd=cls.chain_path, sudo=sudo)

    @classmethod
    def build(cls, 
             chain:str = chain, 
             build_runtime:bool=True,
             build_spec:bool=True, 
             build_snapshot:bool=False,  
             verbose:bool=True, 
             mode = mode,
             sync:bool=False,

             ):
        if build_runtime:
            cls.build_runtime(verbose=verbose , mode=mode)

        if build_snapshot or sync:
            cls.build_snapshot(chain=chain, verbose=verbose, sync=sync)

        if build_spec:
            cls.build_spec(chain=chain, mode=mode)

    @classmethod
    def build_image(cls, tag=image_tag, push=True, no_cache=False):
        response = c.build_image('subspace', tag=tag, no_cache=no_cache)
        return response

        
    @classmethod
    def add_node_keys(cls, chain:str=chain, valis:int=24, nonvalis:int=0, refresh:bool=True , mode=mode):
        for i in range(valis):
            cls.add_node_key(node=f'vali_{i}',  chain=chain, refresh=refresh, mode=mode)
        for i in range(nonvalis):
            cls.add_node_key(node=f'nonvali_{i}' , chain=chain, refresh=refresh, mode=mode)

        return {'success': True, 'msg': f'Added {valis} valis and {nonvalis} nonvalis to {chain}'}

    @classmethod
    def add_vali_keys(cls, n:int=24, chain:str=chain,  refresh:bool=False , timeout=10, mode=mode):
        results = []
        for i in range(n):
            result = cls.add_node_key(node=f'vali_{i}',  chain=chain, refresh=refresh, mode=mode)
            results += [results]
        return results

    
    @classmethod
    def rm_node_key(cls,node, chain=chain):
        base_path = cls.resolve_base_path(node=node, chain=chain)
        if c.exists(base_path):
            c.rm(base_path)
        for key in cls.node_key_paths(node=node, chain=chain):
            c.print(f'removing node key {key}')
            c.rm_key(key)
        keystore_path = cls.keystore_path(node=node, chain=chain)
        if c.exists(keystore_path):
            c.rm(keystore_path)
            
        return {'success':True, 'message':'removed all node keys', 'chain':chain, 'keys_left':cls.node_keys(chain=chain)}
    
        
    @classmethod
    def resolve_node_path(cls, node:str='alice', chain=chain, tag_seperator='_'):
        return f'{cls.node_prefix()}.{chain}.{node}'

    @classmethod
    def get_node_key(cls, node='alice', chain=chain, vali=True, crease_if_not_exists:bool=True):
        if crease_if_not_exists:
            if not cls.node_exists(node=node, chain=chain, vali=vali):
                cls.add_node_key(node=node, chain=chain)
        return cls.node_keys(chain=chain)[node]
    
    @classmethod
    def node_key_paths(cls, node=None, chain=chain):
        key = f'{cls.node_key_prefix}.{chain}.{node}.'
        return c.keys(key)
    
    @classmethod
    def node_keys(cls,chain=chain, vali= True):
        prefix = f'{cls.node_key_prefix}.{chain}'
        if vali:
            prefix = f'{prefix}.vali'
        else:
            prefix = f'{prefix}.nonvali'
        key_module= c.module('key')
        node_keys = {}
        for k in c.keys(prefix):
            name = k.split('.')[-2]
            key_type = k.split('.')[-1]
            if name not in node_keys:
                node_keys[name] = {}
            node_keys[name][key_type] = key_module.get_key(k).ss58_address

        # sort by node number

        def get_node_number(node):
  
            if '_' in node and node.split('_')[-1].isdigit():
                return int(node.split('_')[-1])
            else:
                return 10e9

            return int(node.split('_')[-1])

        node_keys = dict(sorted(node_keys.items(), key=lambda item: get_node_number(item[0])))


        return node_keys

    @classmethod
    def node_key(cls, name, chain=chain):
        path = cls.resolve_node_path(node=name, chain=chain)
        node_key = {}
        for key_name in c.keys(path):
            role = key_name.split('.')[-1]
            key = c.get_key(key_name)
            node_key[role] =  key.ss58_address
        return node_key


    @classmethod
    def node_key_mems(cls,node = 'vali_1', chain=chain):
        vali_node_keys = {}
        for key_name in c.keys(f'{cls.node_key_prefix}.{chain}.{node}.'):
            name = key_name.split('.')[-2]
            role = key_name.split('.')[-1]
            key = c.get_key(key_name)
            if name not in vali_node_keys:
                vali_node_keys[name] = { }
            vali_node_keys[name][role] =  key.mnemonic

        if node in vali_node_keys:
            return vali_node_keys[node]

        # if node in vali_node_keys:
        #     return vali_node_keys[node]
        return vali_node_keys
    @classmethod
    def send_node_keys(cls, node:str, chain:str=chain, module:str=None):
        assert module != None, 'module must be specified'
        node_key_mems = cls.node_key_mems()
        for node, key_mems in node_key_mems.items():
            module.add_node_key(node=node, node_key_mems=key_mems)

    @classmethod
    def node_infos(cls, chain=chain):
        return cls.getc(f'chain_info.{chain}.nodes', {})

    @classmethod
    def public_node_infos(cls, chain=chain):
        return cls.getc(f'chain_info.{chain}.public_nodes', {})

    @classmethod
    def vali_infos(cls, chain=chain):
        return {k:v for k,v in cls.node_infos(chain=chain).items() if v['validator']}

    @classmethod
    def nodes(cls, chain=chain):
        node_infos = cls.node_infos(chain=chain)
        nodes = list(node_infos.keys())
        return sorted(nodes, key=lambda n: int(n.split('_')[-1]) if n.split('_')[-1].isdigit() else 10e9)

    @classmethod
    def vali_nodes(cls, chain=chain):
        return [k for k,v in cls.node_infos(chain=chain).items() if v['validator']]

    @classmethod
    def public_nodes(cls, chain=chain):
        return [k for k,v in cls.public_node_infos(chain=chain).items()]


    @classmethod
    def rm_public_nodes(cls, chain=chain):
        config = cls.config()
        
        nodes = {}
        for node, node_info in config['chain_info'][chain]['nodes'].items():
            if node_info['validator']:
                nodes[node] = node_info
        config['chain_info'][chain]['nodes'] = nodes
        cls.save_config(config)
            
        return {'success':True, 'message':'removed all nonvali node keys', 'chain':chain, 'keys_left':cls.node_keys(chain=chain)}

    @classmethod
    def vali_node_keys(cls,chain=chain):
        keys =  {k:v for k,v in  cls.node_keys(chain=chain).items() if k.startswith('vali')}
        keys = dict(sorted(keys.items(), key=lambda k: int(k[0].split('_')[-1]) if k[0].split('_')[-1].isdigit() else 0))
        return keys
    
    @classmethod
    def nonvali_node_keys(self,chain=chain):
        return {k:v for k,v in  self.node_keys(chain=chain).items() if not k.startswith('vali')}
    
    @classmethod
    def node_key_exists(cls, node='alice', chain=chain):
        path = cls.resolve_node_path(node=node, chain=chain)
        return len(c.keys(path+'.')) > 0

    @classmethod
    def add_node_key(cls,
                     node:str,
                     mode = mode,
                     chain = chain,
                     key_mems:dict = None, # pass the keys mems
                     refresh: bool = False,
                     ):
        '''
        adds a node key
        '''
        if key_mems == None:
            key_mems = cls.node_key_mems(node=node, chain=chain)
        cmds = []

        node = str(node)

        c.print(f'adding node key {node} for chain {chain}')
        if  cls.node_key_exists(node=node, chain=chain):
            if refresh:
                cls.rm_node_key(node=node, chain=chain)
            else:
                c.print(f'node key {node} for chain {chain} already exists')
                return {'success':True, 'message':f'node key {node} for chain {chain} already exists', 'chain':chain, 'keys_left':cls.node_keys(chain=chain)}

        chain_path = cls.chain_release_path(mode=mode)

        # we need to resolve the node2keystore path based on the node and chain
        node2keystore_path = cls.keystore_path(node=node, chain=chain)
        if len(c.ls(node2keystore_path)) > 0:
            c.rm(node2keystore_path)
            
        for key_type in ['gran', 'aura']:
            # we need to resolve the schema based on the key type
            if key_type == 'gran':
                schema = 'Ed25519'
            elif key_type == 'aura':
                schema = 'Sr25519'

            # we need to resolve the key path based on the key type
            
            key_path = f'{cls.node_key_prefix}.{chain}.{node}.{key_type}'

            if key_mems != None and len(key_mems) == 2 :
                assert key_type in key_mems, f'key_type {key_type} not in keys {key_mems}'
                key_info = c.add_key(key_path, mnemonic = key_mems[key_type], refresh=True, crypto_type=schema)
                key = c.get_key(key_path, refresh=False)
                assert key_info['mnemonic'] == key_mems[key_type], f'key mnemonic {key.mnemonic} does not match {key_mems[key_type]}'

            else:
                # we need to resolve the key based on the key path
                key = c.get_key(key_path,crypto_type=schema, refresh=refresh)
            

            # we need to resolve the base path based on the node and chain
        
            base_path = cls.resolve_base_path(node=node, chain=chain)
            c.print(f'inserting key {key.mnemonic} into {node2keystore_path}')
            cmd  = f'''{chain_path} key insert --base-path {base_path} --chain {chain} --scheme {schema} --suri "{key.mnemonic}" --key-type {key_type}'''

            if mode == 'docker':
                container_base_path = base_path.replace(cls.chain_path, '/subspace')
                volumes = f'-v {container_base_path}:{base_path}'
                cmd = f'docker run {volumes} {cls.image} {cmd}'
                c.print(c.cmd(cmd, verbose=True, cwd = cls.chain_path))
                c.print(len(c.ls(node2keystore_path)), 'keys in', node2keystore_path)
            elif mode == 'local':
                c.print(cmd)
                c.print(c.cmd(cmd, verbose=True, cwd = cls.chain_path))
                c.print(len(c.ls(node2keystore_path)), 'keys in', node2keystore_path)
        assert len(c.ls(node2keystore_path)) == 2, f'node2keystore_path {node2keystore_path} must have 2 keys'
        return {'success':True, 'node':node, 'chain':chain, 'keys': cls.node_keys(chain=chain)}



    @classmethod   
    def purge_chain(cls,
                    base_path:str = None,
                    chain:str = chain,
                    node:str = 'alice',
                    mode = mode,
                    sudo = False):
        if base_path == None:
            base_path = cls.resolve_base_path(node=node, chain=chain)
        path = base_path+'/chains/commune/db'
        if mode == 'docker':
            c.print(c.chown(path))

        try:
            return c.rm(path)
        except Exception as e:
            c.print(e)
            c.print(c.chown(path))
            return c.rm(path)
            
    
    


    @classmethod
    def chain_target_path(self, chain:str = chain):
        return f'{self.chain_path}/target/release/node-subspace'

    @classmethod
    def build_runtime(cls, verbose:bool=True, mode=mode):
        if mode == 'docker':
            c.module('docker').build(cls.chain_name)
        elif mode == 'local':
            c.cmd('cargo build --release --locked', cwd=cls.chain_path, verbose=verbose)
        else:
            raise ValueError(f'Unknown mode {mode}, must be one of docker, local')

    @classmethod
    def chain_release_path(cls, mode='local'):

        if mode == 'docker':
            chain_path = f'/subspace'
        elif mode == 'local':
            chain_path = cls.chain_path
        else:
            raise ValueError(f'Unknown mode {mode}, must be one of docker, local')
        path =   f'{chain_path}/target/release/node-subspace'
        return path

    @classmethod
    def resolve_base_path(cls, node='alice', chain=chain):
        return cls.resolve_path(f'nodes/{chain}/{node}')
    
    @classmethod
    def keystore_path(cls, node='alice', chain=chain):
        path =  cls.resolve_base_path(node=node, chain=chain) + f'/chains/commune/keystore'
        if not c.exists(path):
            c.mkdir(path)
        return path


    def node2keystore(self, chain=chain):
        node2keystore = {}
        for node in self.nodes():
            node2keystore[node] = [p.split('/')[-1] for p in c.ls(self.keystore_path(node=node, chain=chain))]
        return node2keystore

    def node2keystore_path(self, chain=chain):
        node2keystore_path = {}
        for node in self.nodes():
            node2keystore_path[node] = self.keystore_path(node=node, chain=chain)
        return node2keystore_path

    @classmethod
    def keystore_keys(cls, node='vali_0', chain=chain):
        return [f.split('/')[-1] for f in c.ls(cls.keystore_path(node=node, chain=chain))]


    def build_runtime_wasm(self, sudo=False):
        return c.cmd('cargo build --release --package node-subspace', cwd=self.libpath, verbose=True, sudo=sudo)

    @classmethod
    def build_spec(cls,
                   chain = chain,
                   disable_default_bootnode: bool = True,
                   vali_node_keys:dict = None,
                   return_spec:bool = False,
                   mode : str = mode,
                   valis: int = 21,
                   ):

        chain_spec_path = cls.chain_spec_path(chain=chain)
        chain_release_path = cls.chain_release_path(mode=mode)

        cmd = f'{chain_release_path} build-spec --chain {chain}'
        
        if disable_default_bootnode:
            cmd += ' --disable-default-bootnode'  
       


        if mode == 'docker':
            # chain_spec_path_dir = os.path.dirname(chain_spec_path)
            container_spec_path = cls.spec_path.replace(cls.chain_path, '/subspace')
            container_snap_path = cls.snapshot_path.replace(cls.chain_path, '/subspace')
            volumes = f'-v {cls.spec_path}:{container_spec_path}'\
                        + f' -v {cls.snapshot_path}:{container_snap_path}'
            cmd = f'bash -c "docker run {volumes} {cls.image} {cmd} > {chain_spec_path}"'
        
        elif mode == 'local':
            c.print(cmd)
            cmd = f'bash -c "{cmd} > {chain_spec_path}"'

        c.print(cmd, color='green')
        value = c.cmd(cmd, verbose=True, cwd=cls.chain_path)
        
        if vali_node_keys == None:
            vali_node_keys = cls.vali_node_keys(chain=chain)
        if len(vali_node_keys) < valis:
            cls.add_node_keys(chain=chain, valis=valis, nonvalis=0, refresh=False, mode=mode)
        assert len(vali_node_keys) >= valis, f'vali_nodes ({len(vali_node_keys)}) must be at least {valis}'
        
        vali_nodes = list(vali_node_keys.keys())[:valis]
        vali_node_keys = {k:vali_node_keys[k] for k in vali_nodes}

        spec = c.get_json(chain_spec_path)
        spec['genesis']['runtime']['aura']['authorities'] = [k['aura'] for k in vali_node_keys.values()]
        spec['genesis']['runtime']['grandpa']['authorities'] = [[k['gran'],1] for k in vali_node_keys.values()]
        c.put_json(chain_spec_path, spec)

        if return_spec:
            return spec
        else:
            return {
                    'success':True, 
                    'message':'built spec', 
                    'chain':chain,
                    'valis': vali_nodes
                    }


    @classmethod
    def chain_specs(cls):
        return c.ls(f'{cls.spec_path}/')
    
    @classmethod
    def chain2spec(cls, chain = None):
        chain2spec = {os.path.basename(spec).replace('.json', ''): spec for spec in cls.specs()}
        if chain != None: 
            return chain2spec[chain]
        return chain2spec
    
    specs = chain_specs
    @classmethod
    def get_spec(cls, chain:str=chain):
        chain = cls.chain_spec_path(chain)
        
        return c.get_json(chain)

    spec = get_spec

    @classmethod
    def spec_exists(cls, chain):
        return c.exists(f'{cls.spec_path}/{chain}.json')

    @classmethod
    def save_spec(cls, spec, chain:str=chain):
        chain = cls.chain_spec_path(chain)
        return c.put_json(chain, spec)

    @classmethod
    def chain_spec_path(cls, chain = chain):
        if chain == None:
            chain = cls.chain
        return cls.spec_path + f'/{chain}.json'
    

    @classmethod
    def rm_chain(self, chain):
        return c.rm(self.chain_spec_path(chain))
    
    @classmethod
    def insert_node_key(cls,
                   node='node01',
                   chain = 'jaketensor_raw.json',
                   suri = 'verify kiss say rigid promote level blue oblige window brave rough duty',
                   key_type = 'gran',
                   scheme = 'Sr25519',
                   password_interactive = False,
                   ):
        
        chain_spec_path = cls.chain_spec_path(chain)
        node_path = f'/tmp/{node}'
        
        if key_type == 'aura':
            schmea = 'Sr25519'
        elif key_type == 'gran':
            schmea = 'Ed25519'
        
        if not c.exists(node_path):
            c.mkdir(node_path)

        cmd = f'{cls.chain_release_path()} key insert --base-path {node_path}'
        cmd += f' --suri "{suri}"'
        cmd += f' --scheme {scheme}'
        cmd += f' --chain {chain_spec_path}'

        key_types = ['aura', 'gran']

        assert key_type in key_types, f'key_type ({key_type})must be in {key_types}'
        cmd += f' --key-type {key_type}'
        if password_interactive:
            cmd += ' --password-interactive'
        
        c.print(cmd, color='green')
        return c.cmd(cmd, cwd=cls.chain_path, verbose=True)
    
    def chain_infos(self):
        return self.getc('chain_info', {})
    
    def chain_info(self, chain):
        return self.getc(f'chain_info.{chain}', {})
    
    def chains(self):
        return list(self.getc('chain_info', {}).keys())

    @classmethod
    def node2path(cls, chain=chain, mode = mode):
        prefix = f'{cls.node_prefix()}.{chain}'
        if mode == 'docker':
            path = prefix
            nodes =  c.module('docker').ps(path)
            return {n.split('.')[-1]: n for n in nodes}
        elif mode == 'local':
        
            nodes =  c.pm2ls(f'{prefix}')
            return {n.split('.')[-1]: n for n in nodes}
    @classmethod
    def nonvalis(cls, chain=chain):
        chain_info = cls.chain_info(cxhain=chain)
        return [node_info['node'] for node_info in chain_info['nodes'].values() if node_info['validator'] == False]

    @classmethod
    def valis(cls, chain=chain):
        chain_info = cls.chain_info(chain=chain)
        return [node_info['node'] for node_info in chain_info['nodes'].values() if node_info['validator'] == True]

    @classmethod
    def num_valis(cls, chain=chain):
        return len(cls.vali_nodes(chain=chain))


    @classmethod
    def node_prefix(cls):
        return cls.node_key_prefix
    


    @classmethod
    def chain_info(cls, chain=chain, default:dict=None ): 
        default = {} if default == None else default
        return cls.getc(f'chain_info.{chain}', default)


    @classmethod
    def rm_node(cls, node='bobby',  chain=chain): 
        cls.rmc(f'chain_info.{chain}.{node}')
        return {'success':True, 'msg': f'removed node_info for {node} on {chain}'}


    @classmethod
    def rm_nodes(cls, node='bobby',  chain=chain): 
        cls.rmc(f'chain_info.{chain}.{node}')
        return {'success':True, 'msg': f'removed node_info for {node} on {chain}'}


    @classmethod
    def get_boot_nodes(cls, chain=chain):
        return cls.getc('chain_info.{chain}.boot_nodes')

    @classmethod
    def pull_image(cls):
        return c.cmd(f'docker pull {cls.image}')
    
    @classmethod
    def chain_spec_hash(cls, chain=chain):
        return c.hash( cls.chain_spec(chain=chain))
    @classmethod
    def chain_spec(cls, chain=chain):
        return c.get_json(cls.chain_spec_path(chain=chain))
    
    @classmethod
    def chain_spec_authorities(cls, chain=chain):
        return cls.chain_spec(chain=chain)['genesis']['runtime']['aura']['authorities']
    

    @classmethod
    def id(self):
        return c.hash(self.hash_map())
    
    @classmethod
    def hash_map(self):
        return {
            'image_id': self.image,
            'chain_spec_hash': self.chain_spec_hash()
        }

    @classmethod
    def push_image(cls, image='subspace.librevo', public_image=image, build:bool = False, no_cache=True ):
        if build:
            c.print(cls.build_image(no_cache=no_cache))
        public_image = f'{public_image.split("-")[0]}-{":".join(c.datetime().split("_")[0]).replace(":", "-")}'
        c.cmd(f'docker tag {image} {public_image}', verbose=True)
        c.cmd(f'docker push {public_image}', verbose=True)
        return {'success':True, 'msg': f'pushed {image} to {public_image}'}


    @classmethod
    def start_local_node(cls,
                     node:str='alice', 
                     mode=mode, 
                     chain=chain, 
                     max_boot_nodes:int=24,
                     node_info = None,

                      **kwargs):
        if node_info == None:
            cls.pull_image()
            cls.add_node_key(node=node, chain=chain, mode=mode)
            response = cls.start_node(node=node, chain=chain, mode=mode, local=True, max_boot_nodes=max_boot_nodes, **kwargs)
            node_info = response['node_info']

        cls.put(f'local_nodes/{chain}/{node}', node_info)

        return response

    add_local_node = start_local_node

    @classmethod
    def add_local_nodes(cls, node:str='local', n=4, mode=mode, chain=chain, node_infos=None, **kwargs):
        responses = []
        for i in range(n):
            add_node_kwargs  = dict(node=f'{node}_{i}', mode=mode, chain=chain, **kwargs)
            if node_infos != None:
                assert len(node_infos) == n
                add_node_kwargs['node_info'] = node_infos[i]
            responses += [cls.add_local_node(**add_node_kwargs)]
        return responses
        


    @classmethod
    def check_public_nodes(cls):
        config = cls.config()


    @classmethod
    def start_public_nodes(cls, node:str='nonvali', 
                           n:int=10,
                            i=0,
                            mode=mode, 
                           chain=chain, 
                           max_boot_nodes=24, 
                           refresh:bool = True,
                           remote:bool = False,
                           trial = 3,
                           **kwargs):
        avoid_ports = []
        node_infos = cls.node_infos(chain=chain)
        served_nodes = []
        remote_addresses = []

        

        if remote:
            remote_addresses = c.addresses(network='remote')

        while len(served_nodes) <= n:
            i += 1
            node_name = f'{node}_{i}'
            if node_name in node_infos and refresh == False:
                c.print(f'Skipping {node_name} (Already exists)')
                continue
            else:
                c.print(f'Deploying {node_name}')

            if remote:
                kwargs['module'] = remote_addresses[i % len(remote_addresses)]
                kwargs['boot_nodes'] = cls.boot_nodes(chain=chain)

            else:
                free_ports = c.free_ports(n=3, avoid_ports=avoid_ports)
                avoid_ports += free_ports
                kwargs['port'] = free_ports[0]
                kwargs['rpc_port'] = free_ports[1]
                kwargs['ws_port'] = free_ports[2]

            kwargs['validator'] = False
            kwargs['max_boot_nodes'] = max_boot_nodes
            try:
                response = cls.start_node(node=node_name , chain=chain, mode=mode, **kwargs)
            except Exception as e:
                c.print(e)
                continue
            if 'node_info' not in response:
                c.print(response, 'response')
                raise ValueError('No node info in response')

            node_info = response['node_info']
            c.print('started node', node_name, '--> ', response['logs'])
            served_nodes += [node_name]
            
            cls.putc(f'chain_info.{chain}.public_nodes.{node_name}', node_info)

            

    @classmethod
    def boot_nodes(cls, chain=chain):
        return cls.getc(f'chain_info.{chain}.boot_nodes', [])

     
    @classmethod
    def local_node_paths(cls, chain=chain):
        return [p for p in cls.ls(f'local_nodes/{chain}')]

    @classmethod
    def remove_local_nodes(cls, search=None, chain=chain):
        paths = cls.ls(f'local_nodes/{chain}')
        for path in paths:
            if search != None and search not in path:
                continue
            cls.rm(path)

    @classmethod
    def local_nodes(cls, chain=chain):
        return [p.split('/')[-1].split('.')[0] for p in cls.ls(f'local_nodes/{chain}')]
    

    @classmethod
    def local_node_infos(cls, chain=chain):
        return [cls.get(p) for p in cls.ls(f'local_nodes/{chain}')]
    
    @classmethod
    def local_node_urls(cls, chain=chain):
        return ['ws://'+info['ip']+':' + str(info['rpc_port']) for info in cls.local_node_infos(chain=chain)]

    def local_node2url(self, chain=chain):
        return {info['node']: 'ws://'+info['ip']+':' + str(info['rpc_port']) for info in self.local_node_infos(chain=chain) if isinstance(info, dict)}
    @classmethod
    def kill_local_node(cls, node, chain=chain):
        node_path = cls.resolve_node_path(node=node, chain=chain)
        docker = c.module('docker')
        if docker.exists(node_path):
            docker.kill(node_path)
        return cls.rm(f'local_nodes/{chain}/{node}')

    @classmethod
    def has_local_node(cls, chain=chain):
        return len(cls.local_nodes(chain=chain)) > 0

    @classmethod
    def resolve_node_url(cls, url = None, chain=chain, local=False, mode='ws'):
        if 'local' in chain:
            chain = cls.chain
        if url == None:
            if local:
                local_node_paths = cls.local_node_paths(chain=chain)
                local_node_info = cls.get(c.choice(local_node_paths))
                assert isinstance(local_node_info, dict), f'local_node_info must be a dict'
                port = local_node_info['rpc_port']
                url = f'{mode}://0.0.0.0:{port}'
            else:
                url = c.choice(cls.urls(chain=chain))

        if not url.startswith(f'{mode}://') and not url.startswith(f'{mode}s://'):
            url = f'{mode}://' + url

        return url

    @classmethod
    def start_nodes(self, node='nonvali', n=10, chain=chain, **kwargs):
        results  = []
        for i in range(n):
            results += [self.start_node(node= f'{node}_{i}', chain=chain, **kwargs)]
        return results

    @classmethod
    def local_public_nodes(cls, chain=chain):
        config = cls.config()
        ip = c.ip()
        nodes = []
        for node, node_info in config['chain_info'][chain]['nodes'].items():
            if node_info['ip'] == ip:
                nodes.append(node)

        return nodes

    @classmethod
    def start_vali(cls,*args, **kwargs):
        kwargs['validator'] = True
        return cls.start_node(*args, **kwargs)
    @classmethod
    def start_node(cls,
                 node : str,
                 chain:int = chain,
                 port:int=None,
                 rpc_port:int=None,
                 ws_port:int=None,
                 telemetry_url:str = False,
                 purge_chain:bool = True,
                 refresh:bool = True,
                 verbose:bool = False,
                 boot_nodes = None,
                 node_key = None,
                 mode :str = mode,
                 rpc_cors = 'all',
                 pruning:str = 500,
                 sync:str = 'warp',
                 validator:bool = False,
                 local:bool = False,
                 max_boot_nodes:int = 24,
                 daemon : bool = True,
                 key_mems:dict = None, # pass the keys mems {aura: '...', gran: '...'}
                 module : str = None , # remote module to call
                 remote = False,
                 debug:bool = False,
                 sid:str = None,
                 timeout:int = 30,
                 arm64:bool = False,
                 ):

        if sid != None:
            assert sid == c.sid(), f'remote_id ({sid}) != self_id ({sid})'

        if debug :
            daemon = False 
        if remote and module == None:
            module = cls.peer_with_least_nodes(chain=chain)

        if module != None:
            remote_kwargs = c.locals2kwargs(locals())
            remote_kwargs['module'] = None
            remote_kwargs.pop('timeout', None)
            remote_kwargs.pop('remote', None)
            module = c.namespace(network='remote').get(module, module) # default to remote namespace
            c.print(f'calling remote node {module} with kwargs {remote_kwargs}')
            kwargs = {'fn': 'subspace.start_node', 'kwargs': remote_kwargs, 'timeout': timeout}
            response =  c.call(module,  fn='submit', kwargs=kwargs, network='remote')[0]
            return response


        ip = c.ip()

        node_info = c.locals2kwargs(locals())
        chain_release_path = cls.chain_release_path()

        cmd = chain_release_path

        # get free ports (if not specified)
        free_ports = c.free_ports(n=3)

        if port == None:
            node_info['port'] = port = free_ports[0]
        if rpc_port == None:
            node_info['rpc_port'] = rpc_port = free_ports[1]
        if ws_port == None:
            node_info['ws_port'] = ws_port = free_ports[2]

        # add the node key if it does not exist
        if key_mems != None:
            c.print(f'adding node key for {key_mems}', color='yellow')
            cls.add_node_key(node=node,chain=chain, key_mems=key_mems, refresh=True)

        base_path = cls.resolve_base_path(node=node, chain=chain)
        
        # purge chain's  db if it exists and you want to start from scratch
        if purge_chain:
            cls.purge_chain(base_path=base_path)
            

        cmd_kwargs = f' --base-path {base_path}'



        chain_spec_path = cls.chain_spec_path(chain)
        cmd_kwargs += f' --chain {chain_spec_path}'

        if telemetry_url != False:
            if telemetry_url == None:
                telemetry_url = cls.telemetry_url(chain=chain)
            cmd_kwargs += f' --telemetry-url "{telemetry_url}"'

        if validator :
            cmd_kwargs += ' --validator'
            cmd_kwargs += f" --pruning={pruning}"
            cmd_kwargs += f' --port {port} --rpc-port {rpc_port}'

        else:
            cmd_kwargs += ' --rpc-external'
            cmd_kwargs += f" --pruning={pruning}"
            cmd_kwargs += f" --sync {sync}"
            cmd_kwargs += f' --rpc-cors={rpc_cors}'
            cmd_kwargs += f' --port {port} --rpc-port {rpc_port}'
        if boot_nodes == None:
            boot_nodes = cls.boot_nodes(chain=chain)
        # add the node to the boot nodes
        if len(boot_nodes) > 0:
            node_info['boot_nodes'] = ' '.join(c.shuffle(boot_nodes)[:5])  # choose a random boot node (at we chose one)
            cmd_kwargs += f" --bootnodes {node_info['boot_nodes']}"
    
        if node_key != None:
            cmd_kwargs += f' --node-key {node_key}'

   
        name = f'{cls.node_prefix()}.{chain}.{node}'

        if mode == 'local':
            # 
            output = c.pm2_start(path=cls.chain_release_path(mode=mode), 
                            name=name,
                            cmd_kwargs=cmd_kwargs,
                            refresh=refresh,
                            verbose=verbose)
            
        elif mode == 'docker':
            cls.pull_image()
            docker = c.module('docker')
            if docker.exists(name):
                docker.kill(name)
    
            cmd = cmd + ' ' + cmd_kwargs
            container_chain_release_path = chain_release_path.replace(cls.chain_path, '/subspace')
            cmd = cmd.replace(chain_release_path, container_chain_release_path)

            # run the docker image
            container_spec_path = chain_spec_path.replace(cls.chain_path, '/subspace')
            cmd = cmd.replace(chain_spec_path, container_spec_path)



            key_path = cls.keystore_path(node=node, chain=chain)
            container_base_path = base_path.replace(cls.tmp_dir(), '')
            cmd = cmd.replace(base_path, container_base_path)

            volumes = f'-v {os.path.dirname(chain_spec_path)}:{os.path.dirname(container_spec_path)}'\
                         + f' -v {base_path}:{container_base_path}'
            daemon_str = '-d' if daemon else ''
            # cmd = 'cat /subspace/specs/main.json'
            platform = ""
            if arm64:
                cmd_kwargs += '--platform linux/arm64/v8'

            cmd = 'docker run ' + daemon_str  + f' --net host {platform} --name {name} {volumes}  {cls.image}  bash -c "{cmd}"'
            node_info['cmd'] = cmd

            output = c.cmd(cmd, verbose=debug)
            logs_sig = ' is already in use by container "'
            if logs_sig in output:
                container_id = output.split(logs_sig)[-1].split('"')[0]
                c.module('docker').rm(container_id)
                output = c.cmd(cmd, verbose=debug)
        else: 
            raise Exception(f'unknown mode {mode}')


        response = {
            'success':True,
            'msg': f'Started node {node} for chain {chain} with name {name}',
            'node_info': node_info,
            'logs': output,
            'cmd': cmd

        }
        if validator:
            # ensure you add the node to the chain_info if it is a bootnode
            node_id = cls.get_node_id(node=node, chain=chain, mode=mode)
            response['boot_node'] =  f'/ip4/{ip}/tcp/{node_info["port"]}/p2p/{node_id}'
    
        return response
       
    @classmethod
    def node_exists(cls, node:str, chain:str=chain, vali:bool=False):
        return node in cls.nodes(chain=chain)

    @classmethod
    def node_running(self, node:str, chain:str=chain) -> bool:
        contianers = c.ps()
        name = f'{self.node_prefix()}.{chain}.{node}'
        return name in contianers
        

    @classmethod
    def release_exists(cls):
        return c.exists(cls.chain_release_path())

    kill_chain = kill_nodes
    @classmethod
    def rm_sudo(cls):
        cmd = f'chown -R $USER:$USER {c.cache_path()}'
        c.cmd(cmd, sudo=True)


    @classmethod
    def peer_with_least_nodes(cls, peer2nodes=None):
        peer2nodes = cls.peer2nodes() if peer2nodes == None else peer2nodes
        peer2n_nodes = {k:len(v) for k,v in peer2nodes.items()}
        return c.choice([k for k,v in peer2n_nodes.items() if v == min(peer2n_nodes.values())])

    def kill_peer_nodes(self, chain:str=chain):
        peer2nodes = self.peer2nodes(chain=chain)
        responses = []
        return responses
    
    def resume_chain(self, chain:str=chain, remote=True, mode=mode):
        vali_nodes = self.unresolved_nodes(chain=chain)
        return self.start_chain(chain=chain, 
                         vali_nodes=vali_nodes,
                         remote=remote,
                          mode=mode, 
                         refresh=False, 
                         build_spec=False )
    
    @classmethod
    def start_chain(cls, 
                    chain:str=chain, 
                    valis:int = 21,
                    nonvalis:int = 1,
                    verbose:bool= False,
                    purge_chain:bool = True,
                    refresh: bool = False,
                    remote:bool = False,
                    build_spec :bool = False,
                    push:bool = False,
                    trials:int = 20,
                    timeout:int = 30,
                    wait_for_nodeid = True,
                    max_boot_nodes:int = 1,
                    batch_size:int = 10,
                    paralle:bool = True,
                    min_boot_nodes_before_parallel = 2,
                    vali_nodes:list = None,
                    mode = mode,
                    ):

        # KILL THE CHAIN
        if refresh:
            c.print(f'KILLING THE CHAIN ({chain})', color='red')
            cls.kill_chain(chain=chain)
            chain_info = {'nodes':{}, 'boot_nodes':[]}
            if mode == 'local':
                cls.kill_chain(chain=chain)
        else:
            chain_info = cls.chain_info(chain=chain, default={'nodes':{}, 'boot_nodes':[]})

        if vali_nodes == None:
            ## VALIDATOR NODES
            vali_node_keys  = cls.vali_node_keys(chain=chain)
            num_vali_keys = len(vali_node_keys)
            c.print(f'{num_vali_keys} vali keys found for chain {chain} with {valis} valis needed')

            if len(vali_node_keys) <= valis:
                cls.add_node_keys(chain=chain, valis=valis, refresh=False)
                vali_node_keys  = cls.vali_node_keys(chain=chain)

            vali_nodes = list(vali_node_keys.keys())[:valis]
            vali_node_keys = {k: vali_node_keys[k] for k in vali_nodes}

            # BUILD THE CHAIN SPEC AFTER SELECTING THE VALIDATOR NODES'
            if build_spec:
                c.print(f'building spec for chain {chain}')
                cls.build_spec(chain=chain, vali_node_keys=vali_node_keys, valis=valis)
                if remote or push:
                    # we need to push this to 
                    cls.push(rpull=remote)
        
    
        remote_address_cnt = 1
        avoid_ports = []

        if remote:
            peer2nodes = cls.peer2nodes(chain=chain, update=True)
            node2peer = cls.node2peer(peer2nodes=peer2nodes)
        
        
        finished_nodes = []
        # START THE VALIDATOR NODES
        finished_nodes =  list(set(finished_nodes))
        c.print(f'finished_nodes {len(finished_nodes)}/{valis}')

        for i, node in enumerate(vali_nodes):

            if node in finished_nodes:
                c.print(f'node {node} already started, skipping', color='yellow')
                finished_nodes += [node]

                continue
            
            c.print(f'[bold]Starting node {node} for chain {chain}[/bold]')
            name = f'{cls.node_prefix()}.{chain}.{node}'

            if remote:
                if name in node2peer:
                    finished_nodes += [node]
                    finished_nodes =  list(set(finished_nodes))
                    c.print(f'node {node} already exists on peer {node2peer[name]}', color='yellow')
                    continue

            # BUILD THE KWARGS TO CREATE A NODE
            
            node_kwargs = {
                            'chain':chain, 
                            'node':node, 
                            'verbose':verbose,
                            'purge_chain': purge_chain,
                            'validator':  True,
                            

                            }

            futures = []
            success = False
            for i in range(trials):
                try:
                    if remote:  
                        peer2num_nodes = {k:len(v) for k,v in peer2nodes.items()}
                        # get the least loaded peer
                        c.print(f'peer2num_nodes {peer2num_nodes}')
                        if i <= 2:
                            remote_address = cls.peer_with_least_nodes(peer2nodes=peer2nodes)
                        else:
                            remote_address = c.choice(list(peer2nodes.keys()))
                        remote_address_cnt += 1
                        node_kwargs['module'] = remote_address
                        module = c.connect(remote_address)

                    else:
                        port_keys = ['port', 'rpc_port', 'ws_port']
                        node_ports = c.free_ports(n=len(port_keys), avoid_ports=avoid_ports)
                        for k, port in zip(port_keys, node_ports):
                            avoid_ports.append(port)
                            node_kwargs[k] = port
                        module = cls


                    node_kwargs['sid'] = c.sid()
                    node_kwargs['boot_nodes'] = chain_info['boot_nodes']
                    node_kwargs['key_mems'] = cls.node_key_mems(node, chain=chain)
                    c.print(f"node_kwargs {node_kwargs['key_mems']}")
                    assert len(node_kwargs['key_mems']) == 2, f'no key mems found for node {node} on chain {chain}'


                    response = module.start_node(**node_kwargs, refresh=refresh, timeout=timeout, mode=mode)
                    
                    assert 'node_info' in response and ('boot_node' in response or 'boot_node' in response['node_info'])
                    
                    response['node_info'].pop('key_mems', None)
                    node = response['node_info']['node']

                    if remote:
                        peer2nodes[remote_address].append(node)

                    node_info = response['node_info']
                    boot_node = response['boot_node']
                    chain_info['boot_nodes'].append(boot_node)
                    chain_info['nodes'][node] = node_info
                    finished_nodes += [name]

                    cls.putc(f'chain_info.{chain}', chain_info)
                    break
                except Exception as e:
                    c.print(c.detailed_error(e))
                    c.print(f'trials {i}/{trials} failed for node {node} on chain {chain}')



        if nonvalis > 0:
            # START THE NON VALIDATOR NODES
            cls.start_public_nodes(n=nonvalis, chain=chain, refresh=True, remote=remote)

        return {'success':True, 'msg': f'Started chain {chain}', 'valis':valis, 'nonvalis':nonvalis}
   
    @classmethod
    def public_node2url(cls, chain:str = chain) -> str:
        assert isinstance(chain, str), f'chain must be a string, not {type(chain)}'
        nodes =  cls.getc(f'chain_info.{chain}.public_nodes', {})
        nodes = {k:v for k,v in nodes.items() if v['validator'] == False}
        assert len(nodes) > 0, f'No url found for {chain}'
        public_node2url = {}
        for k_n, v_n in nodes.items():
            public_node2url[k_n] = v_n['ip'] + ':' + str(v_n['rpc_port'])
        return public_node2url

    @classmethod
    def urls(cls, chain: str = chain) -> str:
        return list(cls.public_node2url(chain=chain).values())

    def random_urls(self, chain: str = chain, n=4) -> str:
        urls = self.urls(chain=chain)
        return c.sample(urls, n=1)


    @classmethod
    def test_node_urls(cls, chain: str = chain) -> str:
        nodes = cls.public_nodes()
        config = cls.config()
        
        for node in nodes:
            try:
                url = cls.resolve_node_url(node)
                s = cls()
                s.set_network(url=url)
                c.print(s.block, 'block for node', node)
                
            except Exception as e:
                c.print(c.detailed_error(e))
                c.print(f'node {node} is down')
                del config['chain_info'][chain]['nodes'][node]

        cls.save_config(config)


    @classmethod
    def filter_endpoints(cls, timeout=10, chain=chain):
        node2pass = cls.test_endpoints(timeout=timeout)
        chain_info = cls.chain_info(chain=chain)
        for node in list(chain_info['nodes'].keys()):
            if node2pass[node] != True:
                c.print(f'removing node {node} from chain {chain}')
                del chain_info['nodes'][node]
        cls.putc(f'chain_info.{chain}', chain_info)

    def node2url(self, chain=chain):
        if "local" in chain:
            return self.local_node2url()
        else:
            return self.public_node2url(chain=chain)

    @classmethod
    def test_endpoints(cls, timeout:int=30):
        public_node2url = cls.public_node2url()
        futures = []
        node2future = {}
        for node, url in public_node2url.items():
            future = c.submit(cls.test_endpoint, kwargs=dict(url=url), return_future=True, timeout=timeout)
            c.print(future)
            node2future[node] = future
        futures = list(node2future.values())
        results = c.wait(futures, timeout=timeout)
        node2results = {k:v for k,v in zip(node2future.keys(), results)}
        return node2results





    def unresolved_nodes (self, chain=chain):
        chain_info = self.chain_info(chain=chain)
        remote_nodes = self.remote_nodes(chain=chain)
        remote_node_suffixes = [n.split('.')[-1] for n in remote_nodes]
        c.print(remote_node_suffixes)
        return [node for node, node_info in chain_info['nodes'].items() if node not  in remote_node_suffixes]


    @classmethod
    def remote_nodes(cls, chain=chain, timeout=5):
        import commune as c
        ps_map = c.module('remote').call('ps', f'{cls.node_prefix()}.{chain}', timeout=timeout)
        all_ps = []
        for ps in ps_map.values():
            if isinstance(ps, list):
                all_ps.extend(ps)
        vali_ps = sorted([p for p in all_ps if '.vali' in p and 'subspace' in p], key=lambda x: int(x.split('_')[-1]))
        return vali_ps

    @classmethod
    def peer2nodes(cls, chain=chain, update:bool = True):
        path = f'chain_info.{chain}.peer2nodes'
        if not update:
            peer2nodes = cls.get(path, {})
            if len(peer2nodes) > 0:
                return peer2nodes
        peer2nodes = c.module('remote').call('ps', f'{cls.node_key_prefix}.{chain}')
        namespace = c.namespace(network='remote')
        peer2nodes = {namespace.get(k):v for k,v in peer2nodes.items() if isinstance(v, list)}

        cls.put(path, peer2nodes)

        return peer2nodes

    @classmethod
    def clean_bootnodes(cls, peer2nodes=None):
        peer2nodes = cls.peer2nodes() if peer2nodes == None else peer2nodes
        boot_nodes = cls.boot_nodes()
        cleaned_boot_nodes = []
        for peer, nodes in peer2nodes.items():
            if len(nodes) > 0:
                peer_ip = ':'.join(peer.split(':')[:-1])
                for i in range(len(boot_nodes)):
  
                    if peer_ip in boot_nodes[i]:
                        if boot_nodes[i] in cleaned_boot_nodes:
                            continue
                        cleaned_boot_nodes.append(boot_nodes[i])
    

        cls.putc('chain_info.main.boot_nodes', cleaned_boot_nodes)
        return len(cleaned_boot_nodes)

                

    @classmethod
    def node2peer(cls, chain=chain, peer2nodes = None):
        node2peer = {}
        if peer2nodes == None:
            peer2nodes = cls.peer2nodes(chain=chain)
        for peer, nodes in peer2nodes.items():
            for node in nodes:
                node2peer[node] = peer
        return node2peer

    @classmethod
    def vali2peer(cls, chain=chain):
        node2peer = cls.node2peer(chain=chain)
        vali2peer = {k:v for k,v in node2peer.items() if '.vali' in k}
        return len(vali2peer)

    @classmethod
    def peer2ip(cls):
        namespace = c.namespace(network='remote')
        peer2ip = {k:':'.join(v.split(':')[:-1]) for k,v in namespace.items()}
        return peer2ip

    @classmethod
    def ip2peer(cls):
        peer2ip = cls.peer2ip()
        ip2peer = {v:k for k,v in peer2ip.items()}
        return ip2peer

    def empty_peers(self, chain=chain):
        peer2nodes = self.peer2nodes(chain=chain)
        empty_peers = [p for p, nodes in peer2nodes.items() if len(nodes) == 0]
        return empty_peers

    def unfound_nodes(self, chain=chain, peer2nodes=None):
        node2peer = self.node2peer(peer2nodes=peer2nodes)
        vali_infos = self.vali_infos(chain=chain)
        vali_nodes = [f'{self.node_key_prefix}.{chain}.' + v for v in vali_infos.keys()]

        unfound_nodes = [n for n in vali_nodes if n not in node2peer]
        return unfound_nodes
    
    @classmethod
    def snapshots(cls):
        return list(cls.snapshot_map().keys())


    @classmethod
    def snapshot_map(cls):
        return {l.split('/')[-1].split('.')[0]: l for l in c.ls(f'{cls.chain_path}/snapshots')}
        
    
    @classmethod
    def up(cls):
        c.cmd('docker-compose up -d', cwd=cls.chain_path)

    @classmethod
    def enter(cls):
        c.cmd('make enter', cwd=cls.chain_path)

    @classmethod
    def snapshot(cls, chain=chain) -> dict:
        path = f'{cls.snapshot_path}/main.json'
        return c.get_json(path)


    @classmethod
    def convert_snapshot(cls, from_version=3, to_version=2, network=network):
        
        
        if from_version == 1 and to_version == 2:
            factor = 1_000 / 42 # convert to new supply
            path = f'{cls.snapshot_path}/{network}.json'
            snapshot = c.get_json(path)
            snapshot['balances'] = {k: int(v*factor) for k,v in snapshot['balances'].items()}
            for netuid in range(len(snapshot['subnets'])):
                for j, (key, stake_to_list) in enumerate(snapshot['stake_to'][netuid]):
                    c.print(stake_to_list)
                    for k in range(len(stake_to_list)):
                        snapshot['stake_to'][netuid][j][1][k][1] = int(stake_to_list[k][1]*factor)
            snapshot['version'] = to_version
            c.put_json(path, snapshot)
            return {'success': True, 'msg': f'Converted snapshot from {from_version} to {to_version}'}

        elif from_version == 3 and to_version == 2:
            path = cls.latest_archive_path()
            state = c.get(path)
            subnet_params : List[str] =  ['name', 'tempo', 'immunity_period', 'min_allowed_weights', 'max_allowed_weights', 'max_allowed_uids', 'trust_ratio', 'min_stake', 'founder']
            module_params : List[str] = ['Keys', 'Name', 'Address']

            modules = []
            subnets = []
            for netuid in range(len(state['subnets'])):
                keys = state['Keys'][netuid]
                for i in range(len(keys)):
                    module = [state[p][netuid] for p in module_params]
                    modules += [module]
                c.print(state['subnets'][netuid])
                subnet = [state['subnets'][netuid][p] for p in subnet_params]
                subnets += [subnet]

            snapshot = {
                'balances': state['balances'],
                'modules': modules,
                'version': 2,
                'subnets' : subnets,
                'stake_to': state['StakeTo'],
            }

            c.put_json(f'{cls.snapshot_path}/{network}-new.json', snapshot)

        else:
            raise Exception(f'Invalid conversion from {from_version} to {to_version}')


    @classmethod
    def build_snapshot(cls, 
              path : str  = None,
            snapshot_chain : str = None ,
            chain : str = chain,
            subnet_params : List[str] =  ['name', 'tempo', 'immunity_period', 'min_allowed_weights', 'max_allowed_weights', 'max_allowed_uids', 'trust_ratio', 'min_stake', 'founder'],
            module_params : List[str] = ['key', 'name', 'address'],
            save: bool = True, 
            min_balance:int = 100000,
            verbose: bool = False,
            sync: bool = False,
            version: str = 2,
             **kwargs):
        if sync:
            c.sync(network=chain)

        if snapshot_chain == None:
            snapshot_chain = chain

    
        subspace = c.module('subspace')()

        path = path if path != None else subspace.latest_archive_path(network=snapshot_chain)
        c.print(f'building snapshot from {path}', color='green')
        state = subspace.get(path)
        
        snap = {
                'subnets' : [[s[p] for p in subnet_params] for s in state['subnets']],
                'modules' : [[[m[p] for p in module_params] for m in modules ] for modules in state['modules']],
                'balances': {k:v for k,v in state['balances'].items() if v > min_balance},
                'stake_to': [[[staking_key, stake_to] for staking_key,stake_to in state['stake_to'][i].items()] for i in range(len(state['subnets']))],
                'block': state['block'],
                'version': version,
                }
                        
        # add weights if not already in module params
        if 'weights' not in module_params:
            snap['modules'] = [[m + c.copy([[]]) for m in modules] for modules in snap['modules']]

        # save snapshot into subspace/snapshots/{network}.json
        if save:
            c.mkdir(cls.snapshot_path)
            snapshot_path = f'{cls.snapshot_path}/{chain}.json'
            c.print('Saving snapshot to', snapshot_path, verbose=verbose)
            c.put_json(snapshot_path, snap)
        # c.print(snap['modules'][0][0])

        date = c.time2date(int(path.split('-')[-1].split('.')[0]))
        
        return {'success': True, 'msg': f'Saved snapshot to {snapshot_path} from {path}', 'date': date}    
    
    snap = build_snapshot

    @classmethod
    def rpull(cls, timeout=10):
        # pull from the remote server
        c.rcmd('c s pull', verbose=True, timeout=timeout)
        c.rcmd('c s pull_image', verbose=True, timeout=timeout)

    @classmethod
    def pull(cls, rpull:bool = False):
        if len(cls.ls(cls.libpath)) < 5:
            c.rm(cls.libpath)
        c.pull(cwd=cls.libpath)
        if rpull:
            cls.rpull()

    @classmethod
    def push(cls, rpull:bool=False, image:bool = False ):
        c.push(cwd=cls.libpath)
        if image:
            cls.push_image()
        if rpull:
            cls.rpull()
