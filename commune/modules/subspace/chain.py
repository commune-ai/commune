
import torch
import scalecodec
from retry import retry
from typing import List, Dict, Union, Optional, Tuple
from substrateinterface import SubstrateInterface
import commune as c
from typing import List, Dict, Union, Optional, Tuple
from commune.utils.network import ip_to_int, int_to_ip
from rich.prompt import Confirm
from commune.modules.subspace.balance import Balance
from commune.modules.subspace.utils import (U16_NORMALIZED_FLOAT,
                                    U64_MAX,
                                    NANOPERTOKEN, 
                                    U16_MAX, 
                                    is_valid_address_or_public_key, 
                                    )
from commune.modules.subspace.chain_data import (ModuleInfo, custom_rpc_type_registry)

import streamlit as st
import json
from loguru import logger
import os
logger = logger.opt(colors=True)



class Subspace(c.Module):
    """
    Handles interactions with the subspace chain.
    """
    fmt = 'j'
    whitelist = []
    chain_name = 'subspace'
    default_config = c.get_config(chain_name, to_munch=False)
    token_decimals = default_config['token_decimals']
    network = default_config['network']
    chain = network
    chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    netuid = default_config['netuid']
    mode = default_config['mode']

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
                node_logs = c.module('docker').logs(node_path)
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
    def start_node(cls,
                 node : str,
                 chain:int = network,
                 port:int=None,
                 rpc_port:int=None,
                 ws_port:int=None,
                 telemetry_url:str = 'wss://telemetry.gpolkadot.io/submit/0',
                 purge_chain:bool = True,
                 refresh:bool = False,
                 verbose:bool = False,
                 boot_nodes = None,
                 node_key = None,
                 mode :str = mode,
                 rpc_cors = 'all',
                 validator:bool = False,
                 
                 ):

        ip = c.ip()

        node_info = c.locals2kwargs(locals())

        cmd = cls.chain_release_path()

        free_ports = c.free_ports(n=3)

        if port == None:
            node_info['port'] = port = free_ports[0]
            
        if rpc_port == None:
            node_info['rpc_port'] = rpc_port = free_ports[1]
        if ws_port == None:
            node_info['ws_port'] = ws_port = free_ports[2]
        # resolve base path
        base_path = cls.resolve_base_path(node=node, chain=chain)
        
        # purge chain
        if purge_chain:
            cls.purge_chain(base_path=base_path)
            
        cmd_kwargs = f' --base-path {base_path}'

        chain_spec_path = cls.chain_spec_path(chain)
        cmd_kwargs += f' --chain {chain_spec_path}'
    
            
        if validator :
            cmd_kwargs += ' --validator'
        else:
            cmd_kwargs += ' --ws-external --rpc-external'
        cmd_kwargs += f' --port {port} --rpc-port {rpc_port} --ws-port {ws_port}'
        
        chain_info = cls.getc(f'chain_info.{chain}', {})
        boot_nodes = chain_info.get('boot_nodes', [])
        chain_info['nodes'] = chain_info.get('nodes', {})
        chain_info['nodes'][node] = node_info
        boot_nodes = chain_info['boot_nodes'] = chain_info.get('boot_nodes', [])
        
        # add the node to the boot nodes
        if len(boot_nodes) > 0:
            node_info['boot_nodes'] = c.choice(boot_nodes) # choose a random boot node (at we chose one)
            cmd_kwargs += f" --bootnodes {node_info['boot_nodes']}"
    
        if node_key != None:
            cmd_kwargs += f' --node-key {node_key}'
            
        cmd_kwargs += f' --rpc-cors={rpc_cors}'

        name = f'{cls.node_prefix()}.{chain}.{node}'

        c.print(f'Starting node {node} for chain {chain} with name {name} and cmd_kwargs {cmd_kwargs}')

        if mode == 'local':
            # 
            cmd = c.pm2_start(path=cls.chain_release_path(mode=mode), 
                            name=name,
                            cmd_kwargs=cmd_kwargs,
                            refresh=refresh,
                            verbose=verbose)
            
        elif mode == 'docker':

            # run the docker image
            volumes = f'-v {base_path}:{base_path} -v {cls.spec_path}:/subspace/specs'
            net = '--net host'
            c.cmd('docker run -d --name  {name} {net} {volumes} subspace bash -c "{cmd}"', verbose=verbose)
        else: 
            raise Exception(f'unknown mode {mode}')
        
        if validator:
            # ensure you add the node to the chain_info if it is a bootnode
            node_id = cls.get_node_id(node=node, chain=chain, mode=mode)
            chain_info['boot_nodes'] +=  [f'/ip4/{ip}/tcp/{node_info["port"]}/p2p/{node_id}']
        chain_info['nodes'][node] = node_info
        cls.putc(f'chain_info.{chain}', chain_info)


        return {'success':True, 'msg': f'Node {node} is not a validator, so it will not be added to the chain'}
    

    @classmethod
    def start_nodes(self, node='node', n=10, chain=chain, **kwargs):
        nodes = self.nodes(chain=chain)
        for node in nodes:
            self.start_node(node=node, chain=chain, **kwargs)



    @classmethod
    def start_chain(cls, 
                    chain:str=chain, 
                    n_valis:int = 8,
                    n_nonvalis:int = 16,
                    verbose:bool = False,
                    purge_chain:bool = True,
                    refresh: bool = True,
                    trials:int = 3,
                    reuse_ports: bool = False, 
                    port_keys: list = ['port', 'rpc_port', 'ws_port'],
                    ):

        # KILL THE CHAIN
        if refresh:
            c.print(f'KILLING THE CHAIN ({chain})', color='red')
            cls.kill_chain(chain=chain)


        ## VALIDATOR NODES
        vali_node_keys  = cls.vali_node_keys(chain=chain)
        vali_nodes = list(cls.vali_node_keys(chain=chain).keys())
        if n_valis != -1:
            vali_nodes = vali_nodes[:n_valis]
        vali_node_keys = {k: vali_node_keys[k] for k in vali_nodes}
        assert len(vali_nodes) >= 2, 'There must be at least 2 vali nodes'
        # BUILD THE CHAIN SPEC AFTER SELECTING THE VALIDATOR NODES
        cls.build_spec(chain=chain, verbose=verbose, vali_node_keys=vali_node_keys)

        ## NON VALIDATOR NODES
        nonvali_node_keys = cls.nonvali_node_keys(chain=chain)
        nonvali_nodes = list(cls.nonvali_node_keys(chain=chain).keys())
        if n_nonvalis != -1:
            nonvali_nodes = nonvali_nodes[:n_valis]
        nonvali_node_keys = {k: nonvali_node_keys[k] for k in nonvali_nodes}

        # refresh the chain info in the config

        
        existing_node_ports = {'vali': [], 'nonvali': []}
        
        if reuse_ports: 
            node_infos = cls.getc(f'chain_info.{chain}.nodes')
            for node, node_info in node_infos.items():
                k = 'vali' if node_info['validator'] else 'nonvali'
                existing_node_ports[k].append([node_info[pk] for pk in port_keys])

        if refresh:
            # refresh the chain info in the config
            cls.putc(f'chain_info.{chain}', {'nodes': {}, 'boot_nodes': [], 'url': []})

        avoid_ports = []

        # START THE VALIDATOR NODES
        for node in (vali_nodes + nonvali_nodes):
            c.print(f'Starting node {node} for chain {chain}')
            name = f'{cls.node_prefix()}.{chain}.{node}'

            # BUILD THE KWARGS TO CREATE A NODE
            
            node_kwargs = {
                            'chain':chain, 
                            'node':node, 
                            'verbose':verbose,
                            'purge_chain': purge_chain,
                            'validator':  bool(node in vali_nodes),
                            }

            # get the ports for (port, rpc_port, ws_port)
            # if we are reusing ports, then pop the first ports from the existing_node_ports
            node_ports= []
            node_type = 'vali' if node_kwargs['validator'] else 'nonvali'
            if len(existing_node_ports[node_type]) > 0:
                node_ports = existing_node_ports[node_type].pop(0)
            else:
                node_ports = c.free_ports(n=3, avoid_ports=avoid_ports)
            assert  len(node_ports) == 3, f'node_ports must be of length 3, not {len(node_ports)}'

            for k, port in zip(port_keys, node_ports):
                avoid_ports.append(port)
                node_kwargs[k] = port


            fails = 0
            while trials > fails:
                try:
                    cls.start_node(**node_kwargs, refresh=refresh)
                    break
                except Exception as e:
                    c.print(f'Error starting node {node} for chain {chain}, {e}', color='red')
                    fails += 1
                    raise e
                    continue

    
    @classmethod
    def vali_nodes(cls, chain=chain):
        return cls.nodes(mode='vali', chain=chain)
    

    @classmethod
    def nonvali_nodes(cls, chain=chain):
        return cls.nodes(mode='nonvali', chain=chain)


    @classmethod
    def vali_node_keys(cls,chain=chain):
        return {k:v for k,v in  cls.node_keys(chain=chain).items() if k.startswith('vali')}
    
    @classmethod
    def nonvali_node_keys(self,chain=chain):
        return {k:v for k,v in  self.node_keys(chain=chain).items() if k.startswith('nonvali')}
    

    @classmethod
    def node_key_exists(cls, node='alice', chain=chain):
        return len(cls.node_key_paths(node=node, chain=chain)) > 0


    @classmethod
    def add_node_key(cls,
                     node:str,
                     mode: str = 'nonvali',
                     chain = chain,
                     tag_seperator = '_', 
                     refresh: bool = False,
                     ):
        '''
        adds a node key
        '''
        cmds = []

        assert mode in ['vali', 'nonvali'], f'Unknown mode {mode}, must be one of vali, nonvali'
        node = str(node)

        c.print(f'adding node key {node} for chain {chain}')

        node = c.copy(f'{mode}{tag_seperator}{node}')


        chain_path = cls.chain_release_path(mode='local')

        for key_type in ['gran', 'aura']:

            if key_type == 'gran':
                schema = 'Ed25519'
            elif key_type == 'aura':
                schema = 'Sr25519'

            key_path = f'{cls.node_key_prefix}.{chain}.{node}.{key_type}'

            key = c.get_key(key_path,crypto_type=schema, refresh=refresh)

            base_path = cls.resolve_base_path(node=node, chain=chain)

            
            cmd  = f'''{chain_path} key insert --base-path {base_path} --chain {chain} --scheme {schema} --suri "{key.mnemonic}" --key-type {key_type}'''
            
            cmds.append(cmd)

        for cmd in cmds:
            # c.print(cmd)
            # volumes = f'-v {base_path}:{base_path}'
            # c.cmd(f'docker run {volumes} subspace {cmd} ', verbose=True)
            c.cmd(cmd, verbose=True, cwd=cls.chain_path)

        return {'success':True, 'node':node, 'chain':chain, 'keys':cls.get_node_key(node=node, chain=chain, mode=mode)}



    @classmethod   
    def purge_chain(cls,
                    base_path:str = None,
                    chain:str = chain,
                    node:str = 'alice',
                    sudo = False):
        if base_path == None:
            base_path = cls.resolve_base_path(node=node, chain=chain)
        
        return c.rm(base_path+'/chains/commune/db')
    
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
    def resolve_node_keystore_path(cls, node='alice', chain=chain):
        path =  cls.resolve_base_path(node=node, chain=chain) + f'/chains/commune/keystore'
        if not c.exists(path):
            c.mkdir(path)
        return path

    @classmethod
    def build_spec(cls,
                   chain = chain,
                   disable_default_bootnode: bool = True,
                   snap:bool = False,
                   verbose:bool = True,
                   vali_node_keys:dict = None,
                   mode = mode,
                   ):

        if snap:
            cls.snap()

        chain_spec_path = cls.chain_spec_path(chain)
        chain_release_path = cls.chain_release_path(mode=mode)

        cmd = f'{chain_release_path} build-spec --chain {chain}'
        
        if disable_default_bootnode:
            cmd += ' --disable-default-bootnode'  
        cmd += f' > {chain_spec_path}'
        
        # chain_spec_path_dir = os.path.dirname(chain_spec_path)
        c.print(cmd)
        if mode == 'docker':
            volumes = f'-v {cls.spec_path}:{cls.spec_path}'
            c.cmd(f'docker run {volumes} subspace bash -c "{cmd}"')
        elif mode == 'local':
            c.cmd(f'bash -c "{cmd}"', cwd=cls.chain_path, verbose=True)    


        # ADD THE VALI NODE KEYS

        if vali_node_keys == None:
            vali_node_keys = cls.vali_node_keys(chain=chain)
        spec = c.get_json(chain_spec_path)
        spec['genesis']['runtime']['aura']['authorities'] = [k['aura'] for k in vali_node_keys.values()]
        spec['genesis']['runtime']['grandpa']['authorities'] = [[k['gran'],1] for k in vali_node_keys.values()]
        c.put_json(chain_spec_path, spec)
        resp = {'spec_path': chain_spec_path, 'spec': spec}
        return {'success':True, 'message':'built spec', 'chain':chain}


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


    @classmethod
    def spec_exists(cls, chain):
        return c.exists(f'{cls.spec_path}/{chain}.json')



    @classmethod
    def chain_spec_path(cls, chain = None):
        if chain == None:
            chain = cls.network
        return cls.spec_path + f'/{chain}.json'


    @classmethod
    def new_chain_spec(self, 
                       chain,
                       base_chain:str = chain, 
                       balances : 'List[str, int]' = None,
                       aura_authorities: 'List[str, int]' = None,
                       grandpa_authorities: 'List[str, int]' = None,
                       ):
        base_spec =  self.get_spec(base_chain)
        new_chain_path = f'{self.spec_path}/{chain}.json'
        
        if balances != None:
            base_spec['balances'] = balances
        if aura_authorities != None:
            base_spec['balances'] = aura_authorities
        c.put_json( new_chain_path, base_spec)
        
        return base_spec
    
    new_chain = new_chain_spec

    @classmethod
    def rm_chain(self, chain):
        return c.rm(self.chain_spec_path(chain))
    