
import torch
import scalecodec
from retry import retry
from typing import List, Dict, Union, Optional, Tuple
from substrateinterface import SubstrateInterface
from typing import List, Dict, Union, Optional, Tuple
from commune.utils.network import ip_to_int, int_to_ip
from rich.prompt import Confirm
from commune.modules.subspace.balance import Balance
from commune.modules.subspace.utils import (U16_MAX,  is_valid_address_or_public_key, )
from commune.modules.subspace.chain_data import (ModuleInfo, custom_rpc_type_registry)

import streamlit as st
import json
from loguru import logger
import os
import commune as c

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
    libpath = chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    netuid = default_config['netuid']
    image = 'vivonasg/subspace:latest'
    mode = 'docker'
    
    def __init__( 
        self, 
        network: str = network,
        **kwargs,
    ):
        config = self.set_config(kwargs=locals())
        if config.loop:
            c.thread(self.loop)

    def set_network(self, 
                network:str = network,
                url : str = None,
                websocket:str=None, 
                ss58_format:int=42, 
                type_registry:dict=custom_rpc_type_registry, 
                type_registry_preset=None, 
                cache_region=None, 
                runtime_config=None, 
                use_remote_preset=False,
                ws_options=None, 
                auto_discover=True, 
                auto_reconnect=True, 
                verbose:bool=False,
                max_trials:int = 10,
                **kwargs):

        '''
        A specialized class in interfacing with a Substrate node.

        Parameters
       A specialized class in interfacing with a Substrate node.

        Parameters
        url : the URL to the substrate node, either in format <https://127.0.0.1:9933> or wss://127.0.0.1:9944
        
        ss58_format : The address type which account IDs will be SS58-encoded to Substrate addresses. Defaults to 42, for Kusama the address type is 2
        
        type_registry : A dict containing the custom type registry in format: {'types': {'customType': 'u32'},..}
        
        type_registry_preset : The name of the predefined type registry shipped with the SCALE-codec, e.g. kusama
        
        cache_region : a Dogpile cache region as a central store for the metadata cache
        
        use_remote_preset : When True preset is downloaded from Github master, otherwise use files from local installed scalecodec package
        
        ws_options : dict of options to pass to the websocket-client create_connection function
        : dict of options to pass to the websocket-client create_connection function
                
        '''

        from substrateinterface import SubstrateInterface
        
        if network == None:
            network = self.config.network

        trials = 0
        while trials < max_trials :
            trials += 1
            url = self.resolve_node_url(url=url, chain=network, local=self.config.local)
            c.print(f'Connecting to {url}...')
            ip = c.ip()
            url = url.replace(ip, '0.0.0.0')

            kwargs.update(url=url, 
                        websocket=websocket, 
                        ss58_format=ss58_format, 
                        type_registry=type_registry, 
                        type_registry_preset=type_registry_preset, 
                        cache_region=cache_region, 
                        runtime_config=runtime_config, 
                        ws_options=ws_options, 
                        auto_discover=auto_discover, 
                        auto_reconnect=auto_reconnect)
            try:
                self.substrate= SubstrateInterface(**kwargs)
                break
            except Exception as e:
                c.print(e, url)
                self.config.local = False
                url = None
                
        self.url = url
        self.network = network
        response = {'success': True, 'message': f'Connected to {url}', 'network': network, 'url': url}

        return response
    def __repr__(self) -> str:
        return f'<Subspace: network={self.network}>'
    def __str__(self) -> str:
        return f'<Subspace: network={self.network}>'
    
    def shortyaddy(self, address, first_chars=4):
        return address[:first_chars] + '...' 


    def my_stake(self, search=None, netuid = None, network = None, fmt=fmt,  decimals=2, block=None):
        mystaketo = self.my_staketo(netuid=netuid, network=network, fmt=fmt, decimals=decimals, block=block)
        key2stake = {}
        for key, staketo_tuples in mystaketo.items():
            stake = sum([s for a, s in staketo_tuples])
            key2stake[key] = c.round_decimals(stake, decimals=decimals)
        if search != None:
            key2stake = {k:v for k,v in key2stake.items() if search in k}

        return key2stake
    mys =  mystake = key2stake =  my_stake

    def my_balance(self, search:str=None, netuid:int = 0, network:str = 'main', fmt=fmt,  decimals=2, block=None, min_value:int = 0):

        balances = self.balances(network=network, fmt=fmt, block=block)
        my_balance = {}
        key2address = c.key2address()
        for key, address in key2address.items():
            if address in balances:
                my_balance[key] = balances[address]

        if search != None:
            my_balance = {k:v for k,v in my_balance.items() if search in k}
            
        my_balance = dict(sorted(my_balance.items(), key=lambda x: x[1], reverse=True))

        if min_value > 0:
            my_balance = {k:v for k,v in my_balance.items() if v > min_value}

        return my_balance
        
    key2balance = myb = mybal = my_balance

    def my_staketo(self,search=None, netuid = None, network = None, fmt=fmt,  decimals=2, block=None):
        staketo = self.stake_to(netuid=netuid, network=network, block=block)
        mystaketo = {}
        key2address = c.key2address()
        for key, address in key2address.items():
            if address in staketo:
                mystaketo[key] = [[a, self.format_amount(s, fmt=fmt)] for a, s in staketo[address]]

        if search != None:
            mystaketo = {k:v for k,v in mystaketo.items() if search in k}
            
        return mystaketo
    my_stake_to = my_staketo


    def my_stakefrom(self, 
                    search:str=None, 
                    netuid:int = None, 
                    network:str = None, 
                    fmt:str=fmt,  
                    decimals:int=2):
        staketo = self.stake_from(netuid=netuid, network=network)
        mystakefrom = {}
        key2address = c.key2address()
        for key, address in key2address.items():
            if address in mystakefrom:
                mystakefrom[key] = self.format_amount(mystakefrom[address])
    
        if search != None:
            mystakefrom = {k:v for k,v in mystakefrom.items() if search in k}
        return mystakefrom

    my_stake_from = my_stakefrom

    

    def key2tokens(self, network = None, fmt=fmt, decimals=2):
        key2tokens = {}
        key2balance = self.key2balance(network=network, fmt=fmt, decimals=decimals)
        for key, balance in key2balance.items():
            if key not in key2tokens:
                key2tokens[key] = 0
            key2tokens[key] += balance
            
        for netuid in self.netuids():
            key2stake = self.key2stake(network=network, fmt=fmt, netuid=netuid, decimals=decimals)

            for key, stake in key2stake.items():
                if key not in key2tokens:
                    key2tokens[key] = 0
                key2tokens[key] += stake
            
        return key2tokens

    def network_balance(self, network = None, fmt=fmt, update=False):
        state_dict = self.state_dict(network=network, update=update)
        total_balance = 0
        for key, value in state_dict['balances'].items():
            total_balance += value
        return self.format_amount(total_balance, fmt=fmt)

    def network_stake(self, network = None, fmt=fmt, update=False):
        state_dict = self.state_dict(network=network, update=update)
        total_stake = 0
        for modules in state_dict['modules']:
            for module in modules:
                total_stake += module['stake']
        return self.format_amount(total_stake, fmt=fmt)
    

    def my_total_supply(self, network = None,fmt=fmt, decimals=2):
        return self.my_total_stake(network=network) + self.my_total_balance(network=network)

    my_tokens = my_supply = my_value = my_total_supply

    key2value = key2tokens    
    def my_total_stake(self, network = None, netuid=None, fmt=fmt, decimals=2):
        return sum(self.my_stake(network=network, netuid=netuid, fmt=fmt, decimals=decimals).values())
    def my_total_balance(self, network = None, fmt=fmt, decimals=2):
        return sum(self.my_balance(network=network, fmt=fmt, decimals=decimals).values())

    def names2uids(self, names: List[str]) -> Union[torch.LongTensor, list]:
        # queries updated network state
        current_network_state = self.modules(update=True) 
        uids = []
        for name in names:
            for node in current_network_state:
                if node['name'] == name:
                    uids.append(node['uid'])
                    break

        return torch.LongTensor(uids)
    
    #####################
    #### Set Weights ####
    #####################
    @retry(delay=0, tries=4, backoff=0, max_delay=0)
    def set_weights(
        self,
        key: 'c.key' = None,
        uids: Union[torch.LongTensor, list] = None,
        weights: Union[torch.FloatTensor, list] = None,
        netuid: int = None,
        wait_for_inclusion:bool = True,
        wait_for_finalization:bool = True,
        network = None,
    ) -> bool:
        network = self.resolve_network(network)
        key = self.resolve_key(key)
        netuid = self.resolve_netuid(netuid)
        
        # checking if the "uids" are passed as names -> strings
        if all(isinstance(item, str) for item in uids):
            uids = self.names2uids(names=uids)

        subnet = self.subnet( netuid = netuid )
        min_allowed_weights = subnet['min_allowed_weights']
        max_allowed_weights = subnet['max_allowed_weights']

        if uids is None:
            uids = self.uids()
    
        if len(uids) == 0:
            c.print(f'No uids to vote on.')
            return False
        if len(uids) > max_allowed_weights:
            c.print(f'Only {max_allowed_weights} uids are allowed to be voted on.')
            uids = uids[:max_allowed_weights]

        
        if len(uids) < min_allowed_weights:
            while len(uids) < min_allowed_weights:
                uid = c.choice(list(range(subnet['n'])))
                if uid not in uids:
                    uids.append(uid)
                    weights.append(0)
            
        if weights is None:
            weights = [1 for _ in uids]
        if isinstance(weights, list):
            weights = torch.tensor(weights)

        weights = weights / weights.sum()
        weights = weights * U16_MAX
        weights = weights.tolist()

        # uids = [int(uid) for uid in uids]
        uid2weight = {uid: int(weight) for uid, weight in zip(uids, weights)}
        uids = list(uid2weight.keys())
        weights = list(uid2weight.values())

        params = {'uids': uids,'weights': weights, 'netuid': netuid}
        response = self.compose_call('set_weights',params = params , key=key)
            
        if response['success']:
            return {'success': True, 'weights': weights, 'uids': uids, 'message': 'Set weights'}
        return response

    vote = set_weights

    def get_netuid_for_subnet(self, network: str = None) -> int:
        netuid = self.subnet_namespace.get(network, None)
        return netuid

    def update(self):
        self.sync()


    @classmethod
    def up(cls):
        c.cmd('docker-compose up -d', cwd=cls.chain_path)

    @classmethod
    def enter(cls):
        c.cmd('make enter', cwd=cls.chain_path)

    def register_servers(self, search=None, **kwargs):
        for m in c.servers(network='local'):
            try:
                self.register(name=m)
            except Exception as e:
                c.print(e, color='red')
    reg_servers = register_servers
    def reged_servers(self, **kwargs):
        servers =  c.servers(network='local')
        c.print(servers)
    def register_ghosts(self, n=10, **kwargs):
        ip = c.ip()
        for i in range(n):
            self.register(name=f'ghost{i}',
                         address= ip + ':' + str(8000 + i), 
                          **kwargs)


    def register(
        self,
        name: str , # defaults to module.tage
        address : str = None,
        stake : float = 0,
        subnet: str = None,
        key : str  = None,
        module_key : str = None,
        network: str = network,
        fmt = 'nano'

    ) -> bool:
        
        assert name != None, f"Module name must be provided"

        # resolve the subnet name
        if subnet == None:
            subnet = self.config.subnet

        network =self.resolve_network(network)

        if address == None:
            address = c.namespace(network='local')[name]
            address = address.replace(c.default_ip,c.ip())
        
        module_key = self.resolve_key(name)
        key = self.resolve_key(key)

        # Validate address.
        if self.subnet_exists(subnet, network=network):
            netuid = self.get_netuid_for_subnet(subnet)
            if self.is_registered(module_key.ss58_address, netuid=netuid):
                c.print(f":cross_mark: [red]Module {name} already registered[/red]")
                return self.update_module(module=name, name=name, address=address , netuid=netuid, network=network)
            min_stake = self.min_stake(netuid=netuid, registration=True)

        else: 
            c.print(f"YOU ARE CREATING A NEW SUBNET: {subnet}")
            min_stake = self.min_stake(netuid=0, registration=True)

            
        # convert to nanos
        balance = self.get_balance(key.ss58_address, fmt=fmt)
        if balance < min_stake:
            return {'success': False, 'message': f'Insufficient balance: {balance} < {min_stake}'}
        if stake < min_stake:
            stake = min_stake
        stake = self.to_nanos(stake)

        params = { 
                    'network': subnet.encode('utf-8'),
                    'address': address.encode('utf-8'),
                    'name': name.encode('utf-8'),
                    'stake': stake,
                    'module_key': module_key.ss58_address,
                } 
        # create extrinsic call
        response = self.compose_call('register', params=params, key=key)

        if response['success']:
            response['msg'] = f'Registered {name} with {stake} stake'

        return response

    reg = register

    ##################
    #### Transfer ####
    ##################
    def transfer(
        self,
        dest: str, 
        amount: float , 
        key: str = None,
        network : str = None,
        netuid : int = None,
    ) -> bool:
        
        key = c.get_key(key)
        network = self.resolve_network(network)
        dest = self.resolve_key_ss58(dest)
        account_balance = self.get_balance( key.ss58_address , fmt='j' )
        if amount > account_balance:
            return {'success': False, 'message': f'Insufficient balance: {account_balance}'}

        amount = self.to_nanos(amount) # convert to nano (10^9 nanos = 1 token)
        dest_balance = self.get_balance( dest , fmt='j')

        response = self.compose_call(
            module='Balances',
            fn='transfer',
            params={
                'dest': dest, 
                'value': amount
            },
            key=key
        )

        if response['success']:
            response.update(
                {
                'from': {
                    'address': key.ss58_address,
                    'old_balance': account_balance,
                    'new_balance': self.get_balance( key.ss58_address , fmt='j')
                } ,
                'to': {
                    'address': dest,
                    'old_balance': dest_balance,
                    'new_balance': self.get_balance( dest , fmt='j'),
                }, 
                }
            )
        
        return response



    send = transfer

    def get_existential_deposit(
        self,
        block: Optional[int] = None,
        fmt = 'nano'
    ) -> Optional[Balance]:
        """ Returns the existential deposit for the chain. """
        result = self.query_constant(
            module_name='Balances',
            constant_name='ExistentialDeposit',
            block = block,
        )
        
        if result is None:
            return None
        
        return self.format_amount( result.value, fmt = fmt )
        
    #################
    #### update or replace a module ####
    #################

    def update_module(
        self,
        module: str,
        # params from here
        name: str = None,
        address: str = None,
        delegation_fee: float = None,
        netuid: int = None,
        network : str = network,


    ) -> bool:
        self.resolve_network(network)
        key = self.resolve_key(module)
        netuid = self.resolve_netuid(netuid)  
        module_info = self.get_module(module)

        if name == None:
            name = module
    
        if address == None:
            namespace_local = c.namespace(network='local')
            address = namespace_local.get(name,  f'{c.ip()}:{c.free_port()}'  )
            address = address.replace(c.default_ip, c.ip())
        # Validate that the module is already registered with the same address
        if name == module_info['name'] and address == module_info['address']:
            c.print(f"{c.emoji('check_mark')} [green] [white]{module}[/white] Module already registered and is up to date[/green]:[bold white][/bold white]")
            return {'success': False, 'message': f'{module} already registered and is up to date with your changes'}
        
        # ENSURE DELEGATE FEE IS BETWEEN 0 AND 100
        assert delegation_fee != None, f"Delegate fee must be provided"
        if delegation_fee < 1.0 and delegation_fee > 0:
            delegation_fee = delegation_fee * 100
        assert delegation_fee >= 0 and delegation_fee <= 100, f"Delegate fee must be between 0 and 100"

        params = {
            'netuid': netuid, # defaults to module.netuid
             # PARAMS #
            'name': name, # defaults to module.tage
            'address': address, # defaults to module.tage
            'delegation_fee': delegation_fee, # defaults to module.delegate_fee
        }

        # remove the params that are the same as the module info
        for k in ['name', 'address']:
            if params[k] == module_info[k]:
                params[k] = ''


        reponse  = self.compose_call('update_module',params=params, key=key)

        return reponse



    #################
    #### Serving ####
    #################
    def update_network(
        self,
        netuid: int = None,
        immunity_period: int = None,
        min_allowed_weights: int = None,
        max_allowed_weights: int = None,
        max_allowed_uids: int = None,
        min_stake : int = None,
        tempo: int = None,
        name:str = None,
        founder: str = None,
        wait_for_inclusion: bool = False,
        wait_for_finalization = True,
        key: str = None,
        network = network,
        prompt: bool = False,
    ) -> bool:
            
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        subnet_state = self.subnet( netuid=netuid )
        # infer the key if you have it
        if key == None:
            key2address = self.address2key()
            if subnet_state['founder'] not in key2address:
                return {'success': False, 'message': f"Subnet {netuid} not found in local namespace, please deploy it "}
            key = c.get_key(key2address.get(subnet_state['founder']))
            c.print(f'Using key: {key}')

        # convert to nanos
        if min_stake != None:
            min_stake = self.to_nanos(min_stake)
        
        params = {
            'immunity_period': immunity_period,
            'min_allowed_weights': min_allowed_weights,
            'max_allowed_uids': max_allowed_uids,
            'max_allowed_weights': max_allowed_weights,
            'tempo': tempo,
            'founder': founder,
            'min_stake': min_stake,
            'name': name,
        }

        # remove the params that are the same as the module info
        old_params = {}
        for k, v in params.items():
            old_params[k] = subnet_state[k]
            if v == None:
                params[k] = old_params[k]
                
        params['netuid'] = netuid

        response = self.compose_call(fn='update_network',params=params, key=key)

        return response


    def get_unique_tag(self, module:str, tag:str=None, netuid:int=None, **kwargs):
        if tag == None:
            tag = ''
        name = f'{module}{tag}'
        return self.resolve_unique_server_name(name=name, netuid=netuid, **kwargs).split('::')[-1]

    def resolve_unique_server_names(self, name:str,  n:int=10,   **kwargs) -> List[str]:
        server_names = []
        for i in range(n):
            server_name = self.resolve_unique_server_name(name=name, n=n, avoid_servers=server_names, **kwargs)

            server_names += [server_name]

        return server_names

            



    def resolve_unique_server_name(self, name:str, tag:str = None, netuid:Union[str, int]=None , avoid_servers:List[str]=None , tag_seperator = '::',  **kwargs): 

        cnt = 0
        if tag == None:
            tag = ''
        name = name + tag_seperator + tag
        servers = self.servers(netuid=netuid,**kwargs)
        if avoid_servers == None:
            avoid_servers = []
        servers += avoid_servers
        new_name = name
        while new_name in servers:
            new_name = name + str(cnt)
            cnt += 1

        c.print(new_name)

        return new_name

    def resolve_module_key(self, module_key: str =None, key: str =None, netuid: int = None, name2key:dict = None):
        if module_key == None:
            module_key = c.get_key(module_key)
            module_key = key.ss58_address
            return module_key
        
        # the name matches a key in the subspace namespace
        if name2key == None:
            name2key = self.name2key(netuid=netuid)
        if module_key in name2key:
            module_key = name2key[module_key]
        assert c.is_valid_ss58_address(module_key), f"Module key {module_key} is not a valid ss58 address"
        return module_key

    def transfer_stake(
            self,
            new_module_key: str ,
            module_key: str ,
            amount: Union[Balance, float] = None, 
            key: str = None,
            netuid:int = None,
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = True,
            network:str = None,
            existential_deposit: float = 0.1,
            sync: bool = False
        ) -> bool:
        # STILL UNDER DEVELOPMENT, DO NOT USE
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = c.get_key(key)

        c.print(f':satellite: Staking to: [bold white]SubNetwork {netuid}[/bold white] {amount} ...')
        # Flag to indicate if we are using the wallet's own hotkey.

        name2key = self.name2key(netuid=netuid)
        module_key = self.resolve_module_key(module_key=module_key, netuid=netuid, name2key=name2key)
        new_module_key = self.resolve_module_key(module_key=new_module_key, netuid=netuid, name2key=name2key)

        assert module_key != new_module_key, f"Module key {module_key} is the same as new_module_key {new_module_key}"
        assert module_key in name2key.values(), f"Module key {module_key} not found in SubNetwork {netuid}"
        assert new_module_key in name2key.values(), f"Module key {new_module_key} not found in SubNetwork {netuid}"

        stake = self.get_stakefrom( module_key, from_key=key.ss58_address , fmt='j', netuid=netuid)

        if amount == None:
            amount = stake
        
        amount = self.to_nanos(amount - existential_deposit)
        
        # Get current stake
        params={
                    'netuid': netuid,
                    'amount': int(amount),
                    'module_key': module_key

                    }

        balance = self.get_balance( key.ss58_address , fmt='j')

        response  = self.compose_call('transfer_stake',params=params, key=key)

        if response['success']:
            new_balance = self.get_balance(key.ss58_address , fmt='j')
            new_stake = self.get_stakefrom( module_key, from_key=key.ss58_address , fmt='j', netuid=netuid)
            msg = f"Staked {amount} from {key.ss58_address} to {module_key}"
            return {'success': True, 'msg':msg, 'balance': {'old': balance, 'new': new_balance}, 'stake': {'old': stake, 'new': new_stake}}
        else:
            return  {'success': False, 'msg':response.error_message}



    def stake(
            self,
            module: Optional[str] = None, # defaults to key if not provided
            amount: Union[Balance, float] = None, 
            key: str = None,  # defaults to first key
            netuid:int = None,
            network:str = None,
            existential_deposit: float = 0.01,
        ) -> bool:
        """
        description: 
            Unstakes the specified amount from the module. 
            If no amount is specified, it unstakes all of the amount.
            If no module is specified, it unstakes from the most staked module.
        params:
            amount: float = None, # defaults to all
            module : str = None, # defaults to most staked module
            key : 'c.Key' = None,  # defaults to first key 
            netuid : Union[str, int] = 0, # defaults to module.netuid
            network: str= main, # defaults to main
        return: 
            response: dict
        
        """
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = c.get_key(key)
        name2key = self.name2key(netuid=netuid)
        if module in name2key:
            module_key = name2key[module]

        # Flag to indicate if we are using the wallet's own hotkey.
        old_balance = self.get_balance( key.ss58_address , fmt='j')
        old_stake = self.get_stakefrom( module, from_key=key.ss58_address , fmt='j', netuid=netuid)
        if amount is None:
            amount = old_balance

        amount = int(self.to_nanos(amount - existential_deposit))
        assert amount > 0, f"Amount must be greater than 0 and greater than existential deposit {existential_deposit}"
        
        # Get current stake
        params={
                    'netuid': netuid,
                    'amount': amount,
                    'module_key': module_key
                    }

        response = self.compose_call('add_stake',params=params, key=key)

        if response['success']:
            new_stake = self.get_stakefrom( module_key, from_key=key.ss58_address , fmt='j', netuid=netuid)
            new_balance = self.get_balance(  key.ss58_address , fmt='j')
            response.update({"message": "Stake Sent", "from": key.ss58_address, "to": module_key, "amount": amount, "balance_before": old_balance, "balance_after": new_balance, "stake_before": old_stake, "stake_after": new_stake})

        return response



    def unstake(
            self,
            amount: float =None, # defaults to all of the amount
            module : str = None, # defaults to most staked module
            key : 'c.Key' = None,  # defaults to first key
            netuid : Union[str, int] = 0, # defaults to module.netuid
            network: str= None,
        ) -> dict:
        """
        description: 
            Unstakes the specified amount from the module. 
            If no amount is specified, it unstakes all of the amount.
            If no module is specified, it unstakes from the most staked module.
        params:
            amount: float = None, # defaults to all
            module : str = None, # defaults to most staked module
            key : 'c.Key' = None,  # defaults to first key 
            netuid : Union[str, int] = 0, # defaults to module.netuid
            network: str= main, # defaults to main
        return: 
            response: dict
        
        """
        network = self.resolve_network(network)
        key = c.get_key(key)
        netuid = self.resolve_netuid(netuid)
        old_balance = self.get_balance( key.ss58_address , fmt='j')       
        # get most stake from the module
        staketo = self.get_staketo(netuid=netuid, names = False)

        module_key = None
        if module == None:
            # find the largest staked module
            max_stake = 0
            for k,v in staketo.items():
                if v > max_stake:
                    max_stake = v
                    module_key = k            
        else:
            key2name = self.key2name(netuid=netuid)
            name2key = {key2name[k]:k for k,v in staketo.items()}
            if module in name2key:
                module_key = name2key[module]
            else:
                module_key = module
        
        # we expected to switch the module to the module key
        assert c.is_valid_ss58_address(module_key), f"Module key {module_key} is not a valid ss58 address"
        assert module_key in staketo, f"Module {module_key} not found in SubNetwork {netuid}"
        stake = staketo[module_key]
        amount = amount if amount != None else stake
        # convert to nanos
        params={
            'amount': int(self.to_nanos(amount)),
            'netuid': netuid,
            'module_key': module_key
            }
        response = self.compose_call(fn='remove_stake',params=params, key=key)
        
        if response['success']: # If we successfully unstaked.
            new_balance = self.get_balance( key.ss58_address , fmt='j')
            new_stake = self.get_stakefrom(module_key, from_key=key.ss58_address , fmt='j') # Get stake on hotkey.
            return {
                'success': True,
                'from': {
                    'key': key.ss58_address,
                    'balance_before': old_balance,
                    'balance_after': new_balance,
                },
                'to': {
                    'key': module_key,
                    'stake_before': stake,
                    'stake_after': new_stake
            }
            }

        return response
            
    ########################
    #### Standard Calls ####
    ########################

    """ Queries subspace named storage with params and block. """
    @retry(delay=2, tries=3, backoff=2, max_delay=4)
    def query_subspace( self, name: str,
                       params: Optional[List[object]] = [], 
                        block: Optional[int] = None, 

                       network=None ) -> Optional[object]:
        network = self.resolve_network(network)
        
        with self.substrate as substrate:
            return substrate.query(
                module='SubspaceModule',
                storage_function = name,
                params = params,
                block_hash = None if block == None else substrate.get_block_hash(block)
                )

    def stake_from(self, netuid = None, block=None, network=None):
        network = self.resolve_network(network)
        netuid  = self.resolve_netuid(netuid)
        return {k.value: list(map(list,v.value)) for k,v in self.query_map('StakeFrom', netuid, block=block)}
    
    def delegation_fee(self, netuid = None, block=None, network=None):
        network = self.resolve_network(network)
        netuid  = self.resolve_netuid(netuid)
        return {k.value:v.value for k,v in self.query_map('DelegationFee', netuid, block=block)}
    
    def stake_to(self, netuid = None, network=None, block=None):
        network = self.resolve_network(network)
        netuid  = self.resolve_netuid(netuid)
        return {k.value: list(map(list,v.value)) for k,v in self.query_map('StakeTo', netuid, block=block)}

    """ Queries subspace map storage with params and block. """
    def query_map( self, 
                 name: str, 
                  params: list = None,
                  block: Optional[int] = None, 
                  network:str = None,
                  max_age = 60,
                  page_size=1000,
                  max_results=100000,
                  records = True
                  
                  ) -> Optional[object]:
        network = self.resolve_network(network)

        if params == None:
            params = []

        if params != None and not isinstance(params, list):
            params = [params]
        
        with self.substrate as substrate:
            block_hash = None if block == None else substrate.get_block_hash(block)
            qmap =  substrate.query_map(
                module='SubspaceModule',
                storage_function = name,
                params = params,
                page_size = page_size,
                max_results = max_results,
                block_hash =block_hash
            )

            qmap = [(k,v) for k,v  in qmap]
                
        return qmap
        
    """ Gets a constant from subspace with module_name, constant_name, and block. """
    def query_constant( self, 
                        constant_name: str, 
                       module_name: str = 'SubspaceModule', 
                       block: Optional[int] = None ,
                       network: str = None) -> Optional[object]:
        
        network = self.resolve_network(network)

        with self.substrate as substrate:
            value =  substrate.query(
                module=module_name,
                storage_function=constant_name,
                block_hash = None if block == None else substrate.get_block_hash(block)
            )
            
        return value
            
    def stale_modules(self, *args, **kwargs):
        modules = self.my_modules(*args, **kwargs)
        servers = c.servers(network='local')
        servers = c.shuffle(servers)
        return [m['name'] for m in modules if m['name'] not in servers]

    #####################################
    #### Hyper parameter calls. ####
    #####################################

    """ Returns network ImmunityPeriod hyper parameter """
    def immunity_period (self, netuid: int = None, block: Optional[int] = None, network :str = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query("ImmunityPeriod",params=netuid, block=block ).value


    """ Returns network MinAllowedWeights hyper parameter """
    def min_allowed_weights (self, netuid: int = None, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query("MinAllowedWeights", params=[netuid], block=block).value
    """ Returns network MinAllowedWeights hyper parameter """
    def max_allowed_weights (self, netuid: int = None, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query("MaxAllowedWeights", params=[netuid], block=block).value

    """ Returns network SubnetN hyper parameter """
    def n(self, network = network , netuid: int = None, block: Optional[int] = None ) -> int:
        self.resolve_network(network)
        netuid = self.resolve_netuid( netuid )
        return self.query('N', netuid, block=block ).value

    """ Returns network MaxAllowedUids hyper parameter """
    def max_allowed_uids (self, netuid: int = None, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query('MaxAllowedUids', netuid, block=block ).value

    """ Returns network Tempo hyper parameter """
    def tempo (self, netuid: int = None, block: Optional[int] = None) -> int:
        netuid = self.resolve_netuid( netuid )
        return self.query('Tempo', params=[netuid], block=block).value

    ##########################
    #### Account functions ###
    ##########################
    
    """ Returns network Tempo hyper parameter """
    def stakes(self, netuid: int = None, block: Optional[int] = None, fmt:str='nano') -> int:
        netuid = self.resolve_netuid( netuid )
        return {k.value: self.format_amount(v.value, fmt=fmt) for k,v in self.query_map('Stake', netuid )}

    """ Returns the stake under a coldkey - hotkey pairing """
    
    
    
    def resolve_key_ss58(self, key:str, network='main', netuid:int=0):
        if key == None:
            key = c.get_key(key)
        if isinstance(key, str):
            if c.is_valid_ss58_address(key):
                key_address = key
            else:
                if c.key_exists( key ):
                    key = c.get_key( key )
                    key_address = key.ss58_address
                else:
                    name2key = self.name2key()
                    assert key in name2key, f"Invalid Key {key} as it should have ss58_address attribute."
                    if key in name2key:
                        key_address = name2key[key]
                    else:
   
                        raise Exception(f"Invalid Key {key} as it should have ss58_address attribute.")   
        # if the key has an attribute then its a key
        elif hasattr(key, 'ss58_address'):
            key_address = key.ss58_address
        assert c.is_valid_ss58_address(key_address), f"Invalid Key {key_address} as it should have ss58_address attribute."
        return key_address


    @classmethod
    def resolve_key(cls, key, create:bool = False):
        if key == None:
            key = 'module'
        if isinstance(key, str):
            if c.key_exists( key ):
                key = c.get_key( key )

        assert hasattr(key, 'ss58_address'), f"Invalid Key {key} as it should have ss58_address attribute."
        return key
        

    @classmethod
    def from_nano(cls,x):
        return x / (10**cls.token_decimals)
    to_token = from_nano
    @classmethod
    def to_nanos(cls,x):
        """
        Converts a token amount to nanos
        """
        return x * (10**cls.token_decimals)
    from_token = to_nanos

    @classmethod
    def format_amount(cls, x, fmt='nano', decimals = None):
        if fmt in ['nano', 'n']:
            x =  x
        elif fmt in ['token', 'unit', 'j', 'J']:
            x = cls.to_token(x)
        
        if decimals != None:
            x = c.round_decimals(x, decimals=decimals)

        return x
    
    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j' ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58 )
        netuid = self.resolve_netuid( netuid )
        stake = self.query( 'Stake',params=[netuid, key_ss58], block=block ).value
        return self.format_amount(stake, fmt=fmt)


    def get_staked_modules(self, key : str , netuid=None, **kwargs) -> Optional['Balance']:
        modules = self.modules(netuid=netuid, **kwargs)
        key_address = self.resolve_key_ss58( key )
        staked_modules = {}
        for module in modules:
            for k,v in module['stake_from']:
                if k == key_address:
                    staked_modules[module['name']] = v

        return staked_modules
        

    def get_staketo( self, key: str = None, module_key=None, block: Optional[int] = None, netuid:int = None , fmt='j' , names:bool = True, network=None) -> Optional['Balance']:
        network = self.resolve_network(network)
        key_address = self.resolve_key_ss58( key )
        netuid = self.resolve_netuid( netuid )
        stake_to =  {k.value: self.format_amount(v.value, fmt=fmt) for k, v in self.query( 'StakeTo', params=[netuid, key_address], block=block )}

        if module_key != None:
            module_key = self.resolve_key_ss58( module_key )
            stake_to : int ={ k:v for k, v in stake_to}.get(module_key, 0)

        if names:
            key2name = self.key2name(netuid=netuid)
            stake_to = {key2name[k]:v for k,v in stake_to.items()}
        return stake_to
    

    def get_stakers( self, key: str, block: Optional[int] = None, netuid:int = None , fmt='j' ) -> Optional['Balance']:
        stake_from = self.get_stakefrom(key=key, block=block, netuid=netuid, fmt=fmt)
        key2module = self.key2module(netuid=netuid)
        return {key2module[k]['name'] : v for k,v in stake_from}
        
    def get_stakefrom( self, key: str, from_key=None, block: Optional[int] = None, netuid:int = None, fmt='j'  ) -> Optional['Balance']:
        key = self.resolve_key_ss58( key )
        netuid = self.resolve_netuid( netuid )
        state_from =  [(k.value, self.format_amount(v.value, fmt=fmt)) for k, v in self.query( 'StakeFrom', block=block, params=[netuid, key] )]
 
        if from_key is not None:
            from_key = self.resolve_key_ss58( from_key )
            state_from ={ k:v for k, v in state_from}.get(from_key, 0)

        return state_from
    get_stake_from = get_stakefrom

    def multistake( self, 
                        modules:List[str],
                        amounts:Union[List[str], float, int],
                        key: str = None, 
                        netuid:int = 0,
                        n:str = 10,
                        network: str = None) -> Optional['Balance']:
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        balance = self.get_balance(key=key, fmt='j')
        name2key = self.name2key(netuid=netuid)

        if isinstance(modules, str):
            modules = [m for m in name2key.keys() if modules in m]

        assert len(modules) > 0, f"No modules found with name {modules}"
        modules = modules[:n] # only stake to the first n modules
        # resolve module keys
        for i, module in enumerate(modules):
            if module in name2key:
                modules[i] = name2key[module]
        module_keys = modules
        if isinstance(amounts, (float, int)): 
            amounts = [amounts] * len(modules)

        for i, amount in enumerate(amounts):
            amounts[i] = self.to_nanos(amount)

        assert len(modules) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."

        params = {
            "netuid": netuid,
            "module_keys": module_keys,
            "amounts": amounts
        }

        response = self.compose_call('add_stake_multiple', params=params, key=key)

        return response
                    

    def multiunstake( self, 
                        modules:List[str],
                        amounts:Union[List[str], float, int] = None,
                        key: str = None, 
                        netuid:int = 0,
                        network: str = None) -> Optional['Balance']:
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        stake_to = self.get_staketo(key=key, netuid=netuid, names=False) # name to amount

        if isinstance(modules, str):
            key2name = self.key2name(netuid=netuid)
            name2key = {key2name[k]:k for k in stake_to.keys()}
            modules = [name2key[m] for m in name2key.keys() if modules in m]

        # RESOLVE AMOUNTS
        if amounts == None:
            amounts = [stake_to[m] for m in modules]
        if isinstance(amounts, (float, int)): 
            amounts = [amounts] * len(modules)

        for i, amount in enumerate(amounts):
            amounts[i] = self.to_nanos(amount)


        assert len(modules) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."

        params = {
            "netuid": netuid,
            "module_keys": modules,
            "amounts": amounts
        }

        response = self.compose_call('remove_stake_multiple', params=params, key=key)

        return response
                    

        

    ###########################
    #### Global Parameters ####
    ###########################

    @property
    def block(self, network:str=None, trials=100) -> int:
        return self.get_block(network=network)
   

    @classmethod
    def archived_blocks(cls, network:str=network, reverse:bool = True) -> List[int]:
        # returns a list of archived blocks 
        
        blocks =  [f.split('.B')[-1].split('.json')[0] for f in cls.glob(f'archive/{network}/state.B*')]
        blocks = [int(b) for b in blocks]
        sorted_blocks = sorted(blocks, reverse=reverse)
        return sorted_blocks

    @classmethod
    def oldest_archive_path(cls, network:str=network) -> str:
        oldest_archive_block = cls.oldest_archive_block(network=network)
        assert oldest_archive_block != None, f"No archives found for network {network}"
        return cls.resolve_path(f'state_dict/{network}/state.B{oldest_archive_block}.json')
    @classmethod
    def newest_archive_block(cls, network:str=network) -> str:
        blocks = cls.archived_blocks(network=network, reverse=True)
        return blocks[0]
    @classmethod
    def newest_archive_path(cls, network:str=network) -> str:
        oldest_archive_block = cls.newest_archive_block(network=network)
        return cls.resolve_path(f'archive/{network}/state.B{oldest_archive_block}.json')
    @classmethod
    def oldest_archive_block(cls, network:str=network) -> str:
        blocks = cls.archived_blocks(network=network, reverse=True)
        if len(blocks) == 0:
            return None
        return blocks[-1]

        
    @classmethod
    def loop(cls, 
                network = network,
                netuid:int = 0,
                 interval = {'sync': 1000, 'register': None, 'vali': None, 'update_modules': None},
                 modules = ['model'], 
                 sleep:float=1,
                 remote:bool=True, **kwargs):
        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn('loop', kwargs=kwargs)

        if isinstance(interval, int):
            interval = {'sync': interval, 'register': interval}
        assert isinstance(interval, dict), f"Interval must be an int or dict. Got {interval}"
        assert all([k in interval for k in ['sync', 'register']]), f"Interval must contain keys 'sync' and 'register'. Got {interval.keys()}"

        time_since_last = {k:0 for k in interval}
        
        time_start = c.time()
        while True:
            c.sleep(sleep)
            current_time = c.time()
            time_since_last = {k:int(current_time - time_start) for k in interval}
            time_left = {k:int(interval[k] - time_since_last[k]) if interval[k] != None else None for k in interval }
        


            subspace = cls(network=network, netuid=netuid)

            if  time_left['update_modules'] != None and time_left['update_modules'] > 0:
                c.update(network='local')

            if time_left['sync'] != None and time_left['sync']:
                c.print(subspace.sync(), color='green')

            if time_left['register'] != None and time_left['register']:
                for m in modules:
                    c.print(f"Registering servers with {m} in it on {network}", color='yellow')
                    subspace.register_servers(m ,network=network, netuid=netuid)
                time_since_last['register'] = current_time

            if time_left['vali'] != None and time_left['vali']:
                c.check_valis(network=network)
                time_since_last['vali'] = current_time

            c.print(f"Looping {time_since_last} / {interval}", color='yellow')
    
    state_dict_cache = {}
    def state_dict(self,
                    network=network, 
                    key: Union[str, list]=None, 
                    inlcude_weights:bool=False, 
                    update:bool=False, 
                    verbose:bool=False, 
                    netuids: List[int] = [0],
                    **kwargs):
        # cache and update are mutually exclusive 
        if  update == False:
            c.print('Loading state_dict from cache', verbose=verbose)
            state_dict = self.latest_archive(network=network)
            if len(state_dict) > 0:
                self.state_dict_cache = state_dict


        if len(self.state_dict_cache) == 0 :
            block = self.block
            netuids = self.netuids() if netuids == None else netuids
            state_dict = {'subnets': [self.subnet(netuid=netuid, network=network, block=block, update=True, fmt='nano') for netuid in netuids], 
                        'modules': [self.modules(netuid=netuid, network=network, include_weights=inlcude_weights, block=block, update=True) for netuid in netuids],
                        'stake_to': [self.stake_to(network=network, block=block) for netuid in netuids],
                        'balances': self.balances(network=network, block=block),
                        'block': block,
                        'network': network,
                        }

            path = f'state_dict/{network}.block-{block}-time-{int(c.time())}'
            c.print(f'Saving state_dict to {path}', verbose=verbose)

            
            self.put(path, state_dict) # put it in storage
            self.state_dict_cache = state_dict # update it in memory

        state_dict = c.copy(self.state_dict_cache)
        if key in state_dict:
            return state_dict[key]
        if isinstance(key,list):
            return {k:state_dict[k] for k in key}
        
        return state_dict
    @classmethod
    def ls_archives(cls, network=network):
        if network == None:
            network = cls.network 
        return [f for f in cls.ls(f'state_dict') if os.path.basename(f).startswith(network)]

    
    @classmethod
    def block2archive(cls, network=network):
        paths = cls.ls_archives(network=network)

        block2archive = {int(p.split('-')[-1].split('-time')[0]):p for p in paths if p.endswith('.json') and f'{network}.block-' in p}
        return block2archive
    @classmethod
    def time2archive(cls, network=network):
        paths = cls.ls_archives(network=network)

        block2archive = {int(p.split('time-')[-1].split('.json')[0]):p for p in paths if p.endswith('.json') and f'time-' in p}
        return block2archive

    @classmethod
    def datetime2archive(cls,search=None, network=network):
        time2archive = cls.time2archive(network=network)
        datetime2archive = {c.time2datetime(time):archive for time,archive in time2archive.items()}
        # sort by datetime
        # 
        datetime2archive = {k:v for k,v in sorted(datetime2archive.items(), key=lambda x: x[0])}
        if search != None:
            datetime2archive = {k:v for k,v in datetime2archive.items() if search in k}
        return datetime2archive



    @classmethod
    def latest_archive_path(cls, network=network):
        latest_archive_time = cls.latest_archive_time(network=network)
    
        if latest_archive_time == None:
            return None
        time2archive = cls.time2archive(network=network)
        return time2archive[latest_archive_time]

    @classmethod
    def latest_archive_time(cls, network=network):
        time2archive = cls.time2archive(network=network)
        if len(time2archive) == 0:
            return None
        latest_time = max(time2archive.keys())
        return latest_time

    @classmethod
    def latest_archive_datetime(cls, network=network):
        latest_archive_time = cls.latest_archive_time(network=network)
        assert latest_archive_time != None, f"No archives found for network {network}"
        return c.time2datetime(latest_archive_time)

    @classmethod
    def archive_staleness(self, network=network):
        return c.time() - self.latest_archive_time(network=network)

    @classmethod
    def latest_archive(cls, network=network):
        path = cls.latest_archive_path(network=network)
        if path == None:
            return {}
        return cls.get(path, {})
    
    def sync(self, network=None, remote:bool=True, local:bool=True, save:bool=True):
        network = self.resolve_network(network)
        self.state_dict(update=True, network=network)
        self.namespace(update=True)
        return {'success': True, 'message': f'Successfully saved {network} locally at block {self.block}'}

    def sync_loop(self, interval=60, network=None, remote:bool=True, local:bool=True, save:bool=True):
        while True:
            self.sync(network=network, remote=remote, local=local, save=save)
            c.sleep(interval)

    def subnet_exists(self, subnet:str, network=None) -> bool:
        subnets = self.subnets(network=network)
        return bool(subnet in subnets)

    def subnet_states(self, *args, **kwargs):
        subnet_states = []
        for netuid in self.netuids():
            subnet_state = self.subnet(*args,  netuid=netuid, **kwargs)
            subnet_states.append(subnet_state)
        return subnet_states

    def total_stake(self, network=network, block: Optional[int] = None, netuid:int=None, fmt='j') -> 'Balance':
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        return self.format_amount(self.query( "TotalStake", params=[netuid], block=block, network=network ).value, fmt=fmt)

    def total_balance(self, network=network, block: Optional[int] = None, fmt='j') -> 'Balance':
        return sum(list(self.balances(network=network, block=block, fmt=fmt).values()))

    def total_supply(self, network=network, block: Optional[int] = None, fmt='j') -> 'Balance':
        return self.total_stake(network=network, block=block) + self.total_balance(network=network, block=block, fmt=fmt)

    mcap = market_cap = total_supply
            
    def subnet(self, 
                    netuid=netuid,
                    network = network,
                    update: bool = False,
                    block : Optional[int] = None,
                    cache:bool = False,
                    fmt:str='j') -> list:
        
        if cache and not update:
            subnet_states =  self.state_dict(network=network, key='subnets', update=update )
            if len(subnet_states) > netuid:
                return subnet_states[netuid]
        subnet_stake = self.query( 'TotalStake', params=netuid , block=block).value
        subnet_emission = self.query( 'SubnetEmission', params=netuid, block=block ).value
        subnet_founder = self.query( 'Founder', params=netuid, block=block ).value
        n = self.query( 'N', params=netuid, block=block ).value
        total_stake = self.total_stake(block=block, fmt='nano')

        subnet = {
                'name': self.netuid2subnet(netuid),
                'netuid': netuid,
                'stake': subnet_stake,
                'emission': subnet_emission,
                'n': n,
                'tempo': self.tempo( netuid = netuid , block=block),
                'immunity_period': self.immunity_period( netuid = netuid , block=block),
                'min_allowed_weights': self.min_allowed_weights( netuid = netuid, block=block ),
                'max_allowed_weights': self.max_allowed_weights( netuid = netuid , block=block),
                'max_allowed_uids': self.max_allowed_uids( netuid = netuid , block=block),
                'min_stake': self.min_stake( netuid = netuid , block=block, fmt='nano'),
                'ratio': min(float(subnet_stake / total_stake), 1.00),
                'founder': subnet_founder
            }
        
        for k in ['stake', 'emission', 'min_stake']:
            subnet[k] = self.format_amount(subnet[k], fmt=fmt)

        return subnet
            

    def get_total_subnets( self, block: Optional[int] = None ) -> int:
        return self.query( 'TotalSubnets', block=block ).value      
    
    def get_emission_value_by_subnet( self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return Balance.from_nano( self.query( 'EmissionValues', block=block, params=[ netuid ] ).value )



    def is_registered( self, key: str, netuid: int = None, block: Optional[int] = None) -> bool:
        netuid = self.resolve_netuid( netuid )
        name2key = self.name2key(netuid=netuid)
        if key in name2key:
            key = name2key[key]
        is_reged =  bool(self.query('Uids', block=block, params=[ netuid, key ]).value)


        return is_reged

    def get_uid_for_key_on_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> int:
        return self.query( 'Uids', block=block, params=[ netuid, key_ss58 ] ).value  


    def total_emission( self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return sum(self.emission(netuid=netuid, block=block))


    def regblock(self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return {k.value:v.value for k,v  in self.query_map('RegistrationBlock',params=netuid, block=block ) }




    def age(self, netuid: int = None) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        regblock = self.regblock(netuid=netuid)
        block = self.block
        age = {}
        for k,v in regblock.items():
            age[k] = block - v
        return age



    def in_immunity(self, netuid: int = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        subnet = self.subnet(netuid=netuid)
        age = self.age(netuid=netuid)
        in_immunity = {}
        for k,v in age.items():
            in_immunity[k] = bool(v < subnet['immunity_period'])
        return in_immunity
    def daily_emission(self, netuid: int = None, network = None, block: Optional[int] = None ) -> Optional[float]:
        self.resolve_network(network)
        netuid = self.resolve_netuid( netuid )
        subnet = self.subnet(netuid=netuid)
        return sum([s['emission'] for s in self.stats(netuid=netuid, block=block, df=False)])*self.format_amount(subnet['emission'], fmt='j') 

    def vali_stats(self, netuid: int = None, network = None, block: Optional[int] = None ) -> Optional[float]:
        self.resolve_network(network)
        netuid = self.resolve_netuid( netuid )
        key2uid = self.key2uid(netuid=netuid)
        names = self.names(netuid=netuid)

        vali_stats = []
        dividends = self.dividends(netuid=netuid, block=block)
        emissions = self.emission(netuid=netuid, block=block)
        
    def stats(self, 
              search = None,
              netuid=0,  
              network = network,
              df:bool=True, 
              update:bool = False , 
              local: bool = True,
              cols : list = ['name', 'registered', 'serving',  'emission', 'dividends', 'incentive', 'stake', 'stake_from', 'regblock'],
              sort_cols = ['registered', 'emission', 'stake'],
              fmt : str = 'j',
              **kwargs
              ):

        ip = c.ip()
        modules = self.modules(netuid=netuid, update=update, fmt=fmt, network=network, **kwargs)
        stats = []


        for i, m in enumerate(modules):

            if local and ip not in m['address']:
                continue
            # sum the stake_from
            m['stake_from'] = sum([v for k,v in m['stake_from']][1:])
            m['registered'] = True

            # we want to round these values to make them look nice
            for k in ['emission', 'dividends', 'incentive', 'stake', 'stake_from']:
                m[k] = c.round(m[k], sig=4)

            stats.append(c.copy(m))

        servers = c.servers(network='local')
        for i in range(len(stats)):
            stats[i]['serving'] = bool(stats[i]['name'] in servers)

        df_stats =  c.df(stats)

        if len(stats) == 0:
            return df_stats
        df_stats = df_stats[cols]


        sort_cols = [c for c in sort_cols if c in df_stats.columns]  
        df_stats.sort_values(by=sort_cols, ascending=False, inplace=True)

        if search is not None:
            df_stats = df_stats[df_stats['name'].str.contains(search, case=True)]
        if not df:
            return df_stats.to_dict('records')
        else:
            return df_stats


    def least_useful_module(self, *args, stats=None,  **kwargs):

        if stats == None:
            stats = self.stats(*args, df=False, **kwargs)
        min_stake = 1e10
        min_module = None
        for s in stats:
            if s['emission'] <= min_stake:
                min_stake = s['emission']
                min_module = s['name']
            if min_stake == 0:
                break
        c.print(f"Least useful module is {min_module} with {min_stake} emission.")
        return min_module
    
    def check_servers(self, search=None,  netuid=None, wait_for_server=False):
        cols = ['name', 'registered', 'serving', 'address']
        for m in c.stats(search=search, netuid=netuid, cols=cols, df=False):
            if m['serving'] == False and m['registered'] == True:
                ip = m['address'].split(':')[0]
                if ':' in ip:
                    port = int(m['address'].split(':')[-1])
                else:
                    port = 8888
                c.serve(m['name'], port=port, wait_for_server=wait_for_server)
            if m['serving'] == True and m['registered'] == False:
                self.register(m['name'])
                
    def key_stats(self, 
                key : str , 
                 netuid=netuid, 
                 network = network,
                 fmt='j',
                **kwargs):
        
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key_address = self.resolve_key_ss58(key)
        key_stats = {}
        key_stats['staketo'] =  self.get_staketo(key_address ,netuid=netuid, fmt=fmt)
        key_stats['total_stake'] = sum([v for k,v in key_stats['staketo']])
        key_stats['registered'] = self.is_registered(key_address, netuid=netuid)
        key_stats['balance'] = self.get_balance(key_address, fmt=fmt)
        key_stats['addresss'] = key_address
        return key_stats

        
    



    def get_balance(self, key: str = None , block: int = None, fmt='j', network=None) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        network = self.resolve_network(network)
        key_ss58 = self.resolve_key_ss58( key )
        
        try:
            @retry(delay=2, tries=3, backoff=2, max_delay=4)
            def make_substrate_call_with_retry():
                with self.substrate as substrate:
                    return substrate.query(
                        module='System',
                        storage_function='Account',
                        params=[key_ss58],
                        block_hash = None if block == None else substrate.get_block_hash( block )
                    )
            result = make_substrate_call_with_retry()
        except scalecodec.exceptions.RemainingScaleBytesNotEmptyException:
            c.critical("Your key it legacy formatted, you need to run btcli stake --ammount 0 to reformat it." )

        return  self.format_amount(result.value['data']['free'] , fmt=fmt)

    balance =  get_balance


    def get_balances(self,fmt:str = 'n', network = None, block: int = None, n = None ) -> Dict[str, Balance]:
        
        network = self.resolve_network(network)
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module='System',
                    storage_function='Account',
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        return_dict = {}
        for r in result:
            bal = self.format_amount(int( r[1]['data']['free'].value ), fmt=fmt)
            return_dict[r[0].value] = bal

        # sort by decending balance
        return_dict = {k:v for k,v in sorted(return_dict.items(), key=lambda x: x[1], reverse=True)}
        if isinstance(n, int):
            return_dict = {k:v for k,v in list(return_dict.items())[:n]}
        return return_dict
    
    balances = get_balances
    
    def resolve_network(self, network: Optional[int] = None) -> int:
        if  not hasattr(self, 'substrate'):
            self.set_network(network)

        if network == None:
            network = self.network
        
        return network
    
    def resolve_subnet(self, subnet: Optional[int] = None) -> int:
        if isinstance(subnet, int):
            assert subnet in self.netuids()
            subnet = self.netuid2subnet(netuid=subnet)
        subnets = self.subnets()
        assert subnet in subnets, f"Subnet {subnet} not found in {subnets} for chain {self.chain}"
        return subnet

    @staticmethod
    def _null_module() -> ModuleInfo:
        module = ModuleInfo(
            uid = 0,
            netuid = 0,
            active =  0,
            stake = '0',
            rank = 0,
            emission = 0,
            incentive = 0,
            dividends = 0,
            last_update = 0,
            weights = [],
            bonds = [],
            is_null = True,
            key = "000000000000000000000000000000000000000000000000",
        )
        return module


    def subnets(self, **kwargs) -> Dict[int, str]:
        subnets = [s['name'] for s in self.subnet_states(**kwargs)]
        return subnets
    
    def netuids(self) -> Dict[int, str]:
        return sorted(list(self.subnet_namespace.values()))

    @property
    def subnet_namespace(self, network=network ) -> Dict[str, str]:
        records = self.query_map('SubnetNamespace')
        return {k.value:v.value for k,v in records}

    
    @property
    def subnet_reverse_namespace(self ) -> Dict[str, str]:
        
        return {v:k for k,v in self.subnet_namespace.items()}
    
    def netuid2subnet(self, netuid = None):
        subnet_reverse_namespace = self.subnet_reverse_namespace
        if netuid != None:
            return subnet_reverse_namespace.get(netuid, None)
        return subnet_reverse_namespace
    def subnet2netuid(self,subnet:str = None):
        subnet2netuid = self.subnet_namespace
        if subnet != None:
            return subnet2netuid.get(subnet, None)
        return subnet2netuid
        

    def resolve_netuid(self, netuid: int = None, namespace:dict=None) -> int:
        '''
        Resolves a netuid to a subnet name.
        '''
        if netuid == None:
            # If the netuid is not specified, use the default.
            netuid = self.netuid

        if isinstance(netuid, str):
            # If the netuid is a subnet name, resolve it to a netuid.
            if namespace == None:
                namespace = self.subnet_namespace
            assert netuid in namespace, f"Subnet {netuid} not found in {namespace} for chain {self.chain}"
            netuid = namespacenetuid
        elif isinstance(netuid, int):
            # If the netuid is an integer, ensure it is valid.
            assert netuid in self.netuids(), f"Netuid {netuid} not found in {self.netuids()} for chain {self.chain}"
            
        assert isinstance(netuid, int), "netuid must be an integer"
        return netuid
    
    resolve_net = resolve_subnet = resolve_netuid


    def key2name(self, key: str = None, netuid: int = None) -> str:
        modules = self.keys()
        key2name =  { m['key']: m['name']for m in modules}
        if key != None:
            return key2name[key]
        
    def name2uid(self, name: str = None, netuid: int = None, network: str = None) -> int:
        modules = self.modules(netuid=netuid)
        name2uid =  { m['name']: m['uid']for m in modules}
        if name != None:
            return name2uid[name]
        return name2uid
            
        
    def name2key(self, search:str=None,  netuid: int = None, network=network) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        self.resolve_network(network)
        names = self.names(netuid=netuid)
        keys = self.keys(netuid=netuid)

        name2key =  { n: k for n, k in zip(names, keys)}
        if search != None:
            name2key = {k:v for k,v in name2key.items() if search in k}
            if len(name2key) == 1:
                return list(name2key.keys())[0]
            
        return name2key

    def key2name(self,search=None, netuid: int = None, network=network) -> Dict[str, str]:
        return {v:k for k,v in self.name2key(search=search, netuid=netuid, network=network).items()}
        
    def is_unique_name(self, name: str, netuid=None):
        return bool(name not in self.namespace(netuid=netuid))

    @classmethod
    def node_paths(cls, name=None, chain=chain, mode=mode) -> Dict[str, str]:
        if mode == 'docker':
            paths = c.module('docker').ps(f'subspace.node.{chain}')
        elif mode == 'local':
            paths = c.pm2ls('subspace.node')
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
        
    def name2inc(self, name: str = None, netuid: int = netuid, nonzero_only:bool=True) -> int:
        name2uid = self.name2uid(name=name, netuid=netuid)
        incentives = self.incentive(netuid=netuid)
        name2inc = { k: incentives[uid] for k,uid in name2uid.items() }

        if name != None:
            return name2inc[name]
        
        name2inc = dict(sorted(name2inc.items(), key=lambda x: x[1], reverse=True))


        return name2inc



    def top_valis(self, netuid: int = netuid, n:int = 10, **kwargs) -> Dict[str, str]:
        name2div = self.name2div(name=None, netuid=netuid, **kwargs)
        name2div = dict(sorted(name2div.items(), key=lambda x: x[1], reverse=True))
        return list(name2div.keys())[:n]

    def name2div(self, name: str = None, netuid: int = netuid, nonzero_only: bool = True) -> int:
        name2uid = self.name2uid(name=name, netuid=netuid)
        dividends = self.dividends(netuid=netuid)
        name2div = { k: dividends[uid] for k,uid in name2uid.items() }
    
        if nonzero_only:
            name2div = {k:v for k,v in name2div.items() if v != 0}

        name2div = dict(sorted(name2div.items(), key=lambda x: x[1], reverse=True))
        if name != None:
            return name2div[name]
        return name2div

    @property
    def block_time(self):
        return self.config.block_time


    def get_block(self, network=None, block_hash=None): 
        self.resolve_network(network)
        return self.substrate.get_block( block_hash=block_hash)['header']['number']

    def seconds_per_epoch(self, netuid=None, network=None):
        self.resolve_network(network)
        netuid =self.resolve_netuid(netuid)
        return self.block_time * self.subnet(netuid=netuid)['tempo']

    
    def get_module(self, name:str = None, key=None, netuid=None, **kwargs) -> ModuleInfo:
        if key != None:
            module = self.key2module(key=key, netuid=netuid, **kwargs)
        if name != None:
            module = self.name2module(name=name, netuid=netuid, **kwargs)
            
        return module

    @property
    def null_module(self):
        return {'name': None, 'key': None, 'uid': None, 'address': None, 'stake': 0, 'balance': 0, 'emission': 0, 'incentive': 0, 'dividends': 0, 'stake_to': {}, 'stake_from': {}, 'weight': []}
        
        
    def name2module(self, name:str = None, netuid: int = None, **kwargs) -> ModuleInfo:
        modules = self.modules(netuid=netuid, **kwargs)
        name2module = { m['name']: m for m in modules }
        default = {}
        if name != None:
            return name2module.get(name, self.null_module)
        return name2module
        
        
        
        
        
    def key2module(self, key: str = None, netuid: int = None, default: dict =None, **kwargs) -> Dict[str, str]:
        modules = self.modules(netuid=netuid, **kwargs)
        key2module =  { m['key']: m for m in modules }
        
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return  key2module.get(key_ss58, default if default != None else {})
        return key2module
        
    def module2key(self, module: str = None, **kwargs) -> Dict[str, str]:
        modules = self.modules(**kwargs)
        module2key =  { m['name']: m['key'] for m in modules }
        
        if module != None:
            return module2key[module]
        return module2key
    

    def module2stake(self,*args, **kwargs) -> Dict[str, str]:
        
        module2stake =  { m['name']: m['stake'] for m in self.modules(*args, **kwargs) }
        
        return module2stake
        

    
    
    def server_exists(self, module:str, netuid: int = None, **kwargs) -> bool:
        return bool(module in self.namespace(netuid=netuid, **kwargs))

    def default_module_info(self, **kwargs):
    
        
        module= {
                    'uid': -1,
                    'address': '0.0.0.0:1234',
                    'name': 'NA',
                    'key': 'NA',
                    'emission': 0,
                    'incentive': 0,
                    'dividends': 0,
                    'stake': 0,
                    'balance': 0,
                    
                }

        for k,v in kwargs.items():
            module[k] = v
        
        
        return module  


    @classmethod
    def get_chain_data(cls, key:str, network:str='main', block:int=None, netuid:int=0):
        self = cls(network=network)

        results =  getattr(self, key)(netuid=netuid, block=block)
        c.print(f"Got {key} for netuid {netuid} at block {block}")
        return results
              
    def modules(self,
                search=None,
                network = 'main',
                netuid: int = 0,
                block: Optional[int] = None,
                fmt='nano', 
                keys = None,
                update: bool = False,
                include_weights = False,
                df = False,
                multithread:bool = False ,
                timeout:int=200, # for multi-threading
                include_balances = False
                ) -> Dict[str, ModuleInfo]:
        import inspect

        cache_path = f'modules/{network}.{netuid}'

        modules = []
        if not update :
            modules = self.get(cache_path, [])

        if len(modules) == 0:

            network = self.resolve_network(network)
            netuid = self.resolve_netuid(netuid)
            block = self.block if block == None else block
            
            keys = ['uid2key', 'addresses', 'names', 'emission', 
                    'incentive', 'dividends', 'regblock', 'last_update', 
                    'stake_from', 'delegation_fee']
            
            if include_balances:
                keys += ['balances']
            if include_weights:
                keys += ['weights']
            if multithread:
                executor = c.module('executor')(max_workers=len(keys))
                futures = []
                for key in keys:
                    futures += [executor.submit(self.get_chain_data, kwargs=dict(key=key, netuid=netuid, block=block, network=network), timeout=timeout)]

                results  = c.wait(futures)
                state = {key: result  for key, result in zip(keys, results)}
            else: 
                state = {}

                for key in c.tqdm(keys):
                    func = getattr(self, key)
                    args = inspect.getfullargspec(func).args

                    kwargs = {}
                    if 'netuid' in args:
                        kwargs['netuid'] = netuid
                    if 'block' in args:
                        kwargs['block'] = block

                    state[key] = func(**kwargs)
            for uid, key in state['uid2key'].items():

                module= {
                    'uid': uid,
                    'address': state['addresses'][uid],
                    'name': state['names'][uid],
                    'key': key,
                    'emission': state['emission'][uid],
                    'incentive': state['incentive'][uid],
                    'dividends': state['dividends'][uid],
                    'stake_from': state['stake_from'].get(key, []),
                    'regblock': state['regblock'].get(uid, 0),
                    'last_update': state['last_update'][uid],
                    'delegation_fee': state['delegation_fee'].get(key, 20)
                }
                module['stake'] = sum([v for k,v in module['stake_from']])
                
                if include_weights:
                    if hasattr(state['weights'][uid], 'value'):
                        
                        module['weight'] = state['weights'][uid].value
                    elif isinstance(state['weights'][uid], list):
                        module['weight'] =state['weights'][uid]
                    else: 
                        raise Exception(f"Invalid weight for module {uid}")

                if include_balances:
                    module['balance'] = state['balances'].get(key, 0)
                modules.append(module)

            self.put(cache_path, modules)

        if len(modules) > 0:
            keys = list(modules[0].keys())
            if isinstance(keys, str):
                keys = [keys]
            keys = list(set(keys))
            for i, module in enumerate(modules):
                modules[i] ={k: module[k] for k in keys}

                for k in ['emission', 'stake']:
                    module[k] = self.format_amount(module[k], fmt=fmt)

                for k in ['incentive', 'dividends']:
                    module[k] = module[k] / (U16_MAX)
                
                module['stake_from']= [(k, self.format_amount(v, fmt=fmt))  for k, v in module['stake_from']]
      
                if include_balances:
                    module['balance'] = self.format_amount(module['balance'], fmt=fmt)

                modules[i] = module
        if search != None:
            modules = [m for m in modules if search in m['name']]

        if df:
            modules = c.df(modules)



        return modules
    

    def my_modules(self,search=None, *args, **kwargs):
        my_modules = []
        address2key = c.address2key()
        for module in self.modules(*args, **kwargs):
            if search != None and search not in module['name']:
                continue
            if module['key'] in address2key:
                my_modules += [module]
            
        return my_modules

    def my_servers(self, search=None,  **kwargs):
        servers = [m['name'] for m in self.my_modules(**kwargs)]
        if search != None:
            servers = [s for s in servers if search in s]
        return servers
    
    def my_modules_names(self, *args, **kwargs):
        my_modules = self.my_modules(*args, **kwargs)
        return [m['name'] for m in my_modules]

    def my_module_keys(self, *args,  **kwargs):
        modules = self.my_modules(*args, **kwargs)
        return [m['key'] for m in modules]

    def my_keys(self, *args, mode='all' , **kwargs):

        modules = self.my_modules(*args,**kwargs)
        address2module = {m['key']: m for m in modules}
        address2balances = self.balances()
        keys = []
        address2key = c.address2key()
        for address, key in address2key.items():
            
            if mode == 'live' and (address in address2module):
                keys += [key]
            elif mode == 'dead' and (address not in address2module and address in address2balances):
                keys += [key]
            elif mode == 'all' and (address in address2module or address in address2balances):
                keys += [key]
            
        return keys

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

    def min_stake(self, netuid: int = None, network: str = network, fmt:str='j', registration=True, **kwargs) -> int:
        netuid = self.resolve_netuid(netuid)
        min_stake = self.query('MinStake', params=[netuid], network=network, **kwargs).value
        min_stake = self.format_amount(min_stake, fmt=fmt)
        if registration:
            registrations_per_block = self.registrations_per_block(netuid=netuid)
            max_registrations_per_block = self.max_registrations_per_block(netuid=netuid)
            
            # 2 to the power of the number of registrations per block over the max registrations per block
            # this is the factor by which the min stake is multiplied, to avoid ddos attacks
            min_stake_factor = 2 **(registrations_per_block // max_registrations_per_block)
            return min_stake * min_stake_factor
        return min_stake


    def registrations_per_block(self, netuid: int = None, network: str = network, fmt:str='j', **kwargs) -> int:
        netuid = self.resolve_netuid(netuid)
        return self.query('RegistrationsPerBlock', params=[], network=network, **kwargs).value
    regsperblock = registrations_per_block

    
    def max_registrations_per_block(self, netuid: int = None, network: str = network, fmt:str='j', **kwargs) -> int:
        netuid = self.resolve_netuid(netuid)
        return self.query('MaxRegistrationsPerBlock', params=[], network=network, **kwargs).value
    max_regsperblock = max_registrations_per_block

    
    def query(self, name:str,  params = None, block=None,  network: str = network, module:str='SubspaceModule'):
        if params == None:
            params = []
        if not isinstance(params, list):
            params = [params]
        self.resolve_network(network)
        with self.substrate as substrate:
            value =  substrate.query(
                module='SubspaceModule',
                storage_function = name,
                block_hash = None if block == None else substrate.get_block_hash(block), 
                params = params
            )
            
        return value
        

        
    
    @classmethod
    def test_chain(cls, chain:str = chain, verbose:bool=True, snap:bool=False ):

        cls.cmd('cargo test', cwd=cls.chain_path, verbose=verbose) 
        

    @classmethod
    def gen_key(cls, *args, **kwargs):
        return c.module('key').gen(*args, **kwargs)
    

    def keys(self, netuid = None, **kwargs):
        return list(self.uid2key(netuid=netuid, **kwargs).values())
    def uids(self, netuid = None, **kwargs):
        return list(self.uid2key(netuid=netuid, **kwargs).keys())

    def uid2key(self, uid=None, netuid = None, **kwargs):
        netuid = self.resolve_netuid(netuid)
        uid2key = {v[0].value: v[1].value for v in self.query_map('Keys', params=[netuid], **kwargs)}
        # sort by uid
        if uid != None:
            return uid2key[uid]
        uids = list(uid2key.keys())
        uid2key = {uid: uid2key[uid] for uid in sorted(uids)}
        return uid2key

    def uid2name(self, netuid: int = None, **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = {v[0].value: v[1].value for v in self.query_map('Names', params=[netuid], **kwargs)}
        names = {k: names[k] for k in sorted(names)}
        return names
    def names(self, netuid: int = None, **kwargs) -> List[str]:
        return list(self.uid2name(netuid=netuid, **kwargs).values())

    def addresses(self, netuid: int = None, **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = {v[0].value: v[1].value for v in self.query_map('Address', params=[netuid], **kwargs)}
        names = list({k: names[k] for k in sorted(names)}.values())
        return names

    def namespace(self, search=None, netuid: int = netuid, network=network, update:bool = True,**kwargs) -> Dict[str, str]:
        cache_path = f'namespace/{network}.{netuid}'
        if update:
            namespace = {}
        else:
            namespace = self.get(cache_path, default={}, **kwargs)
        if len(namespace) == 0:
            self.resolve_network(network)
            names = self.names(netuid=netuid, **kwargs)
            addresses = self.addresses(netuid=netuid, **kwargs)
            namespace = dict(zip(names, addresses))
            self.put(cache_path, namespace)
        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k}
        return namespace
    

    
    def registered_keys(self, netuid = None, **kwargs):
        keys = self.keys(netuid=netuid, **kwargs)
        address2key = c.address2key()
        registered_keys = []
        for k_addr in keys:
            if k_addr in address2key:
                registered_keys += [address2key[k_addr]]
        return registered_keys

    def registered_servers(self, netuid = None, network = network,  **kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        servers = c.servers(network='local')
        registered_keys = []
        for s in servers:
            if self.is_registered(s, netuid=netuid):
                registered_keys += [s]
        return registered_keys
    reged = reged_servers = registered_servers

    def unregistered_servers(self, netuid = None, network = network,  **kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        network = self.resolve_network(network)
        servers = c.servers(network='local')
        unregistered_keys = []
        for s in servers:
            if not self.is_registered(s, netuid=netuid):
                unregistered_keys += [s]
        return unregistered_keys

    
    def check_reged(self, netuid = None, network = network,  **kwargs):
        reged = self.reged(netuid=netuid, network=network, **kwargs)
        jobs = []
        for module in reged:
            job = c.call(module=module, fn='info',  network='subspace', netuid=netuid, return_future=True)
            jobs += [job]

        results = dict(zip(reged, c.gather(jobs)))

        return results 


    
    unreged = unreged_servers = unregistered_servers
                
    def most_valuable_key(self, **kwargs):
        my_balance = self.my_balance( **kwargs)
        return  dict(sorted(my_balance.items(), key=lambda item: item[1]))

    def most_staketo_key(self, key, netuid = 0,  **kwargs):
        staketo = self.get_staketo(key, netuid=netuid, names=False, **kwargs)
        most_stake = 0
        most_stake_key = None
        for k, v in staketo.items():
            if v > most_stake:
                most_stake = v
                most_stake_key = k
        return {'key': most_stake_key, 'stake': most_stake}

    reged = registered_keys
    
    def weights(self, netuid = None, **kwargs) -> list:
        netuid = self.resolve_netuid(netuid)
        subnet_weights =  self.query_map('Weights', netuid, **kwargs)
        weights = {uid.value:list(map(list, w.value)) for uid, w in subnet_weights if w != None and uid != None}
        uids = self.uids(netuid=netuid, **kwargs)
        weights = {uid: weights[uid] if uid in weights else [] for uid in uids}
        return {uid: w for uid, w in weights.items()}
    
    def num_voters(self, netuid = None, **kwargs) -> list:
        weights = self.weights(netuid=netuid, **kwargs)
        return len({k:v for k,v in weights.items() if len(v) > 0})
            
        
    def regprefix(self, prefix, netuid = None, network=None, **kwargs):
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        c.servers(network=network, prefix=prefix)
        
    
    def emission(self, netuid = netuid, network=None, nonzero=False, **kwargs):
        emissions = [v.value for v in self.query('Emission', params=[netuid], network=network, **kwargs)]
        if nonzero:
            emissions = [e for e in emissions if e > 0]
        return emissions
        
    def nonzero_emission(self, netuid = netuid, network=None, **kwargs):
        emission = self.emission(netuid=netuid, network=network, **kwargs)
        nonzero_emission =[e for e in emission if e > 0]
        return len(nonzero_emission)

    def incentive(self, netuid = netuid, block=None,   network=network, **kwargs):
        return [v.value for v in self.query('Incentive', params=netuid, network=network, block=block, **kwargs)]
        
    def trust(self, netuid = netuid, network=None, nonzero=False, **kwargs):
        trust = [v.value for v in self.query('Trust', params=netuid, network=network, **kwargs)]
        if nonzero:
            trust = [t for t in trust if t > 0]
        return trust
    def last_update(self, netuid = netuid, block=None,   network=network, **kwargs):
        return [v.value for v in self.query('LastUpdate', params=[netuid], network=network, block=block, **kwargs)]
        
    def dividends(self, netuid = netuid, network=None, nonzero=False, **kwargs):
        dividends =  [v.value for v in self.query('Dividends', params=netuid, network=network,  **kwargs)]
        if nonzero:
            dividends = {i:d for i,d in enumerate(dividends) if d > 0}
        return dividends

    def registration_blocks(self, netuid: int = None, network:str=  None, nonzeros_only:bool = True,  **kwargs):
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        
        registration_blocks = self.query_map('RegistrationBlock', netuid, **kwargs)
        registration_blocks = {k.value:v.value for k, v in registration_blocks if k != None and v != None}
        # filter based on key of registration_blocks
        registration_blocks = {uid:regblock for uid, regblock in sorted(list(registration_blocks.items()), key=lambda v: v[0])}
        registration_blocks =  list(registration_blocks.values())
        if nonzeros_only:
            registration_blocks = [r for r in registration_blocks if r > 0]
        return registration_blocks

    regblocks = registration_blocks


    def key2uid(self, key = None, network:str=  None ,netuid: int = None, **kwargs):
        key2uid =  {v:k for k,v in self.uid2key(network=network, netuid=netuid, **kwargs).items()}
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return key2uid[key_ss58]
        return key2uid
        


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

 

    def get_archive_blockchain_archives(self, netuid=netuid, network:str=network, **kwargs) -> List[str]:

        datetime2archive =  self.datetime2archive(network=network, **kwargs) 
        break_points = []
        last_block = 10e9
        blockchain_id = 0
        get_archive_blockchain_ids = []
        for dt, archive_path in enumerate(datetime2archive):
            
            archive_block = int(archive_path.split('block-')[-1].split('-')[0])
            if archive_block < last_block :
                break_points += [archive_block]
                blockchain_id += 1
            last_block = archive_block
            get_archive_blockchain_ids += [{'blockchain_id': blockchain_id, 'archive_path': archive_path, 'block': archive_block}]

            c.print(archive_block, archive_path)

        return get_archive_blockchain_ids



    def get_archive_blockchain_info(self, netuid=netuid, network:str=network, **kwargs) -> List[str]:

        datetime2archive =  self.datetime2archive(network=network, **kwargs) 
        break_points = []
        last_block = 10e9
        blockchain_id = 0
        get_archive_blockchain_info = []
        for i, (dt, archive_path) in enumerate(datetime2archive.items()):
            c.print(archive_path)
            archive_block = int(archive_path.split('block-')[-1].split('-time')[0])
            
            c.print(archive_block < last_block, archive_block, last_block)
            if archive_block < last_block :
                break_points += [archive_block]
                blockchain_id += 1
                blockchain_info = {'blockchain_id': blockchain_id, 'archive_path': archive_path, 'block': archive_block, 'earliest_block': archive_block}
                get_archive_blockchain_info.append(blockchain_info)
                c.print(archive_block, archive_path)
            last_block = archive_block
            if len(break_points) == 0:
                continue


        return get_archive_blockchain_info


    


            

    @classmethod
    def most_recent_archives(cls,):
        archives = cls.search_archives()
        return archives
    
    @classmethod
    def num_archives(cls, *args, **kwargs):
        return len(cls.datetime2archive(*args, **kwargs))

    @classmethod
    def search_archives(cls, 
                    lookback_hours : int = 10,
                    end_time :str = 'now', 
                    start_time: Optional[Union[int, str]] = None, 
                    netuid=0, 
                    n = 1000,
                    **kwargs):


        if end_time == 'now':
            end_time = c.time()
        elif isinstance(end_time, str):
            c.print(end_time)
            
            end_time = c.datetime2time(end_time)
        elif isinstance(end_time, int):
            pass
        else:
            raise Exception(f'Invalid end_time {end_time}')
            end_time = c.time2datetime(end_time)



        if start_time == None:
            start_time = end_time - lookback_hours*3600
            start_time = c.time2datetime(start_time)

        if isinstance(start_time, int) or isinstance(start_time, float):
            start_time = c.time2datetime(start_time)
        
        if isinstance(end_time, int) or isinstance(end_time, float):
            end_time = c.time2datetime(end_time)
        

        assert end_time > start_time, f'end_time {end_time} must be greater than start_time {start_time}'
        datetime2archive = cls.datetime2archive()
        datetime2archive= {k: v for k,v in datetime2archive.items() if k >= start_time and k <= end_time}
        c.print(len(datetime2archive))
        factor = len(datetime2archive)//n
        if factor == 0:
            factor = 1
        archives = []

        c.print('Searching archives from', start_time, 'to', end_time)

        cnt = 0
        for i, (archive_dt, archive_path) in enumerate(datetime2archive.items()):
            if i % factor != 0:
                continue
            archive_block = int(archive_path.split('block-')[-1].split('-time')[0])
            archive = c.get(archive_path)
            total_balances = sum([b for b in archive['balances'].values()])
            # st.write(archive['modules']netuid[:3])
            total_stake = sum([sum([_[1]for _ in m['stake_from']]) for m in archive['modules'][netuid]])
            subnet = archive['subnets'][netuid]
            row = {
                    'block': archive_block,  
                    'total_stake': total_stake*1e-9,
                    'total_balance': total_balances*1e-9, 
                    'market_cap': (total_stake+total_balances)*1e-9 , 
                    'dt': archive_dt, 
                    'block': archive['block'], 
                    'path': archive_path, 
                    'mcap_per_block': 0,
                }
            
            if len(archives) > 0:
                denominator = ((row['block']//subnet['tempo']) - (archives[-1]['block']//subnet['tempo']))*subnet['tempo']
                if denominator > 0:
                    row['mcap_per_block'] = (row['market_cap'] - archives[-1]['market_cap'])/denominator

            archives += [row]
            
        return archives

    @classmethod
    def archive_history(cls, *args, 
                     network=network, 
                     netuid= 0 , update=True,  **kwargs):
        path = f'history/{network}.{netuid}.json'

        archive_history = []
        if not update:
            archive_history = cls.get(path, [])
        if len(archive_history) == 0:
            archive_history =  cls.search_archives(*args,network=network, netuid=netuid, **kwargs)
            cls.put(path, archive_history)
            
        
        return archive_history
        

        

        
        
        

    @classmethod
    def dashboard(cls):
        import streamlit as st
        block = 7014
        netuid = 0
        c.module('subspace.dashboard').dashboard()
        


    @classmethod
    def st_search_archives(cls,
                        start_time = '2023-09-08 04:00:00', 
                        end_time = '2023-09-08 04:30:00'):
        start_time = st.text_input('start_time', start_time)
        end_time = st.text_input('end_time', end_time)
        df = cls.search_archives(end_time=end_time, start_time=start_time)

        
        st.write(df)

    
    snapshot_path = f'{chain_path}/snapshots'

    def compose_call(self,
                     fn:str, 
                    params:dict = None, 
                    key:str = None,
                    module:str = 'SubspaceModule', 
                    wait_for_inclusion: bool = True,
                    wait_for_finalization: bool = True,
                    process_events : bool = True,
                    color: str = 'yellow',
                    verbose: bool = True,
                     **kwargs):

        """
        Composes a call to a Substrate chain.

        """
        params = {} if params == None else params
        key = self.resolve_key(key)
        if verbose:
            c.print('params', params, color=color)
            kwargs = c.locals2kwargs(locals())
            kwargs['verbose'] = False
            with c.status(f":satellite: Calling [bold]{fn}[/bold] on [bold yellow]{self.network}[/bold yellow]"):
                return self.compose_call(**kwargs)


        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module=module,
                call_function=fn,
                call_params=params
            )
            extrinsic = substrate.create_signed_extrinsic(call=call,keypair=key)

            response = substrate.submit_extrinsic(extrinsic=extrinsic,
                                                  wait_for_inclusion=wait_for_inclusion, 
                                                  wait_for_finalization=wait_for_finalization)

    
        if process_events:
            response.process_events()

        if response.is_success:
            return {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.network} with key {key}'}
        else:
            return {'success': False, 'error': response.error_message, 'msg': f'Failed to call {module}.{fn} on {self.network} with key {key}'}

    @classmethod
    def convert_snapshot(cls, from_version=1, to_version=2, network=network):
        
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

        else:
            raise Exception(f'Invalid conversion from {from_version} to {to_version}')
    @classmethod
    def snapshot(cls, network=network) -> dict:
        path = f'{self.snapshot_path}/{network}.json'
        return c.get_json(path)


    @classmethod
    def build_snapshot(cls, 
              path : str  = None,
             network : str =network,
             subnet_params : List[str] =  ['name', 'tempo', 'immunity_period', 'min_allowed_weights', 'max_allowed_weights', 'max_allowed_uids', 'founder'],
            module_params : List[str] = ['key', 'name', 'address'],
            save: bool = True, 
            min_balance:int = 100000,
            verbose: bool = False,
            sync: bool = True,
            version: str = 2,
             **kwargs):
        if sync:
            c.sync(network=network)

        path = path if path != None else cls.latest_archive_path(network=network)
        state = cls.get(path)
        
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
            snapshot_path = f'{cls.snapshot_path}/{network}.json'
            c.print('Saving snapshot to', snapshot_path, verbose=verbose)
            c.put_json(snapshot_path, snap)
        # c.print(snap['modules'][0][0])

        date = c.time2date(int(path.split('-')[-1].split('.')[0]))
        
        return {'success': True, 'msg': f'Saved snapshot to {snapshot_path} from {path}', 'date': date}    
    
    snap = build_snapshot
    
    @classmethod
    def check(cls, netuid=0):
        self = cls()

        # c.print(len(self.modules()))
        c.print(len(self.query_map('Keys', netuid)), 'keys')
        c.print(len(self.query_map('Names', netuid)), 'names')
        c.print(len(self.query_map('Address', netuid)), 'address')
        c.print(len(self.incentive()), 'incentive')
        c.print(len(self.uids()), 'uids')
        c.print(len(self.stakes()), 'stake')
        c.print(len(self.query_map('Emission')[0][1].value), 'emission')
        c.print(len(self.query_map('Weights', netuid)), 'weights')

    def vote_pool(self, netuid=None, network=None):
        my_modules = self.my_modules(netuid=netuid, network=network, names_only=True)
        for m in my_modules:
            c.vote(m, netuid=netuid, network=network)
        return {'success': True, 'msg': f'Voted for all modules {my_modules}'}


    
    @classmethod
    def snapshots(cls):
        return list(cls.snapshot_map().keys())

    @classmethod
    def snapshot_map(cls):
        return {l.split('/')[-1].split('.')[0]: l for l in c.ls(f'{cls.chain_path}/snapshots')}
        
    
    @classmethod
    def install_rust(cls, sudo=True):
        c.cmd(f'chmod +x scripts/install_rust_env.sh',  cwd=cls.chain_path, sudo=sudo)

    @classmethod
    def build(cls, chain:str = chain, 
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
    def build_image(cls):
        c.build_image('subspace')
        return {'success': True, 'msg': 'Built subspace image'}
    @classmethod
    def prune_node_keys(cls, max_valis:int=6, chain=chain):

        keys = c.keys(f'subspace.node.{chain}.vali')
        rm_keys = []
        for key in keys:
            if int(key.split('.')[-2].split('_')[-1]) > max_valis:
                rm_keys += [key]
        for key in rm_keys:
            c.rm_key(key)
        return rm_keys
        
        
    @classmethod
    def add_node_keys(cls, chain:str=chain, valis:int=24, nonvalis:int=16, refresh:bool=False , mode=mode):
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

    node_key_prefix = 'subspace.node'
    
    @classmethod
    def rm_node_key(cls,node, chain=chain):
        base_path = cls.resolve_base_path(node=node, chain=chain)
        if c.exists(base_path):
            c.rm(base_path)
        for key in cls.node_key_paths(node=node, chain=chain):
            c.print(f'removing node key {key}')
            c.rm_key(key)
        return {'success':True, 'message':'removed all node keys', 'chain':chain, 'keys_left':cls.node_keys(chain=chain)}
    
        
    @classmethod
    def resolve_node_path(cls, node:str='alice', chain=chain, tag_seperator='_'):
        node = str(node)
        return f'{cls.node_key_prefix}.{chain}.{node}'

    @classmethod
    def get_node_key(cls, node='alice', chain=chain, vali=True, crease_if_not_exists:bool=True):
        if crease_if_not_exists:
            if not cls.node_exists(node=node, chain=chain, vali=vali):
                cls.add_node_key(node=node, chain=chain)
        return cls.node_keys(chain=chain)[node]
    
    @classmethod
    def node_key_paths(cls, node=None, chain=chain):
        key = f'{cls.node_key_prefix}.{chain}.{node}'
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
            c.print(k)
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
    def node_key_mems(cls,node = None, chain=chain):
        vali_node_keys = {}
        for key_name in c.keys(f'{cls.node_key_prefix}.{chain}.{node}'):
            name = key_name.split('.')[-2]
            role = key_name.split('.')[-1]
            key = c.get_key(key_name)
            if name not in vali_node_keys:
                vali_node_keys[name] = { }
            vali_node_keys[name][role] =  key.mnemonic

        if node in vali_node_keys:
            return vali_node_keys[node]
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
    def nodes(cls, chain=chain):
        node_infos = cls.node_infos(chain=chain)
        nodes = list(node_infos.keys())
        return sorted(nodes, key=lambda n: int(n.split('_')[-1]) if n.split('_')[-1].isdigit() else 10e9)

    @classmethod
    def vali_nodes(cls, chain=chain):
        return [k for k,v in cls.node_infos(chain=chain).items() if v['validator']]

    @classmethod
    def nonvali_nodes(cls, chain=chain):
        return [k for k,v in cls.node_infos(chain=chain).items() if not v['validator']]

    public_nodes = nonvali_nodes


    @classmethod
    def rm_nonvali_nodes(cls, chain=chain):
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
        c.print(path)
        return len(c.keys(path+'.')) > 0

    @classmethod
    def add_node_key(cls,
                     node:str,
                     mode = mode,
                     chain = chain,
                     key_mems:dict = {'aura': None, 'gran': None}, # pass the keys mems
                     refresh: bool = False,
                     insert_key:bool = False,
                     ):
        '''
        adds a node key
        '''
        cmds = []

        node = str(node)

        c.print(f'adding node key {node} for chain {chain}')
        if  cls.node_key_exists(node=node, chain=chain):
            if refresh:
                cls.rm_node_key(node=node, chain=chain)
            else:
                c.print(f'node key {node} for chain {chain} already exists')
                return {'success':False, 'msg':f'node key {node} for chain {chain} already exists'}
        chain_path = cls.chain_release_path(mode=mode)
        for key_type in ['gran', 'aura']:
            # we need to resolve the schema based on the key type
            if key_type == 'gran':
                schema = 'Ed25519'
            elif key_type == 'aura':
                schema = 'Sr25519'

            # we need to resolve the key path based on the key type
            key_path = f'{cls.node_key_prefix}.{chain}.{node}.{key_type}'

            if key_mems != None:
                assert key_type in key_mems, f'key_type {key_type} not in keys {key_mems}'
                c.add_key(key_path, mnemonic = key_mems[key_type], refresh=True, crypto_type=schema)

            # we need to resolve the key based on the key path
            key = c.get_key(key_path,crypto_type=schema, refresh=refresh)

            # do we want
            if insert_key:
                # we need to resolve the base path based on the node and chain
                base_path = cls.resolve_base_path(node=node, chain=chain)
                cmd  = f'''{chain_path} key insert --base-path {base_path} --chain {chain} --scheme {schema} --suri "{key.mnemonic}" --key-type {key_type}'''
                # c.print(cmd)
                if mode == 'docker':
                    container_base_path = base_path.replace(cls.chain_path, '/subspace')
                    volumes = f'-v {container_base_path}:{base_path}'
                    cmd = f'docker run {volumes} {cls.image} {cmd}'
                    c.print(c.cmd(cmd, verbose=True))
                elif mode == 'local':
                    c.cmd(cmd, verbose=True, cwd=cls.chain_path)
                else:
                    raise ValueError(f'Unknown mode {mode}, must be one of docker, local')

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

    @classmethod
    def keystore_keys(cls, node='vali_0', chain=chain):
        return [f.split('/')[-1] for f in c.ls(cls.keystore_path(node=node, chain=chain))]

    @classmethod
    def build_spec(cls,
                   chain = chain,
                   disable_default_bootnode: bool = True,
                   vali_node_keys:dict = None,
                   return_spec:bool = False,
                   mode : str = mode,
                   valis: int = 12,
                   ):

        chain_spec_path = cls.chain_spec_path(chain)
        chain_release_path = cls.chain_release_path(mode=mode)

        cmd = f'{chain_release_path} build-spec --chain {chain}'
        
        if disable_default_bootnode:
            cmd += ' --disable-default-bootnode'  
       
        
        # chain_spec_path_dir = os.path.dirname(chain_spec_path)
        if mode == 'docker':
            container_spec_path = cls.spec_path.replace(cls.chain_path, '/subspace')
            container_snap_path = cls.snapshot_path.replace(cls.chain_path, '/subspace')
            volumes = f'-v {cls.spec_path}:{container_spec_path}'\
                         + f' -v {cls.snapshot_path}:{container_snap_path}'
            cmd = f'bash -c "docker run {volumes} {cls.image} {cmd} > {chain_spec_path}"'
            value = c.cmd(cmd, verbose=True)
        elif mode == 'local':
            cmd = f'bash -c "{cmd} > {chain_spec_path}"'
            c.print(cmd)
            c.cmd(cmd, cwd=cls.chain_path, verbose=True)  
            
              
        if vali_node_keys == None:
            vali_node_keys = cls.vali_node_keys(chain=chain)

        vali_nodes = list(vali_node_keys.keys())[:valis]
        vali_node_keys = {k:vali_node_keys[k] for k in vali_nodes}
        spec = c.get_json(chain_spec_path)
        spec['genesis']['runtime']['aura']['authorities'] = [k['aura'] for k in vali_node_keys.values()]
        spec['genesis']['runtime']['grandpa']['authorities'] = [[k['gran'],1] for k in vali_node_keys.values()]
        c.put_json(chain_spec_path, spec)

        if return_spec:
            return spec
        else:
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

    spec = get_spec

    @classmethod
    def spec_exists(cls, chain):
        return c.exists(f'{cls.spec_path}/{chain}.json')

    @classmethod
    def save_spec(cls, spec, chain:str=chain):
        chain = cls.chain_spec_path(chain)
        return c.put_json(chain, spec)

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
        chain_info = cls.chain_info(chain=chain)
        return [node_info['node'] for node_info in chain_info['nodes'].values() if node_info['validator'] == False]

    @classmethod
    def valis(cls, chain=chain):
        chain_info = cls.chain_info(chain=chain)
        return [node_info['node'] for node_info in chain_info['nodes'].values() if node_info['validator'] == True]

    @classmethod
    def num_valis(cls, chain=chain):
        return len(cls.vali_nodes(chain=chain))


    @classmethod
    def node_prefix(cls, chain=chain):
        return f'{cls.module_path()}.node'
    


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
    def push_image(cls, image='vivonasg/subspace'):
        c.build_image('subspace')
        c.cmd(f'docker tag subspace {image}', verbose=True)
        c.cmd(f'docker push {image}', verbose=True)
        return {'success':True, 'msg': f'pushed image {image}'}


    @classmethod
    def pull(cls, rpull:bool = False):
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

    @classmethod
    def rpull(cls):
        # pull from the remote server
        c.rcmd('c s pull', verbose=True)
        c.rcmd('c s pull_image', verbose=True)



    @classmethod
    def status(cls):
        return c.status(cwd=cls.libpath)

    @classmethod
    def start_local_node(cls, node:str='alice', mode=mode, chain=chain, max_boot_nodes:int=24, **kwargs):
        cls.pull_image()
        cls.add_node_key(node=node, chain=chain, mode=mode)
        response = cls.start_node(node=node, chain=chain, mode=mode, local=True, max_boot_nodes=max_boot_nodes, **kwargs)
        node_info = response['node_info']
        cls.put(f'local_nodes/{chain}/{node}', node_info)

        return response



    @classmethod
    def check_public_nodes(cls):
        config = cls.config()


    @classmethod
    def start_public_nodes(cls, node:str='nonvali', 
                           n:int=4,
                            i=0,
                            mode=mode, 
                           chain=chain, 
                           max_boot_nodes=24, 
                           refresh:bool = False,
                           remote:bool = False,
                           **kwargs):
        avoid_ports = []
        node_infos = cls.node_infos(chain= chain)
        served_nodes = []
        remote_addresses = []
        if remote:
            remote_addresses = c.module('remote').addresses()

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

            response = cls.start_node(node=node_name , chain=chain, mode=mode, **kwargs)
            if 'node_info' not in response:
                c.print(response, 'response')
                raise ValueError('No node info in response')

            node_info = response['node_info']
            c.print('started node', node_name, '--> ', response['logs'])
            served_nodes += [node_name]
            
            cls.putc(f'chain_info.{chain}.nodes.{node_name}', node_info)

    @classmethod
    def boot_nodes(cls, chain=chain):
        return cls.getc(f'chain_info.{chain}.boot_nodes', [])

     
    @classmethod
    def local_nodes(cls, chain=chain):
        return [p for p in cls.ls(f'local_nodes/{chain}')]

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
    def resolve_node_url(cls, url = None, chain=chain, local:bool = False):
        node2url = cls.node2url(network=chain)
        if url != None:
            if url in node2url: 
                url = node2url[url] 
            return url
        else:
            if local:
                local_node_paths = cls.local_nodes(chain=chain)
                local_node_info = cls.get(c.choice(local_node_paths))
                if local_node_info == None:
                    url = c.choice(list(node2url.values()))
                else:
                    port = local_node_info['ws_port']
                    url = f'ws://0.0.0.0:{port}'
            else:
                url = c.choice(cls.urls(network=chain))

            if not url.startswith('ws://'):
                url = 'ws://' + url

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
    def start_node(cls,
                 node : str,
                 chain:int = network,
                 port:int=None,
                 rpc_port:int=None,
                 ws_port:int=None,
                 telemetry_url:str = False,
                 purge_chain:bool = True,
                 refresh:bool = False,
                 verbose:bool = False,
                 boot_nodes = None,
                 node_key = None,
                 mode :str = mode,
                 rpc_cors = 'all',
                 pruning:str = 20000,
                 sync:str = 'warp',
                 validator:bool = False,
                 local:bool = False,
                 max_boot_nodes:int = 24,
                 daemon : bool = True,
                 key_mems:dict = None, # pass the keys mems {aura: '...', gran: '...'}
                 module : str = None , # remote module to call
                 ):

        if module != None:
            remote_kwargs = c.locals2kwargs(locals())
            namespace = c.namespace(network='remote') # default to remote namespace
            c.print(f'calling remote module {module}')
            if module in namespace:
                module = namespace[module]
            remote_kwargs.pop('module')
            c.print(f'calling remote node {module} with kwargs {remote_kwargs}')
            response =  c.call(module,  fn='submit', kwargs={'fn': 'subspace.start_node', 'kwargs': remote_kwargs}, timeout=100)[0]
            c.print(response, 'repsonse')
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
            c.print(f'adding node key for {key_mems}')
            cls.add_node_key(node=node,chain=chain, key_mems=key_mems, refresh=True, insert_key=True)

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
            cmd_kwargs += f' --telemetry-url {telemetry_url}'

        if validator :
            cmd_kwargs += ' --validator'
        else:
            cmd_kwargs += ' --ws-external --rpc-external'
            cmd_kwargs += f" --pruning={pruning}"
            cmd_kwargs += f" --sync {sync}"
            cmd_kwargs += f' --rpc-cors={rpc_cors}'

        cmd_kwargs += f' --port {port} --rpc-port {rpc_port} --ws-port {ws_port}'
        if boot_nodes == None:
            boot_nodes = cls.boot_nodes(chain=chain)
        # add the node to the boot nodes
        if len(boot_nodes) > 0:
            node_info['boot_nodes'] = c.choice(boot_nodes)  # choose a random boot node (at we chose one)
            cmd_kwargs += f" --bootnodes {node_info['boot_nodes']}"
    
        if node_key != None:
            cmd_kwargs += f' --node-key {node_key}'
            
 

        name = f'{cls.node_prefix()}.{chain}.{node}'


        if mode == 'local':
            # 
            cmd = c.pm2_start(path=cls.chain_release_path(mode=mode), 
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
            c.print(cmd, color='yellow')
            cmd = 'docker run ' + daemon_str  + f' --net host --name {name} {volumes} {cls.image}  bash -c "{cmd}"'
            node_info['cmd'] = cmd
            output = c.cmd(cmd, verbose=False)
            logs_sig = ' is already in use by container "'
            if logs_sig in output:
                container_id = output.split(logs_sig)[-1].split('"')[0]
                c.module('docker').rm(container_id)
                output = c.cmd(cmd, verbose=False)
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
    def start_chain(cls, 
                    chain:str=chain, 
                    valis:int = 8,
                    nonvalis:int = 1,
                    verbose:bool = False,
                    purge_chain:bool = True,
                    refresh: bool = True,
                    port_keys: list = ['port', 'rpc_port', 'ws_port'],
                    remote:bool = False,
                    build_spec :bool = True,
                    push:bool = True,
                    trials:int = 4
                    ):

        # KILL THE CHAIN
        if refresh:
            c.print(f'KILLING THE CHAIN ({chain})', color='red')
            cls.kill_chain(chain=chain)
            
        ## VALIDATOR NODES
        vali_node_keys  = cls.vali_node_keys(chain=chain)
        if len(vali_node_keys) <= valis:
            cls.add_node_keys(chain=chain, valis=valis, refresh=False)
            vali_node_keys  = cls.vali_node_keys(chain=chain)
        vali_nodes = list(vali_node_keys.keys())[:valis]
        vali_node_keys = {k: vali_node_keys[k] for k in vali_nodes}

        
        # BUILD THE CHAIN SPEC AFTER SELECTING THE VALIDATOR NODES'
        if build_spec:
            cls.build_spec(chain=chain, vali_node_keys=vali_node_keys, valis=valis)
            if push:
                cls.push(rpull=remote)

        chain_info = {'nodes':{}, 'boot_nodes':[]}
        

        remote_address_cnt = 1
        avoid_ports = []

        if remote: 
            remote_addresses = c.addresses('module', network='remote')

        # START THE VALIDATOR NODES
        for i, node in enumerate(vali_nodes):
            c.print(f'[bold]Starting node {node} for chain {chain}[/bold]')
            name = f'{cls.node_prefix()}.{chain}.{node}'

            # BUILD THE KWARGS TO CREATE A NODE
            
            node_kwargs = {
                            'chain':chain, 
                            'node':node, 
                            'verbose':verbose,
                            'purge_chain': purge_chain,
                            'validator':  True,

                            }


            success = False

            for t in range(trials):
                try:
                    if remote:  
                        remote_address_cnt += 1
                        node_kwargs['module'] = remote_addresses[remote_address_cnt % len(remote_addresses)]
                        node_kwargs['boot_nodes'] = chain_info['boot_nodes']

                    else:
                        node_ports = c.free_ports(n=3, avoid_ports=avoid_ports)
                        for k, port in zip(port_keys, node_ports):
                            avoid_ports.append(port)
                            node_kwargs[k] = port


                    node_kwargs['key_mems'] = cls.node_key_mems(node, chain=chain)

                    response = {}
                    response = cls.start_node(**node_kwargs, refresh=refresh)
                    assert 'boot_node' in response, f'boot_node must be in response, not {response.keys()}'

                    node_info = response['node_info']
                    boot_node = response['boot_node']
                    chain_info['boot_nodes'].append(boot_node)
                    chain_info['nodes'][node] = node_info
                    success = True 
                    break
                except Exception as e:
                    c.print(c.detailed_error(e))
                    c.print(f'failed to start node {node} for chain {chain}, trying again -> {t}/{trials} trials')

            assert success, f'failed to start node {node} for chain {chain} after {trials} trials'

                
        cls.putc(f'chain_info.{chain}', chain_info)

        if nonvalis > 0:
            # START THE NON VALIDATOR NODES
            cls.start_public_nodes(n=nonvalis, chain=chain, refresh=True, remote=remote)

        return {'success':True, 'msg': f'Started chain {chain}', 'valis':valis, 'nonvalis':nonvalis}
   
    @classmethod
    def node2url(cls, network:str = network) -> str:
        assert isinstance(network, str), f'network must be a string, not {type(network)}'
        nodes =  cls.getc(f'chain_info.{network}.nodes', {})
        nodes = {k:v for k,v in nodes.items() if v['validator'] == False}
        
        assert len(nodes) > 0, f'No url found for {network}'

        node2url = {}
        for k_n, v_n in nodes.items():
            node2url[k_n] = v_n['ip'] + ':' + str(v_n['ws_port'])
        return node2url

    @classmethod
    def urls(cls, network: str = network) -> str:
        return list(cls.node2url(network=network).values())

    def random_urls(self, network: str = network, n=4) -> str:
        urls = self.urls(network=network)
        return c.sample(urls, n=1)


    @classmethod
    def test_node_urls(cls, network: str = network) -> str:
        nodes = cls.nonvali_nodes()
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
                del config['chain_info'][network]['nodes'][node]

        cls.save_config(config)



    def storage_functions(self, network=network, block_hash = None):
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
    storage_fns = storage_functions
        

    def storage_names(self, network=network, block_hash = None):
        self.resolve_network(network)
        return [f['storage_name'] for f in self.substrate.get_metadata_storage_functions( block_hash=block_hash)]


    def stake_spread_top_valis(self):
        top_valis = self.top_valis()
        name2key = self.name2key()
        for vali in top_valis:
            key = name2key[vali]

    def ensure_stake(self, min_balance:int = 100, network:str='main', netuid:int=0):
        my_balance = self.my_balance(network=network, netuid=netuid)
        return my_balance



    def stake_spread(self,  modules:list=None, key:str = None,ratio = 1.0, n:int=20):
        key = self.resolve_key(key)
        name2key = self.name2key()
        if modules == None:
            modules = self.top_valis(n=n)
        if isinstance(modules, str):
            modules = [k for k,v in name2key.items() if modules in k]

        modules = modules[:n]
        modules = c.shuffle(modules)

        name2key = {k:name2key[k] for k in modules if k in name2key}


        module_names = list(name2key.keys())
        module_keys = list(name2key.values())
        n = len(name2key)

        # get the balance, of the key
        balance = self.get_balance(key)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        assert ratio <= 1.0, f'ratio must be less than or equal to 1.0, not {ratio}'
        assert ratio > 0.0, f'ratio must be greater than or equal to 0.0, not {ratio}'

        balance = int(balance * ratio)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        stake_per_module = int(balance/n)


        c.print(f'staking {stake_per_module} per module for ({module_names}) modules')

        s = c.module('subspace')()

        s.multistake(key=key, modules=module_keys, amounts=stake_per_module)

    

    @classmethod
    def unstake_many(cls, modules:list ='vali', key=None, remove_staketo:bool = False):
        if isinstance(modules, str):
            modules = c.my_modules(modules, fmt='j')
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        module_names = [m['name'] for m in modules]
        c.print(f'staking {stake_per_module} per module for ({module_names}) modules')
        for m in modules:
            for module_key, module_stake in m['stake_to']:
                # if the module_key is the same as the module we are unstaking, then unstake it
                if remove_staketo or m['key'] ==  module_key:
                    c.unstake(key=m['key'], module_key=module_key)
        

    @classmethod
    def test(cls):
        s = c.module('subspace')()
        n = s.n()
        assert isinstance(n, int)
        assert n > 0

        market_cap = s.mcap()
        assert isinstance(market_cap, float), market_cap

        name2key = s.name2key()
        assert isinstance(name2key, dict)
        assert len(name2key) == n

        stats = s.stats(df=False)
        c.print(stats)
        assert isinstance(stats, list) 

    telemetry_backend_image = 'parity/substrate-telemetry-backend'
    telemetry_frontend_image = 'parity/substrate-telemetry-frontend'
    @classmethod
    def install_telemetry(cls):
        c.cmd(f'docker build -t {cls.telemetry_image} .', sudo=False, bash=True)




    @classmethod
    def start_telemetry(cls, 
                    port:int=None, 
                    network:str='host', 
                    name='telemetry', 
                    chain=chain, 
                    trials:int=3, 
                    reuse_ports:bool=True,
                    frontend:bool = True):

        names = {'core': name, 'shard': f'{name}.shard', 'frontend': f'{name}.frontend'}
        docker = c.module('docker')
        config = cls.config()
        success = False
        cmd = {}
        output = {}
        k = f'chain_info.{chain}.telemetry_urls'
        telemetry_urls = cls.get(k, {})

        while trials > 0 and success == False:
            ports = {}
            ports['core'], ports['shard'], ports['frontend'] = c.free_ports(n=3, random_selection=True)
            if reuse_ports:
                telemetry_urls = cls.getc(k, {})
            if len(telemetry_urls) == 0:
                telemetry_urls[name] = {'shard': f"ws://{c.ip()}:{ports['shard']}/submit 0", 
                                        'feed': f"ws://{c.ip()}:{ports['core']}/feed", 
                                        'frontend': f'http://{c.ip()}:{ports["frontend"]}'}
                reuse_ports = False

            


            if reuse_ports:
                ports = {k:int(v.split(':')[-1].split('/')[0]) for k, v in telemetry_urls.items()}
            cmd['core'] = f"docker run  -d --network={network} --name {names['core']} \
                        --read-only \
                        {cls.telemetry_backend_image} \
                        telemetry_core -l 0.0.0.0:{ports['core']}"

            cmd['shard'] = f"docker run  -d --network={network} \
                        --name {names['shard']} \
                        --read-only \
                        {cls.telemetry_backend_image} \
                        telemetry_shard -l 0.0.0.0:{ports['shard']} -c http://0.0.0.0:{ports['core']}/shard_submit"

            cmd['frontend'] = f"docker run -d  \
                    --name {names['frontend']} \
                    -p {ports['frontend']}:8000\
                    -e SUBSTRATE_TELEMETRY_URL={telemetry_urls[name]['feed']} \
                    {cls.telemetry_frontend_image}"

            for k in cmd.keys():
                if docker.exists(names[k]):
                    docker.kill(names[k])
                output[k] = c.cmd(cmd[k])
                logs_sig = ' is already in use by container "'
                if logs_sig in output[k]:
                    container_id = output[k].split(logs_sig)[-1].split('"')[0]
                    docker.rm(container_id)
                    output[k] = c.cmd(cmd[k], verbose=True)
        
            success = bool('error' not in output['core'].lower()) and bool('error' not in output['shard'].lower())
            trials -= 1
            if success: 
                cls.putc(k, telemetry_urls)
        return {
            'success': success,
            'cmd': cmd,
            'output': output,
        }

    @classmethod
    def telemetry_urls(cls, name = 'telemetry', chain=chain):
        telemetry_urls = cls.getc(f'chain_info.{chain}.telemetry_urls', {})
        assert len(telemetry_urls) > 0, f'No telemetry urls found for {chain}, c start_telemetry'
        return telemetry_urls[name] 


    @classmethod
    def telemetry_url(cls,endpoint:str='submit', chain=chain, ):


        telemetry_urls = cls.telemetry_urls(chain=chain)
        if telemetry_urls == None:
            raise Exception(f'No telemetry urls found for {chain}')
        url = telemetry_urls[endpoint]

        if not url.startswith('ws://'):
            url = 'ws://' + url
        url = url.replace(c.ip(), '0.0.0.0')
        return url

    @classmethod
    def stop_telemetry(cls, name='telemetry'):
        return c.module('docker').kill(name)


    def telemetry_running(self):
        return c.module('docker').exists('telemetry')


    def check_storage(self, block_hash = None, network=network):
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)

    @classmethod
    def sand(cls): 
        node_keys =  cls.node_keys()
        spec = cls.spec()
        addy = c.root_key().ss58_address

        for i, (k, v) in enumerate(cls.datetime2archive('2023-10-17').items()):
            if i % 10 != 0:
                c.print(i, '/', len(v))
                continue
            state = c.get(v)
            c.print(state.keys())
            c.print(k, state['balances'].get(addy, 0))
        



    def test_balance(self, network:str = network, n:int = 10, timeout:int = 10, verbose:bool = False, min_amount = 10, key=None):
        key = c.get_key(key)

        balance = self.get_balance(network=network)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        balance = int(balance * 0.5)
        c.print(f'testing network {network} with {n} transfers of {balance} each')


    def test_commands(self, network:str = network, n:int = 10, timeout:int = 10, verbose:bool = False, min_amount = 10, key=None):
        key = c.get_key(key)

        key2 = c.get_key('test2')
        
        balance = self.get_balance(network=network)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        c.transfer(dest=key, amount=balance, timeout=timeout, verbose=verbose)
        balance = int(balance * 0.5)
        c.print(f'testing network {network} with {n} transfers of {balance} each')



        


        

