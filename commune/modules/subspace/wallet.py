import commune as c
from typing import *
from .subspace import Subspace

class Wallet(Subspace):
    subspace_config = Subspace.config()
    chain_path = f'{c.home}/.local/share/commune/chains'
    snapshot_path = f'{chain_path}/snapshots'
    network = subspace_config['network']
    netuid = 0
    fmt = 'j'

    def __init__(self, **kwargs):
        config = self.subspace_config
        config = self.set_config(config=config, kwargs=kwargs)
        c.print(config)

        
    def register(
        self,
        name: str , # defaults to module.tage
        address : str = 'NA',
        stake : float = 0,
        subnet: str = 'commune',
        key : str  = None,
        module_key : str = None,
        network: str = network,
        update_if_registered = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        existential_balance = 0.1,
        nonce=None,
        fmt = 'nano',


    ) -> bool:
        
        assert name != None, f"Module name must be provided"

        # resolve the subnet name
        if subnet == None:
            subnet = self.config.subnet

        network =self.resolve_network(network)

        if address:
            address = c.namespace(network='local').get(name, name)
            address = address.replace(c.default_ip,c.ip())
        
        if module_key == None:
            info = c.connect(address).info(timeout=5)
            module_key = info['ss58_address']


        key = self.resolve_key(key)

        # Validate address.
        netuid = self.get_netuid_for_subnet(subnet)
        min_stake = self.min_stake(netuid=netuid, registration=True)


        # convert to nanos
        min_stake = min_stake + existential_balance

        if stake == None:
            stake = min_stake 
        if stake < min_stake:
            stake = min_stake

        stake = self.to_nanos(stake)

        params = { 
                    'network': subnet.encode('utf-8'),
                    'address': address.encode('utf-8'),
                    'name': name.encode('utf-8'),
                    'stake': stake,
                    'module_key': module_key,
                } 
        # create extrinsic call
        response = self.compose_call('register', params=params, key=key, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization, nonce=nonce)

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
        nonce= None,
        
    ) -> bool:
        
        key = self.resolve_key(key)
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
            key=key,
            nonce = nonce
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

    ##################
    #### Transfer ####
    ##################
    def add_profit_shares(
        self,
        keys: List[str], 
        shares: List[float] = None , 
        key: str = None,
        network : str = None,
    ) -> bool:
        
        key = self.resolve_key(key)
        network = self.resolve_network(network)
        assert len(keys) > 0, f"Must provide at least one key"
        assert all([c.valid_ss58_address(k) for k in keys]), f"All keys must be valid ss58 addresses"
        if shares == None:
            shares = [1 for _ in keys]
        
        assert len(keys) == len(shares), f"Length of keys {len(keys)} must be equal to length of shares {len(shares)}"

        response = self.compose_call(
            module='SubspaceModule',
            fn='add_profit_shares',
            params={
                'keys': keys, 
                'shares': shares
            },
            key=key
        )

        return response




    def switch_module(self, module:str, new_module:str, n=10, timeout=20):
        stats = c.stats(module, df=False)

        namespace = c.namespace(new_module, public=True)
        servers = list(namespace.keys())[:n]
        stats = stats[:len(servers)]


        kwargs_list = []

        for m in stats:
            if module in m['name']:
                if len(servers)> 0: 
                    server = servers.pop()
                    server_address = namespace.get(server)
                    kwargs_list += [{'module': m['name'], 'name': server, 'address': server_address}]

        results = c.wait([c.submit(c.update_module, kwargs=kwargs, timeout=timeout, return_future=True) for kwargs in kwargs_list])
        
        return results
                


    def update_module(
        self,
        module: str, # the module you want to change
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
        c.print(module_info,  module)
        if module_info['key'] == None:
            return {'success': False, 'msg': 'not registered'}
        
        c.print(module_info)

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

        for k in ['delegation_fee']:
            if params[k] == None:
                params[k] = module_info[k]

        # check delegation_bounds
        assert params[k] != None, f"Delegate fee must be provided"
        delegation_fee = params['delegation_fee']
        if delegation_fee < 1.0 and delegation_fee > 0:
            delegation_fee = delegation_fee * 100
        assert delegation_fee >= 0 and delegation_fee <= 100, f"Delegate fee must be between 0 and 100"



        reponse  = self.compose_call('update_module',params=params, key=key)

        return reponse



    #################
    #### UPDATE SUBNET ####
    #################
    def update_subnet(
        self,
        netuid: int = None,
        key: str = None,
        network = network,
        nonce = None,
        **params,


    ) -> bool:
            
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        subnet_params = self.subnet_params( netuid=netuid , update=True, network=network )
        # infer the key if you have it
        if key == None:
            key2address = self.address2key()
            if subnet_params['founder'] not in key2address:
                return {'success': False, 'message': f"Subnet {netuid} not found in local namespace, please deploy it "}
            key = c.get_key(key2address.get(subnet_params['founder']))
            c.print(f'Using key: {key}')

        # remove the params that are the same as the module info
        params = {**subnet_params, **params}
        for k in ['name', 'vote_mode']:
            params[k] = params[k].encode('utf-8')
        params['netuid'] = netuid

        response = self.compose_call(fn='update_subnet',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


        return response


    #################
    #### Serving ####
    #################
    def propose_subnet_update(
        self,
        netuid: int = None,
        key: str = None,
        network = 'main',
        nonce = None,
        **params,


    ) -> bool:

        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        c.print(f'Adding proposal to subnet {netuid}')
        subnet_params = self.subnet_params( netuid=netuid , update=True)
        # remove the params that are the same as the module info
        params = {**subnet_params, **params}
        for k in ['name', 'vote_mode']:
            params[k] = params[k].encode('utf-8')
        params['netuid'] = netuid

        response = self.compose_call(fn='add_subnet_proposal',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


        return response



    #################
    #### Serving ####
    #################
    def vote_proposal(
        self,
        proposal_id: int = None,
        key: str = None,
        network = 'main',
        nonce = None,
        **params,

    ) -> bool:

        self.resolve_network(network)
        # remove the params that are the same as the module info
        params = {
            'proposal_id': proposal_id,
            'netuid': netuid,
        }

        response = self.compose_call(fn='add_subnet_proposal',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


        return response



    #################
    #### Serving ####
    #################
    def update_global(
        self,
        netuid: int = None,
        max_name_length: int = None,
        max_allowed_subnets : int = None,
        max_allowed_modules: int = None,
        max_registrations_per_block : int = None,
        unit_emission : int =None ,
        tx_rate_limit: int = None,
        key: str = None,
        network = network,
    ) -> bool:

        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        global_params = self.global_params( netuid=netuid )
        key = self.resolve_key(key)

        params = {
            'max_name_length': max_name_length,
            'max_allowed_subnets': max_allowed_subnets,
            'max_allowed_modules': max_allowed_modules,
            'max_registrations_per_block': max_registrations_per_block,
            'unit_emission': unit_emission,
            'tx_rate_limit': tx_rate_limit
        }

        # remove the params that are the same as the module info
        for k, v in params.items():
            if v == None:
                params[k] = global_params[k]
                
        # this is a sudo call
        response = self.compose_call(fn='update_global',params=params, key=key, sudo=True)

        return response





    #################
    #### set_code ####
    #################
    def set_code(
        self,
        wasm_file_path = None,
        key: str = None,
        network = network,
    ) -> bool:

        if wasm_file_path == None:
            wasm_file_path = self.wasm_file_path()

        assert os.path.exists(wasm_file_path), f'Wasm file not found at {wasm_file_path}'

        self.resolve_network(network)
        key = self.resolve_key(key)

        # Replace with the path to your compiled WASM file       
        with open(wasm_file_path, 'rb') as file:
            wasm_binary = file.read()
            wasm_hex = wasm_binary.hex()

        code = '0x' + wasm_hex

        # Construct the extrinsic
        response = self.compose_call(
            module='System',
            fn='set_code',
            params={
                'code': code.encode('utf-8')
            },
            unchecked_weight=True,
            sudo = True,
            key=key
        )

        return response

    
    def transfer_stake(
            self,
            new_module_key: str ,
            module_key: str ,
            amount: Union['Balance', float] = None, 
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
            amount: Union['Balance', float] = None, 
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
        else:
            module_key = module


        # Flag to indicate if we are using the wallet's own hotkey.
        old_balance = self.get_balance( key.ss58_address , fmt='j')
        old_stake = self.get_stakefrom( module, from_key=key.ss58_address , fmt='j', netuid=netuid, update=True)
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

        new_stake = self.get_stakefrom( module_key, from_key=key.ss58_address , fmt='j', netuid=netuid, update=True)
        new_balance = self.get_balance(  key.ss58_address , fmt='j', update=True)
        response.update({"message": "Stake Sent", "from": key.ss58_address, "to": module_key, "amount": amount, "balance_before": old_balance, "balance_after": new_balance, "stake_before": old_stake, "stake_after": new_stake})

        return response



    def unstake(
            self,
            module : str = None, # defaults to most staked module
            amount: float =None, # defaults to all of the amount
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
        if isinstance(module, int):
            amount = module
            module = None
        network = self.resolve_network(network)
        key = c.get_key(key)
        netuid = self.resolve_netuid(netuid)
        old_balance = self.get_balance( key.ss58_address , fmt='j')       
        # get most stake from the module
        stake_to = self.get_stake_to(netuid=netuid, names = False, fmt='nano', key=key)


        module_key = None
        if module == None:
            # find the largest staked module
            max_stake = 0
            for k,v in stake_to.items():
                if v > max_stake:
                    max_stake = v
                    module_key = k            
        else:
            key2name = self.key2name(netuid=netuid)
            name2key = {key2name[k]:k for k,v in key2name.items()}
            if module in name2key:
                module_key = name2key[module]
            else:
                module_key = module
        
        # we expected to switch the module to the module key
        assert c.valid_ss58_address(module_key), f"Module key {module_key} is not a valid ss58 address"
        assert module_key in stake_to, f"Module {module_key} not found in SubNetwork {netuid}"
        stake = stake_to[module_key]
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
            
    
    

    def stake_many( self, 
                        modules:List[str],
                        amounts:Union[List[str], float, int] = None,
                        key: str = None, 
                        netuid:int = 0,
                        n:str = 100,
                        network: str = None) -> Optional['Balance']:
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        name2key = self.name2key(netuid=netuid)

        if isinstance(modules, str):
            modules = [m for m in name2key.keys() if modules in m]
        modules = modules[:n] # only stake to the first n modules
        # resolve module keys
        for i, module in enumerate(modules):
            if module in name2key:
                modules[i] = name2key[module]
        assert len(modules) > 0, f"No modules found with name {modules}"
        module_keys = modules


        if amounts == None:
            balance = self.get_balance(key=key, fmt='nanos')
            amounts = [balance // len(modules)] * len(modules)
            assert sum(amounts) <= balance, f'The total amount is {sum(amounts)} > {balance}'

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
                    


    def transfer_multiple( self, 
                        destinations:List[str],
                        amounts:Union[List[str], float, int],
                        key: str = None, 
                        netuid:int = 0,
                        n:str = 10,
                        local:bool = False,
                        network: str = None) -> Optional['Balance']:
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        balance = self.get_balance(key=key, fmt='j')

        # name2key = self.name2key(netuid=netuid)


        
        key2address = c.key2address()
        name2key = self.name2key(netuid=netuid)

        if isinstance(destinations, str):
            local_destinations = [k for k,v in key2address.items() if destinations in k]
            if len(destinations) > 0:
                destinations = local_destinations
            else:
                destinations = [_k for _n, _k in name2key.items() if destinations in _n]

        assert len(destinations) > 0, f"No modules found with name {destinations}"
        destinations = destinations[:n] # only stake to the first n modules
        # resolve module keys
        for i, destination in enumerate(destinations):
            if destination in name2key:
                destinations[i] = name2key[destination]
            if destination in key2address:
                destinations[i] = key2address[destination]

        if isinstance(amounts, (float, int)): 
            amounts = [amounts] * len(destinations)

        assert len(destinations) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."
        assert all([c.valid_ss58_address(d) for d in destinations]), f"Invalid destination address {destinations}"



        total_amount = sum(amounts)
        assert total_amount < balance, f'The total amount is {total_amount} > {balance}'


        # convert the amounts to their interger amount (1e9)
        for i, amount in enumerate(amounts):
            amounts[i] = self.to_nanos(amount)

        assert len(destinations) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."

        params = {
            "netuid": netuid,
            "destinations": destinations,
            "amounts": amounts
        }

        response = self.compose_call('transfer_multiple', params=params, key=key)

        return response

    transfer_many = transfer_multiple


    def unstake_many( self, 
                        modules:Union[List[str], str] = None,
                        amounts:Union[List[str], float, int] = None,
                        key: str = None, 
                        netuid:int = 0,
                        network: str = None) -> Optional['Balance']:
        
        network = self.resolve_network( network )
        key = self.resolve_key( key )

        if modules == None or modules == 'all':
            stake_to = self.get_staketo(key=key, netuid=netuid, names=False, update=True, fmt='nanos') # name to amount
            module_keys = [k for k in stake_to.keys()]
            # RESOLVE AMOUNTS
            if amounts == None:
                amounts = [stake_to[m] for m in module_keys]

        else:
            stake_to = self.get_staketo(key=key, netuid=netuid, names=False, update=True, fmt='j') # name to amount
            name2key = self.name2key(netuid=netuid, update=True)

            module_keys = []
            for i, module in enumerate(modules):
                if c.valid_ss58_address(module):
                    module_keys += [module]
                else:
                    assert module in name2key, f"Invalid module {module} not found in SubNetwork {netuid}"
                    module_keys += [name2key[module]]
                
            # RESOLVE AMOUNTS
            if amounts == None:
                amounts = [stake_to[m] for m in module_keys]

            if isinstance(amounts, (float, int)): 
                amounts = [amounts] * len(module_keys)

            for i, amount in enumerate(amounts):
                amounts[i] = self.to_nanos(amount) 

        assert len(module_keys) == len(amounts), f"Length of modules and amounts must be the same. Got {len(module_keys)} and {len(amounts)}."

        params = {
            "netuid": netuid,
            "module_keys": module_keys,
            "amounts": amounts
        }

        response = self.compose_call('remove_stake_multiple', params=params, key=key)

        return response
                    



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

    def my_key2uid(self, *args, mode='all' , **kwargs):
        key2uid = self.key2uid(*args, **kwargs)
        key2address = c.key2address()
        key_addresses = list(key2address.values())
        my_key2uid = { k: v for k,v in key2uid.items() if k in key_addresses}
        return my_key2uid

    def vote_pool(self, netuid=None, network=None):
        my_modules = self.my_modules(netuid=netuid, network=network, names_only=True)
        for m in my_modules:
            c.vote(m, netuid=netuid, network=network)
        return {'success': True, 'msg': f'Voted for all modules {my_modules}'}

    
    def self_votes(self, search=None, netuid: int = None, network: str = None, parity=False, n=20, normalize=False, key=None) -> int:
        modules = self.my_modules(search=search, netuid=netuid, network=network)
        uids = [module['uid'] for module in modules]
        weights = [1 for _ in uids]



        if parity:
            votes = self.parity_votes(modules=modules)
        else:
            votes =  {'uids': uids, 'weights': weights}

        if n != None:
            votes['uids'] = votes['uids'][:n]
            votes['weights'] = votes['weights'][:n]

        return votes
    

    def self_vote(self, search= None, netuid: int = None, network: str = None, parity=False, n=20, timeout=100, normalize=False, key=None) -> int:
        votes = self.self_votes(netuid=netuid, network=network, parity=parity, n=n, normalize=normalize, key=key)
        if key == None:
            key = self.rank_my_modules(n=1, k='stake')[0]['name']
        kwargs={**votes, 'key': key, 'netuid': netuid, 'network': network}        
        return self.vote(**kwargs)



    def self_vote_pool(self, netuid: int = None, network: str = None, parity=False, n=20, timeout=20, normalize=False, key=None) -> int:
        keys = [m['name'] for m in self.rank_my_modules(n=n, k='stake')[:n] ]
        results = []
        for key in keys:
            kwargs = {'key': key, 'netuid': netuid, 'network': network, 'parity': parity, 'n': n, 'normalize': normalize}
            result = self.self_vote(**kwargs)
            results += [result]
        return results
    
    
    def vote_parity_loop(self, netuid: int = None, network: str = None, n=20, timeout=20, normalize=False, key=None) -> int:
        kwargs = {'key': key, 'netuid': netuid, 'network': network, 'parity': True, 'n': n, 'normalize': normalize}
        return self.self_vote(**kwargs)
        

    def vote(
        self,
        uids: Union['torch.LongTensor', list] = None,
        weights: Union['torch.FloatTensor', list] = None,
        netuid: int = None,
        key: 'c.key' = None,
        network = None,
        update=False,
        n = 10,
    ) -> bool:
        import torch
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = self.resolve_key(key)
        
        subnet = self.subnet( netuid = netuid )
        min_allowed_weights = subnet['min_allowed_weights']
        max_allowed_weights = subnet['max_allowed_weights']

        # checking if the "uids" are passed as names -> strings
        if uids != None and all(isinstance(item, str) for item in uids):
            names2uid = self.names2uids(names=uids, netuid=netuid)
            for i, name in enumerate(uids):
                if name in names2uid:
                    uids[i] = names2uid[name]
                else:
                    c.print(f'Could not find {name} in network {netuid}')
                    return False

        if uids is None:
            uids = self.subspace.uids(netuid=netuid, network=network)
            # shuffle the uids
            uids = c.shuffle(uids)
            
        if weights is None:
            weights = [1 for _ in uids]

  
        if len(uids) < min_allowed_weights:
            while len(uids) < min_allowed_weights:
                uid = c.choice(list(range(subnet['n'])))
                if uid not in uids:
                    uids.append(uid)
                    weights.append(1)

        uid2weight = {uid: weight for uid, weight in zip(uids, weights)}

        uids = list(uid2weight.keys())
        weights = weights[:len(uids)]

        c.print(f'Voting for {len(uids)} uids in network {netuid} with {len(weights)} weights')

        
        if len(uids) == 0:
            return {'success': False, 'message': f'No uids found in network {netuid}'}
        
        assert len(uids) == len(weights), f"Length of uids {len(uids)} must be equal to length of weights {len(weights)}"




        uids = uids[:max_allowed_weights]
        weights = weights[:max_allowed_weights]


        # uids = [int(uid) for uid in uids]
        uid2weight = {uid: int(weight) for uid, weight in zip(uids, weights)}
        uids = list(uid2weight.keys())
        weights = list(uid2weight.values())

        # sort the uids and weights
        uids = torch.tensor(uids)
        weights = torch.tensor(weights)
        indices = torch.argsort(weights, descending=True)
        uids = uids[indices]
        weights = weights[indices]

        weights = weights / weights.sum()
        weights = weights * (2**16)
        weights = list(map(int, weights.tolist()))
        uids = list(map(int, uids.tolist()))

        params = {'uids': uids,
                  'weights': weights, 
                  'netuid': netuid}
        
        response = self.compose_call('set_weights',params = params , key=key)
            
        if response['success']:
            return {'success': True, 'weights': weights, 'uids': uids, 'message': 'Set weights'}
        
        return response

    set_weights = vote



    def register_servers(self, search=None, **kwargs):
        stakes = self.stakes()
        for m in c.servers(network='local'):
            try:
                key = c.get_key(m)
                if key.ss58_address in stakes:
                    self.update_module(module=m)
                else:
                    self.register(name=m)
            except Exception as e:
                c.print(e, color='red')
    reg_servers = register_servers
    def reged_servers(self, **kwargs):
        servers =  c.servers(network='local')




    def stats(self, 
              search = None,
              netuid=0,  
              network = network,
              df:bool=True, 
              update:bool = False , 
              local: bool = True,
              cols : list = ['name', 'registered', 'serving',  'emission', 'dividends', 'incentive','stake', 'trust', 'regblock', 'last_update'],
              sort_cols = ['registered', 'emission', 'stake'],
              fmt : str = 'j',
              include_total : bool = True,
              **kwargs
              ):

        ip = c.ip()
        modules = self.modules(netuid=netuid, update=update, fmt=fmt, network=network, **kwargs)
        stats = []

        local_key_addresses = list(c.key2address().values())
        for i, m in enumerate(modules):

            if m['key'] not in local_key_addresses :
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

        if len(stats) > 0:

            df_stats = df_stats[cols]

            if 'last_update' in cols:
                block = self.block
                df_stats['last_update'] = df_stats['last_update'].apply(lambda x: block - x)

            c.print(df_stats)
            if 'emission' in cols:
                epochs_per_day = self.epochs_per_day(netuid=netuid, network=network)
                df_stats['emission'] = df_stats['emission'] * epochs_per_day


            sort_cols = [c for c in sort_cols if c in df_stats.columns]  
            df_stats.sort_values(by=sort_cols, ascending=False, inplace=True)

            if search is not None:
                df_stats = df_stats[df_stats['name'].str.contains(search, case=True)]

        if not df:
            return df_stats.to_dict('records')
        else:
            return df_stats

    def my_uids(self):
        return list(self.my_key2uid().values())
    
    def my_names(self, *args, **kwargs):
        my_modules = self.my_modules(*args, **kwargs)
        return [m['name'] for m in my_modules]
 


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
    
    def most_stake_key(self, **kwargs):
        my_stake = self.my_stake( **kwargs)
        return  dict(sorted(my_stake.items(), key=lambda item: item[1]))

    
    def my_balances(self, search=None, min_value=1000, fmt='j', **kwargs):
        balances = self.balances(fmt=fmt, **kwargs)
        address2key = c.address2key(search)
        my_balances = {k:balances.get(k, 0) for k in address2key.keys()}

        # sort the balances
        my_balances = {k:my_balances[k] for k in sorted(my_balances.keys(), key=lambda x: my_balances[x], reverse=True)}
        if min_value != None:
            my_balances = {k:v for k,v in my_balances.items() if v >= min_value}
        return my_balances

    

    def launcher_key(self, search=None, min_value=1000, **kwargs):
        
        my_balances = self.my_balances(search=search, min_value=min_value, **kwargs)
        key_address =  c.choice(list(my_balances.keys()))
        key_name = c.address2key(key_address)
        return key_name
    
    def launcher_keys(self, search=None, min_value=1000, n=1000, **kwargs):
        my_balances = self.my_balances(search=search, min_value=min_value, **kwargs)
        key_addresses = list(my_balances.keys())[:n]
        address2key = c.address2key()
        return [address2key[k] for k in key_addresses]
    
    def my_total_balance(self, search=None, min_value=1000, **kwargs):
        my_balances = self.my_balances(search=search, min_value=min_value, **kwargs)
        return sum(my_balances.values())
    
    


    def stake_spread(self,  modules:list=None, key:str = None,ratio = 1.0, n:int=50):
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

        s.stake_many(key=key, modules=module_keys, amounts=stake_per_module)

       

    def my_stake(self, search=None, netuid = None, network = None, fmt=fmt,  decimals=2, block=None, update=False):
        mystaketo = self.my_staketo(netuid=netuid, network=network, fmt=fmt, decimals=decimals, block=block, update=update)
        key2stake = {}
        for key, staketo_tuples in mystaketo.items():
            stake = sum([s for a, s in staketo_tuples])
            key2stake[key] = c.round_decimals(stake, decimals=decimals)
        if search != None:
            key2stake = {k:v for k,v in key2stake.items() if search in k}
        return key2stake
    


    def stake_top_modules(self,search=None, netuid=netuid, **kwargs):
        top_module_keys = self.top_module_keys(k='dividends')
        self.stake_many(modules=top_module_keys, netuid=netuid, **kwargs)
    
    def rank_my_modules(self,search=None, k='stake', n=10, **kwargs):
        modules = self.my_modules(search=search, **kwargs)
        ranked_modules = self.rank_modules(modules=modules, search=search, k=k, n=n, **kwargs)
        return modules[:n]


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

    def my_staketo(self,search=None, netuid = None, network = None, fmt=fmt,  decimals=2, block=None, update=False):
        staketo = self.stake_to(netuid=netuid, network=network, block=block, update=update)
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


    def my_value(self, network = None,fmt=fmt, decimals=2):
        return self.my_total_stake(network=network) + self.my_total_balance(network=network)
    
    my_supply   = my_value

    def my_total_stake(self, network = None, netuid=None, fmt=fmt, decimals=2, update=False):
        return sum(self.my_stake(network=network, netuid=netuid, fmt=fmt, decimals=decimals, update=update).values())
    def my_total_balance(self, network = None, fmt=fmt, decimals=2, update=False):
        return sum(self.my_balance(network=network, fmt=fmt, decimals=decimals).values())


    def parity_votes(self, modules=None, netuid: int = 0, network: str = None, n=None) -> int:
        if modules == None:
            modules = self.modules(netuid=netuid, network=network)
        # sample inversely proportional to emission rate
        weights = [module['emission'] for module in modules]
        uids = [module['uid'] for module in modules]
        weights = torch.tensor(weights)
        max_weight = weights.max()
        weights = max_weight - weights
        weights = weights / weights.sum()
        # weights = weights * U16_MAX
        weights = weights.tolist()
        return {'uids': uids, 'weights': weights}



    def check_valis(self):
        return self.check_servers(search='vali', netuid=None, wait_for_server=False, update=False)
    
    def check_servers(self, search=None, wait_for_server=False, update:bool=False, key=None, network='local'):
        cols = ['name', 'registered', 'serving', 'address', 'last_update']
        module_stats = self.stats(search=search, netuid=0, cols=cols, df=False, update=update)
        module2stats = {m['name']:m for m in module_stats}
        subnet = self.subnet()
        namespace = c.namespace(search=search, network=network, update=True)

        for module, stats in module2stats.items():
            if not c.server_exists(module):
                c.serve(module)

        c.print('checking', list(namespace.keys()))
        for name, address in namespace.items():
            if name not in module2stats :
                # get the stats for this module
                self.register(name=name, address=address, key=key)
                continue
            
            m_stats = module2stats.get(name)
            if 'vali' in module: # if its a vali
                if stats['last_update'] > subnet['tempo']:
                    c.print(f"Vali {module} has not voted in {stats['last_update']} blocks. Restarting...")
                    c.restart(module)
                    
            else:
                if m_stats['serving']:
                    if address != m_stats['address']:
                        c.update_module(module=m_stats['name'], address=address, name=name)
                else:
                    
                    if ':' in m_stats['address']:
                        port = int(m_stats['address'].split(':')[-1])
                    else:
                        port = None
                        
                    c.serve(name, port=port, wait_for_server=wait_for_server)



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
                    save_history : bool = True,
                    sudo:bool  = False,
                    nonce: int = None,
                    remote_module: str = None,
                    unchecked_weight: bool = False,
                     **kwargs):

        """
        Composes a call to a Substrate chain.

        """
        key = self.resolve_key(key)

        if remote_module != None:
            kwargs = c.locals2kwargs(locals())
            return c.connect(remote_module).compose_call(**kwargs)

        params = {} if params == None else params
        if verbose:
            kwargs = c.locals2kwargs(locals())
            kwargs['verbose'] = False
            c.status(f":satellite: Calling [bold]{fn}[/bold] on [bold yellow]{self.network}[/bold yellow]")
            return self.compose_call(**kwargs)

        start_time = c.datetime()
        ss58_address = key.ss58_address


        pending_path = f'history/{ss58_address}/pending/{self.network}_{module}::{fn}::nonce_{nonce}.json'
        complete_path = f'history/{ss58_address}/complete/{start_time}_{self.network}_{module}::{fn}.json'

        # if self.exists(pending_path):
        #     nonce = self.get_nonce(key=key, network=self.network) + 1
            
        compose_kwargs = dict(
                call_module=module,
                call_function=fn,
                call_params=params,
        )

        c.print('compose_kwargs', compose_kwargs, color=color)
        tx_state = dict(status = 'pending',start_time=start_time, end_time=None)

        self.put_json(pending_path, tx_state)

        with self.substrate as substrate:

            call = substrate.compose_call(**compose_kwargs)

            if sudo:
                call = substrate.compose_call(
                    call_module='Sudo',
                    call_function='sudo',
                    call_params={
                        'call': call,
                    }
                )
            if unchecked_weight:
                # uncheck the weights for set_code
                call = substrate.compose_call(
                    call_module="Sudo",
                    call_function="sudo_unchecked_weight",
                    call_params={
                        "call": call,
                        'weight': (0,0)
                    },
                )
            # get nonce 
            extrinsic = substrate.create_signed_extrinsic(call=call,keypair=key,nonce=nonce)

            response = substrate.submit_extrinsic(extrinsic=extrinsic,
                                                  wait_for_inclusion=wait_for_inclusion, 
                                                  wait_for_finalization=wait_for_finalization)


        if wait_for_finalization:
            if process_events:
                response.process_events()

            if response.is_success:
                response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.network} with key {key.ss58_address}'}
            else:
                response =  {'success': False, 'error': response.error_message, 'msg': f'Failed to call {module}.{fn} on {self.network} with key {key.ss58_address}'}

            if save_history:
                self.add_history(response)
        else:
            response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.network} with key {key.ss58_address}'}
        
        
        tx_state['end_time'] = c.datetime()
        tx_state['status'] = 'completed'
        tx_state['response'] = response

        # remo 
        self.rm(pending_path)
        self.put_json(complete_path, tx_state)

        return response
            

    @classmethod
    def add_history(cls, response:dict) -> dict:
        return cls.put(cls.history_path + f'/{c.time()}',response)

    @classmethod
    def clear_history(cls):
        return cls.put(cls.history_path,[])

    def tx_history(self, key:str=None, mode='pending', **kwargs):
        pending_path = self.resolve_pending_dirpath(key=key, mode=mode, **kwargs)
        return self.ls(pending_path)
    
    def pending_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='pending', **kwargs)

    def complete_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='complete', **kwargs)

    def clean_tx_history(self):
        return self.ls(f'tx_history')

        

    def resolve_tx_dirpath(self, key:str=None, mode:'str([pending,complete])'='pending',  **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'tx_history/{key_ss58}/pending'
        return pending_path
    
    def resolve_tx_history_path(self, key:str=None, mode:str='pending', **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'tx_history/{key_ss58}/{mode}'
        return pending_path

    def has_tx_history(self, key:str, mode='pending', **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        return self.exists(f'tx_history/{key_ss58}')


    def resolve_key(self, key = None):
        if key == None:
            key = self.config.key
        if key == None:
            key = 'module'
        if isinstance(key, str):
            if c.key_exists( key ):
                key = c.get_key( key )
        assert hasattr(key, 'ss58_address'), f"Invalid Key {key} as it should have ss58_address attribute."
        return key
        
    @classmethod
    def test_endpoint(cls, url=None):
        if url == None:
            url = c.choice(cls.urls())
        self = cls()
        c.print('testing url -> ', url, color='yellow' )

        try:
            self.set_network(url=url, max_trials=1)
            success = isinstance(self.block, int)
        except Exception as e:
            c.print(c.detailed_error(e))
            success = False

        c.print(f'success {url}-> ', success, color='yellow' )
        
        return success


    def stake_spread_top_valis(self):
        top_valis = self.top_valis()
        name2key = self.name2key()
        for vali in top_valis:
            key = name2key[vali]
