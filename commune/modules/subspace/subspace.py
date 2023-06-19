
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
from commune.modules.subspace.chain_data import (ModuleInfo, 
                                         SubnetInfo, 
                                         custom_rpc_type_registry)
from commune.modules.subspace.errors import (ChainConnectionError,
                                     ChainTransactionError, 
                                     ChainQueryError, StakeError,
                                     UnstakeError, 
                                     TransferError,
                                     RegistrationError, 
                                     SubspaceError)
import streamlit as st
import json
from loguru import logger
logger = logger.opt(colors=True)



class Subspace(c.Module):
    """
    Handles interactions with the subspace chain.
    """
    default_config = c.get_config('subspace', to_munch=False)
    token_decimals = default_config['token_decimals']
    retry_params = default_config['retry_params']
    network2url = default_config['network2url']
    chain = default_config['chain']
    network = default_config['network']
    subnet = default_config['subnet']
    chain_path = eval(default_config['chain_path'])
    chain_release_path = eval(default_config['chain_release_path'])
    spec_path = eval(default_config['spec_path'])
    key_types = default_config['key_types']
    supported_schemas = default_config['supported_schemas']
    default_netuid = default_config['default_netuid']

    
    def __init__( 
        self, 
        network: str = network,
        **kwargs,
    ):


        self.set_subspace( network)
    @classmethod
    def get_network_url(cls, network:str = network) -> str:
        assert isinstance(network, str), f'network must be a string, not {type(network)}'
        return cls.network2url.get(network)
    @classmethod
    def url2network(cls, url:str) -> str:
        return {v: k for k, v in cls.network2url.items()}.get(url, None)
    
    @classmethod
    def resolve_network_url(cls, network:str ):  
        external_ip = cls.external_ip()      
        url = cls.get_network_url(network)

        # if not url.startswith('wss://') and not url.startswith('ws://'):
        if not url.startswith('ws://'):
            url = f'ws://{url}'
        
        return url
    def set_subspace(self, 
                network:str,
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
                *args, 
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
        

        url = self.resolve_network_url(network)
        
        self.url = self.chain_endpoint = url
        
        
        self.substrate= SubstrateInterface(
                                    url=url, 
                                    websocket=websocket, 
                                    ss58_format=ss58_format, 
                                    type_registry=type_registry, 
                                    type_registry_preset=type_registry_preset, 
                                    cache_region=cache_region, 
                                    runtime_config=runtime_config, 
                                    use_remote_preset=use_remote_preset,
                                    ws_options=ws_options, 
                                    auto_discover=auto_discover, 
                                    auto_reconnect=auto_reconnect, 
                                    *args,
                                    **kwargs)
        
      

    def __repr__(self) -> str:
        return self.__str__()


    #####################
    #### Set Weights ####
    #####################
    def set_weights(
        self,
        uids: Union[torch.LongTensor, list] ,
        weights: Union[torch.FloatTensor, list],
        network: int = None,
        key: 'c.key' = None,
        wait_for_inclusion:bool = True,
        wait_for_finalization:bool = True,
        prompt:bool = False,
    ) -> bool:
        netuid = self.get_netuid_for_network(network)
        # First convert types.
        if isinstance( uids, list ):
            uids = torch.tensor( uids, dtype = torch.int64 )
        if isinstance( weights, list ):
            weights = torch.tensor( weights, dtype = torch.float32 )

        # Reformat and normalize.
        weight_uids, weight_vals =  uids, weights/weights.sum() 

        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]?".format( [float(v/65535) for v in weight_vals], weight_uids) ):
                return False

        with c.status(":satellite: Setting weights on [white]{}[/white] ...".format(self.network)):
            try:
                with self.substrate as substrate:
                    call = substrate.compose_call(
                        call_module='SubspaceModule',
                        call_function='set_weights',
                        call_params = {
                            'dests': weight_uids,
                            'weights': weight_vals,
                            'netuid': netuid,
                            'version_key': 1
                        }
                    )
                    # Period dictates how long the extrinsic will stay as part of waiting pool
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key, era={'period':100})
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        c.print(":white_heavy_check_mark: [green]Sent[/green]")
                        return True

                    response.process_events()
                    if response.is_success:
                        c.print(":white_heavy_check_mark: [green]Finalized[/green]")
                        logger.print(  prefix = 'Set weights', sufix = '<green>Finalized: </green>' + str(response.is_success) )
                        return True
                    else:
                        c.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                        c.print(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(response.error_message) )
                        return False

            except Exception as e:
                c.print(":cross_mark: [red]Failed[/red]: error:{}".format(e))
                c.status(  'Set weights <red>Failed: </red>' + str(e) )
                return False

        if response.is_success:
            c.print("Set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]".format( [float(v/4294967295) for v in weight_vals], weight_uids ))
            message = '<green>Success: </green>' + f'Set {len(uids)} weights, top 5 weights' + str(list(zip(uids.tolist()[:5], [round (w,4) for w in weights.tolist()[:5]] )))
            c.debug('Set weights:'.ljust(20) +  message)
            return True
        
        return False


    @classmethod
    def get_key(cls, uri= None) -> 'c.Key':
        
        key = c.module('key')
        if uri != None:
            key = key.create_from_uri(uri)
        else:
            raise NotImplementedError('No uri, mnemonic, privatekey or publickey provided')
        return key
    def get_netuid_for_network(self, network: str = None) -> int:
        netuid = self.subnet_namespace[network]
        return netuid
    
    
    def register(
        self,
        key ,
        name: str ,
        stake : int = 0,
        address: str = None,
        network = subnet,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
        max_allowed_attempts: int = 3,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,

    ) -> bool:

        r""" Registers the wallet to chain.
        Args:
            network (int):
                The netuid of the subnet to register on.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
            max_allowed_attempts (int):
                Maximum number of attempts to register the wallet.
            update_interval (int):
                The number of nonces to solve between updates.
            log_verbose (bool):
                If true, the registration process will log more information.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        
        key = self.resolve_key(key)
        
        # netuid = self.subnet_namespace(network)
        if address is None:
            address = f'{c.ip()}:{c.free_port()}'
    
        # Attempt rolling registration.
        call_params = { 
                    'network': network.encode('utf-8'),
                    'address': address.encode('utf-8'),
                    'name': name.encode('utf-8'),
                    'stake': stake,
                } 
        c.print(f":satellite: Registering {key} \n Params : {call_params}")

        with self.substrate as substrate:
            
            # create extrinsic call
            call = substrate.compose_call( 
                call_module='SubspaceModule',  
                call_function='register', 
                call_params=call_params
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key  )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )
            
            # process if registration successful, try again if pow is still valid
            response.process_events()
            
            if response.is_success:
                c.print(":white_heavy_check_mark: [green]Success[/green]")
                return True
            else:
                c.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                return False
    
            


    ##################
    #### Transfer ####
    ##################
    def transfer(
        self,
        dest: str, 
        amount: float , 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
        key: 'c.Key' =  None,
        keep_alive: bool = True
    ) -> bool:
        key = c.get_key(key)


        # Validate destination address.
        if not is_valid_address_or_public_key( dest ):
            c.print(":cross_mark: [red]Invalid destination address[/red]:[bold white]\n  {}[/bold white]".format(dest))
            return False

        if isinstance( dest, bytes):
            # Convert bytes to hex string.
            dest = "0x" + dest.hex()


        # Check balance.
        with c.status(":satellite: Checking Balance..."):
            account_balance = self.get_balance( key.ss58_address )
            existential_deposit = self.get_existential_deposit()

        transfer_balance =  Balance.to_nanos(amount)
        with c.status(":satellite: Transferring..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Balances',
                    call_function='transfer',
                    call_params={
                        'dest': dest, 
                        'value': transfer_balance
                    }
                )

                try:
                    payment_info = substrate.get_payment_info( call = call, keypair = key )
                except Exception as e:
                    c.print(":cross_mark: [red]Failed to get payment info[/red]:[bold white]\n  {}[/bold white]".format(e))
                    payment_info = {
                        'partialFee': 2e7, # assume  0.02 joule 
                    }

                fee = transfer_balance.to_nanos(payment_info['partialFee'])
        
        if not keep_alive:
            # Check if the transfer should keep_alive the account
            existential_deposit = 0

        # Check if we have enough balance.
        if account_balance < (transfer_balance + fee + existential_deposit):
            c.print(":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance: {}\n  amount: {}\n  for fee: {}[/bold white]".format( account_balance, transfer_balance, fee ))
            return False


        with c.status(":satellite: Transferring to {}"):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Balances',
                    call_function='transfer',
                    call_params={
                        'dest': dest, 
                        'value': transfer_balance
                    }
                )

                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    c.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                # Otherwise continue with finalization.
                response.process_events()
                if response.is_success:
                    c.print(":white_heavy_check_mark: [green]Finalized[/green]")
                    block_hash = response.block_hash
                    c.print("[green]Block Hash: {}[/green]".format( block_hash ))
                    new_balance = self.get_balance( key.ss58_address )
                    c.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(account_balance, new_balance))
                    return True
                else:
                    c.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))


        
        return False
    
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
        
        
    
    def serve(self,
              module: str,
              name = None,
              tag = None,
              network = subnet, 
              address = None,
              stake = 0,
              key = None, **kwargs):
        
        
        key = self.resolve_key(key)
        name = c.resolve_server_name(module=module, name=name, tag=tag)

        
        c.serve(module=module, address=address, name=name, **kwargs)
        c.wait_for_server(name)
        address = c.namespace().get(name)
        address = address.replace(c.default_ip, c.ip())
        
        if self.is_registered(key):
            self.update_module(key=key, 
                               name=name, 
                               address=address,
                               netuid=self.get_netuid_for_network(network), 
                               **kwargs)
        else:
        
            self.register(
                name = name,
                address = address,
                key = key,
                network = network,
            )
        

    #################
    #### Serving ####
    #################
    def update_module (
        self,
        key: 'c.Key' ,
        name: str = None,
        address: str = None,
        netuid: int = None,
        wait_for_inclusion: bool = False,
        wait_for_finalization = True,
        prompt: bool = False,
    ) -> bool:

        key = self.resolve_key(key)
        netuid = self.resolve_netuid(netuid)
        module = self.get_module( key )
        
        if name is None:
            name = module['name']
        if address is None:
            address = module['address']
        
        with c.status(":satellite: Serving module on: [white]{}:{}[/white] ...".format(self.network, netuid)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubspaceModule',
                    call_function='update_module',
                    call_params= {
                                'address': address,
                                'name': name,
                                'netuid': netuid,
                            }
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key)
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        module = self.get_module( key )
                        c.print(f':white_heavy_check_mark: [green]Updated Module[/green]\n  [bold white]{module}[/bold white]')
                        return True
                    else:
                        c.print(f':cross_mark: [green]Failed to Serve module[/green] error: {response.error_message}')
                        return False
                else:
                    return True


    def add_stake(
            self,
            key: Optional[str] ,
            amount: Union[Balance, float] = None, 
            netuid:int = None,
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        
        key = c.get_key(key)
        netuid = self.resolve_netuid(netuid)
        
        # Flag to indicate if we are using the wallet's own hotkey.
        old_balance = self.get_balance( key.ss58_address , fmt='j')
        old_stake = self.get_stake( key.ss58_address , fmt='j')

        if amount is None:
            amount = old_balance
        
        # assume amount is in joules
        amount = self.to_nano(amount)

        # Get current stake

        try:
            with c.status(":satellite: Staking to: [bold white]{}[/bold white] ...".format(self.network)):

                with self.substrate as substrate:
                    call = substrate.compose_call(
                    call_module='SubspaceModule', 
                    call_function='add_stake',
                    call_params={
                        'netuid': netuid,
                        'amount_staked': amount
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
                    response = substrate.submit_extrinsic( extrinsic, 
                                                          wait_for_inclusion = wait_for_inclusion,
                                                          wait_for_finalization = wait_for_finalization )


            if response : # If we successfully staked.
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    c.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                c.print(":white_heavy_check_mark: [green]Finalized[/green]")
                with c.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                    new_balance = self.get_balance(  key.ss58_address , fmt='j')
                    block = self.get_current_block()
                    new_stake = self.get_stake(key.ss58_address,block=block, fmt='j') # Get current stake

                    c.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                    c.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                    return True
            else:
                c.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                return False
        except StakeError as e:
            c.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            return False





    def unstake (
            self,
            key: 'c.Key', 
            amount: float = None, 
            netuid = None,
            wait_for_inclusion:bool = True, 
            wait_for_finalization:bool = False,
            prompt: bool = False,
        ) -> bool:

        key = c.get_key(key)
        netuid = self.resolve_netuid(netuid)
        old_stake = self.get_stake( key.ss58_address, netuid=netuid, fmt='nano' )
        if amount == None:
            amount = old_stake
            
        old_balance = self.get_balance(  key.ss58_address , fmt='nano')
            
            
        c.print("Unstaking [bold white]{}[/bold white] from [bold white]{}[/bold white]".format(amount, self.network))
        
        try:
            with c.status(":satellite: Unstaking from chain: [white]{}[/white] ...".format(self.network)):


                with self.substrate as substrate:
                    call = substrate.compose_call(
                    call_module='SubspaceModule', 
                    call_function='remove_stake',
                    call_params={
                        'amount_unstaked': amount,
                        'netuid': netuid
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        return True

                    response.process_events()


            if response.is_success: # If we successfully unstaked.
                c.print(":white_heavy_check_mark: [green]Finalized[/green]")
                with c.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                    old_balance = self.to_token(old_balance)
                    old_stake = self.to_token(old_stake)
                    
                    new_balance = self.get_balance( key.ss58_address , fmt='token')
                    new_stake = self.get_stake( key.ss58_address , fmt='token') # Get stake on hotkey.
                    
                    c.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                    c.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                    return True
            else:
                c.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                return False

        # except Exception as e:
        #     c.print(f":cross_mark: [red]key: {key} is not registered.[/red]")
        #     return False
        except StakeError as e:
            c.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            return False

    ########################
    #### Standard Calls ####
    ########################

    """ Queries subspace named storage with params and block. """
    @retry(delay=2, tries=3, backoff=2, max_delay=4)
    def query_subspace( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[object]:
        with self.substrate as substrate:
            return substrate.query(
                module='SubspaceModule',
                storage_function = name,
                params = params,
                block_hash = None if block == None else substrate.get_block_hash(block)
                )

    """ Queries subspace map storage with params and block. """
    def query_map( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [default_netuid] ) -> Optional[object]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query_map(
                    module='SubspaceModule',
                    storage_function = name,
                    params = params,
                    block_hash = None if block == None else substrate.get_block_hash(block)
                )
        return make_substrate_call_with_retry()
    
    """ Gets a constant from subspace with module_name, constant_name, and block. """
    def query_constant( self, module_name: str, constant_name: str, block: Optional[int] = None ) -> Optional[object]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.get_constant(
                    module_name=module_name,
                    constant_name=constant_name,
                    block_hash = None if block == None else substrate.get_block_hash(block)
                )
        return make_substrate_call_with_retry()
      
    #####################################
    #### Hyper parameter calls. ####
    #####################################

    """ Returns network ImmunityPeriod hyper parameter """
    def immunity_period (self, netuid: int = None, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query_subspace("ImmunityPeriod", block, [netuid] ).value


    """ Returns network MinAllowedWeights hyper parameter """
    def min_allowed_weights (self, netuid: int = None, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query_subspace("MinAllowedWeights", block, [netuid] ).value

    """ Returns network MaxWeightsLimit hyper parameter """
    def max_weight_limit (self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return U16_NORMALIZED_FLOAT( self.query_subspace('MaxWeightsLimit', block, [netuid] ).value )

    """ Returns network SubnetN hyper parameter """
    def n (self, netuid: int = None, block: Optional[int] = None ) -> int:
        netuid = self.resolve_netuid( netuid )
        return self.query_subspace('N', block, [netuid] ).value

    """ Returns network MaxAllowedUids hyper parameter """
    def max_allowed_uids (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query_subspace('MaxAllowedUids', block, [netuid] ).value

    """ Returns network Tempo hyper parameter """
    def tempo (self, netuid: int = None, block: Optional[int] = None) -> int:
        netuid = self.resolve_netuid( netuid )
        return self.query_subspace('Tempo', block, [netuid] ).value

    ##########################
    #### Account functions ###
    ##########################
    
    """ Returns network Tempo hyper parameter """
    def stake (self, key = None, netuid: int = None, block: Optional[int] = None) -> int:
        netuid = self.resolve_netuid( netuid )
        return {k.value:v.value for k,v in self.query_map('Stake', block, [netuid] ).records}

    """ Returns the stake under a coldkey - hotkey pairing """
    
    
    @classmethod
    def resolve_key_ss58(cls, key_ss58):
        
        if isinstance(key_ss58, str):
            if c.key_exists( key_ss58 ):
                key_ss58 = c.get_key( key_ss58 ).ss58_address
        if hasattr(key_ss58, 'ss58_address'):
            key_ss58 = key_ss58.ss58_address
            
            
    
        return key_ss58

    @classmethod
    def resolve_key(cls, key):
        
        if isinstance(key, str):
            assert c.key_exists( key ), f"Key {key} does not exist."
            key = c.get_key( key )
        return key
        
    @classmethod
    def from_nano(cls,x):
        return x / (10**cls.token_decimals)
    to_token = from_nano
    @classmethod
    def to_nano(cls,x):
        return x * (10**cls.token_decimals)
    from_token = to_nano
    @classmethod
    def format_amount(cls, x, fmt='nano'):
        if fmt in ['nano', 'n']:
            return x
        elif fmt in ['token', 'unit', 'j', 'J']:
            return cls.to_token(x)
        else:
            raise ValueError(f"Invalid format {fmt}.")
    
    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j' ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58 )
        netuid = self.resolve_netuid( netuid )
        return self.format_amount(self.query_subspace( 'Stake', block, [netuid, key_ss58] ).value, fmt=fmt)
       
    ###########################
    #### Global Parameters ####
    ###########################

    @property
    def block (self) -> int:
        r""" Returns current chain block.
        Returns:
            block (int):
                Current chain block.
        """
        return self.get_current_block()

    def total_stake (self,block: Optional[int] = None ) -> 'Balance':
        return Balance.from_nano( self.query_subspace( "TotalStake", block ).value )

    def subnets(self):
        subnets = []
        subnet_stake = {k.value:v.value for k,v in self.query_map( 'SubnetTotalStake', params=[] ).records}
        subnet_emission = {k.value:v.value for k,v in self.query_map( 'SubnetEmission', params=[] ).records}
        subnet_founders = {k.value:v.value for k,v in self.query_map( 'Founder', params=[] ).records}
        n = {k.value:v.value for k,v in self.query_map( 'N', params=[] ).records}
        total_stake = self.total_stake()
        
        for name, netuid  in self.subnet_namespace.items():
            
            subnets.append(
                {
                    'name': name,
                    'netuid': netuid,
                    'stake': Balance.from_nano(subnet_stake[netuid]),
                    'emission': Balance.from_nano(subnet_emission[netuid]),
                    'n': n[netuid],
                    'tempo': self.tempo( netuid = netuid ),
                    'immunity_period': self.immunity_period( netuid = netuid ),
                    'min_allowed_weights': self.min_allowed_weights( netuid = netuid ),
                    'max_allowed_uids': self.max_allowed_uids( netuid = netuid ),
                    'ratio': Balance.from_nano(subnet_stake[netuid]) / total_stake,
                    'founder': subnet_founders[netuid]
                    
                }
            )
            
        return subnets
            
            
            
            

    def get_total_subnets( self, block: Optional[int] = None ) -> int:
        return self.query_subspace( 'TotalSubnets', block ).value      
    
    def get_emission_value_by_subnet( self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return Balance.from_nano( self.query_subspace( 'EmissionValues', block, [ netuid ] ).value )



    def is_registered( self, key: str, netuid: int = None, block: Optional[int] = None) -> bool:
        key_address = self.resolve_key( key ).ss58_address
        key_addresses = self.keys(netuid=netuid, block=block)
        if key_address in key_addresses:
            return True
        else:
            return False

    def get_uid_for_key_on_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> int:
        return self.query_subspace( 'Uids', block, [ netuid, key_ss58 ] ).value  


    def get_current_block(self) -> int:
        r""" Returns the current block number on the chain.
        Returns:
            block_number (int):
                Current chain blocknumber.
        """        
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.get_block_number(None)
        return make_substrate_call_with_retry()



    def get_balance(self, key: str, block: int = None, fmt='j') -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        
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

    def get_balances(self, block: int = None, fmt:str = 'n') -> Dict[str, Balance]:
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
        return return_dict
    balances = get_balances
    
    def resolve_network(self, network: Optional[int] = None) -> int:
        if network == None:
            network = self.network
        return network
    
    def resolve_subnet(self, subnet: Optional[int] = None) -> int:
        if subnet == None:
            subnet = self.subnet
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



    @property
    def subnet_namespace(self ) -> Dict[str, str]:
        
        # Get the namespace for the netuid.
        records = self.query_map('SubnetNamespace', params=[]).records
        
        subnet_namespace = {}
        for r in records:
            name = r[0].value
            uid = int(r[1].value)
            subnet_namespace[name] = int(uid)
        
        return subnet_namespace
    
        
    def subnet_netuids(self) -> List[int]:
        return list(self.subnet_namespace.values())

    @classmethod
    def resolve_netuid(cls, netuid: int = None) -> int:
        if netuid == None:
            netuid = cls.default_netuid
        return netuid


    def key2name(self, key: str = None, netuid: int = None) -> str:
        modules = self.modules(netuid)
        key2name =  { m['key']: m['name']for m in modules.values() }
        if key != None:
            return key2name[key]
            
        
    def name2key(self, name:str=None,  netuid: int = None) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        
        # Get the namespace for the netuid.
        netuid = self.resolve_netuid(netuid)        
        keys = { r[0].value: r[1].value for r in self.query_map('Keys', params=[netuid]).records}
        namespace = { r[0].value: keys[r[1].value] for r in self.query_map('Namespace', params=[netuid]).records}
        if name != None:
            return namespace[name]
        else:
            return namespace
        
    def namespace(self, netuid: int = None) -> Dict[str, str]:
        
        # Get the namespace for the netuid.
        netuid = self.resolve_netuid(netuid)        
        addresses = { r[0].value: r[1].value for r in self.query_map('Address', params=[netuid]).records}
        namespace = { r[0].value: addresses[r[1].value] for r in self.query_map('Namespace', params=[netuid]).records}
        return namespace
    
    
    def name2uid(self, name: str = None, netuid: int = None) -> int:
        netuid = self.resolve_netuid(netuid)
        name2uid = { r[0].value: r[1].value for r in self.query_map('Namespace', params=[netuid]).records}
        
        if name != None:
            return name2uid[name]
        return name2uid
    
    
    def get_module(self, key:str, netuid:int=None):
        return self.key2module(key, netuid)
        
        
        
        
        
    def key2module(self, key: str = None, netuid: int = None) -> Dict[str, str]:
        modules = self.modules(netuid)
        key2module =  { m['key']: m for m in modules.values() }
        
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return key2module[key_ss58]
        return key2module
        
    
    def modules(self, netuid: int = None) -> Dict[str, ModuleInfo]:
        netuid = self.resolve_netuid(netuid) 
        uid2addresses = { r[0].value: r[1].value for r in self.query_map('Address', params=[netuid]).records}
        uid2key = { r[0].value: r[1].value for r in self.query_map('Keys', params=[netuid]).records}
        uid2name = { r[1].value : r[0].value for r in self.query_map('Namespace', params=[netuid]).records}
        modules = {}
        
        emission = self.emission(netuid=netuid)
        incentive = self.incentive(netuid=netuid)
        dividends = self.dividends(netuid=netuid)
        stake = self.stake(netuid=netuid)
        balances = self.balances()
        
        for uid, address in uid2addresses.items():
            key = uid2key[uid]
            modules[uid] = {
                'uid': uid,
                'address': address,
                'name': uid2name[uid],
                'key': key,
                'emission': emission[uid],
                'incentive': incentive[uid],
                'dividends': dividends[uid],
                'stake': stake[ key],
                'balance': balances[key],
                
            }
            
        return modules
    
    


    @classmethod
    def nodes(cls):
        return c.pm2ls('subspace')
    
    @classmethod
    def query(cls, name,  *params,  block=None):
        self = cls()
        return self.query_map(name=name,params=list(params),block=block).records

    def register_fleet(cls, fleet: List[str], ):
        netuid = cls.resolve_netuid(netuid)
        for name in fleet:
            cls.register(name=name, netuid=netuid)


    @classmethod
    def test(cls, network=subnet):
        subspace = cls()        
        for key_path, key in c.get_keys('test').items():
            port  = c.free_port()
            subspace.register(key=key, 
                              network=network,
                              address=f'{c.external_ip()}:{port}', 
                              name=f'module{key_path}')
        c.print(subspace.query_map('SubnetNamespace', params=[]).records)
        c.print(subspace.uids())
        # for key in keys.values():
        #     subspace.set_weights(key=key, netuid=1, weights=[0.5 for n in modules], uids=[n.uid for n in modules])

    @classmethod
    def test_balance(cls):
        self = cls()
        key = cls.get_key('//Alice')
        c.print(self.get_balance(key.ss58_address))
        
        key2 = cls.get_key('//Bob')
        c.print(self.get_balance(key2.ss58_address))
        
        self.transfer(key=key, dest=key2.ss58_address, amount=10)
        
        c.print(self.get_balance(key2.ss58_address))
        
        # c.print(self.query_map('SubnetNamespace', params=[]).records)
    

    chains = ['dev', 'test', 'main']
    @classmethod
    def build(cls, chain:str = 'dev', verbose:bool=False):
        cls.cmd('cargo build --release', cwd=cls.chain_path, verbose=verbose)
        
        for chain in cls.chains:
            c.print(f'CHAIN: {chain}')
            cls.build_spec(chain)    
        

    @classmethod   
    def purge_chain(cls,
                    chain:str = 'dev',
                    user:str = 'alice',
                    base_path:str = None,
                    sudo = False):
        if base_path == None:
            base_path = cls.resolve_chain_base_path(user=user)
        return c.rm(base_path)
    
    
    @classmethod
    def resolve_chain_base_path(cls, user='alice'):
        return cls.resolve_path(f'{user}')

  
    @classmethod
    def build_spec(cls,
                   chain = 'test',
                   raw:bool  = False,
                   disable_default_bootnode = True,

                   ):

        chain_spec = cls.resolve_chain_spec(chain)
        
            
            

        cmd = f'{cls.chain_release_path} build-spec --chain {chain}'
        
        if disable_default_bootnode:
            cmd += ' --disable-default-bootnode'  
        if raw:
            assert c.exists(chain_spec), f'Chain {chain_spec} does not exist.'
            cmd += ' --raw'
            spec_path =chain_spec.replace('.json', '_raw.json')

        cmd += f' > {chain_spec}'
        return c.cmd(f'bash -c "{cmd}"', cwd=cls.chain_path, verbose=True)

    @classmethod
    def chain_specs(cls):
        specs = c.ls(f'{cls.spec_path}/')
        
        return [spec for spec in specs if '_raw' not in spec]
    
    specs = chain_specs
    @classmethod
    def get_spec(cls, chain):
        chain = cls.resolve_chain_spec(chain)
        
        return c.get_json(chain)

    @classmethod
    def spec_exists(cls, chain):
        c.print(c.exists)
        return c.exists(f'{cls.spec_path}/{chain}.json')


    @classmethod
    def resolve_chain_spec(cls, chain):
        if not chain.endswith('.json'):
            chain = f'{chain}.json'
        if not cls.spec_exists(chain):
            chain = f'{cls.spec_path}/{chain}'
        return chain
        
        

    @classmethod
    def insert_node_key(cls,
                   node='node01',
                   chain = 'jaketensor_raw.json',
                   suri = 'verify kiss say rigid promote level blue oblige window brave rough duty',
                   key_type = 'gran',
                   scheme = 'Sr25519',
                   password_interactive = False,
                   ):
        
        chain = cls.resolve_chain_spec(chain)
        node_path = f'/tmp/{node}'
        
        if key_type == 'aura':
            schmea = 'Sr25519'
        elif key_type == 'gran':
            schmea = 'Ed25519'
        
        if not c.exists(node_path):
            c.mkdir(node_path)

        cmd = f'{cls.chain_release_path} key insert --base-path {node_path}'
        cmd += f' --suri "{suri}"'
        cmd += f' --scheme {scheme}'
        cmd += f' --chain {chain}'
        assert key_type in cls.key_types, f'key_type ({key_type})must be in {cls.key_types}'
        cmd += f' --key-type {key_type}'
        if password_interactive:
            cmd += ' --password-interactive'
        
        c.print(cmd, color='green')
        return c.cmd(cmd, cwd=cls.chain_path, verbose=True)
    
    @classmethod
    def insert_node_keys(cls,
                   aura_suri : str, 
                   grandpa_suri :str,
                    node='node01',
                   password_interactive = False,
                   ):
        '''
        Insert aura and gran keys for a node
        '''
        cls.insert_node_key(node=node, key_type='aura',  suri=aura_suri)
        cls.insert_node_key(node=node, key_type='gran', suri=grandpa_suri)
       
        return c.cmd(cmd, cwd=cls.chain_path, verbose=True)
    
    
    @classmethod
    def nodes(cls, chain='dev'):
        return c.pm2ls(f'{cls.node_prefix()}::{chain}')

    @classmethod
    def node_prefix(cls):
        return f'{cls.module_path()}.node'
    @classmethod
    def start_node(cls,

                 chain:int = 'dev',
                 port:int=30333,
                 rpc_port:int=9933,
                 ws_port:int=9945,
                 user : str = 'alice',
                 telemetry_url:str = 'wss://telemetry.polkadot.io/submit/0',
                 validator: bool = True,          
                 boot_nodes : str = None,       
                 purge_chain:bool = True,
                 remote:bool = True,
                 refresh:bool = True,
                 verbose:bool = True,
                 
                 ):


        
        port = c.resolve_port(port)
        rpc_port = c.resolve_port(rpc_port)
        ws_port = c.resolve_port(ws_port)
        base_path = cls.resolve_chain_base_path(user=user)
        if purge_chain:
            cls.purge_chain(base_path=base_path)
        
        chain_spec = cls.resolve_chain_spec(chain)

        cmd = cls.chain_release_path
        cmd_kwargs = f'''--base-path {base_path} --chain {chain_spec} --{user} --port {port} --ws-port {ws_port} --rpc-port {rpc_port}'''
        
        if validator :
            cmd += ' --validator'
            
        if boot_nodes != None:
            cmd += f' --bootnodes {boot_nodes}'

        
        if remote:
            cmd = c.pm2_start(path=cls.chain_release_path, 
                              name=f'{cls.node_prefix()}::{chain}::{user}',
                              cmd_kwargs=cmd_kwargs,
                              refresh=refresh,
                              verbose=verbose)
        else:
            cls.cmd(f'{cmd} {cmd_kwargs}', color='green',verbose=True)
       
    @classmethod
    def release_exists(cls):
        return c.exists(cls.chain_release_path)
       
    @classmethod
    def start_chain(cls, 
                    users = ['alice','bob'] ,
                    chain:str='dev', 
                    verbose:bool = True,
                    reuse_ports : bool = True,
                    sleep :int = 2,
                    build: bool = False,
                    external:bool = False,
                    boot_nodes : str = None,
                    port_keys: list = ['port','rpc_port','ws_port'],):
        if build:
            cls.build(verbose=verbose)
        avoid_ports = []
        
        ip = c.ip(external=external)
        chain_info_path = f'chain_info.{chain}'
        chain_info = cls.getc(chain_info_path, default={})
        for i, user in enumerate(users):
            

            
            if user in chain_info and reuse_ports:
                node_kwargs = chain_info[user]
                for k in port_keys:
                    node_kwargs[k] = c.resolve_port(node_kwargs[k])
                
            else:
                node_kwargs = {'chain':chain, 'user':user, 'verbose':verbose}
                for k in port_keys:
                    port = c.free_port(avoid_ports=avoid_ports)
                    avoid_ports.append(port)
                    node_kwargs[k] = port
                

            if boot_nodes == None:
                boot_nodes = f'/ip4/{ip}/tcp/{node_kwargs["port"]}/p2p/12D3KooWFYXNTRKT7Nc2podN4RzKMTJKZaYmm7xcCX5aE5RvagxV'
            
            node_kwargs['boot_nodes'] = boot_nodes
            chain_info[user] = c.copy(node_kwargs)

                
            cls.start_node(**chain_info[user])

            cls.sleep(sleep)
        
        cls.putc(chain_info_path, chain_info)


        cls.putc(f'network2url.{chain}', f'{c.external_ip()}:{node_kwargs["ws_port"]}')
        
       
    @classmethod
    def gen_key(cls, *args, **kwargs):
        return c.module('key').gen(*args, **kwargs)
        
    
    
    key_store_path = '/tmp/subspace/keys'

    @classmethod
    def resolve_node_keystore_path(cls, node):
        path = cls.resolve_path(f'nodes/{node}')
        if not c.exists(path):
            c.mkdir(path)
        return path
    
    @classmethod
    def gen_node_keys(cls, path, **kwargs):
        key_class = c.module('key')
        node_path = f'node.{path}'
        c.print(key_class.add_key(path=f'{node_path}.aura', crypto_type='Sr25519'))
        key_class.add_key(path=f'{node_path}.gran',crypto_type='Ed25519')
        return key_class.keys(node_path, **kwargs)
    
    
    def keys(self, netuid = None, **kwargs):
        netuid = self.resolve_netuid(netuid)
        return [r[1].value for r in self.query_map('Keys', params= [netuid], **kwargs).records]
    
    def registered_keys(self, netuid = None, **kwargs):
        key_addresses = self.keys(netuid=netuid, **kwargs)
        address2key = c.address2key()
        registered_keys = {}
        for k_addr in key_addresses:
            if k_addr in address2key:
                registered_keys[address2key[k_addr]] = k_addr
                
        return registered_keys

    reged = registered_keys
    
    
    

    def subnet_state(self, key, netuid = None,  **kwargs):
        netuid = self.resolve_netuid(netuid)
        for r in self.query_map(key, params=[], **kwargs).records:
            if r[0].value == netuid:
                incentive = r[1].value
        
        return incentive
    
    def incentive(self, netuid = None, **kwargs):
        return self.subnet_state('Incentive', netuid=netuid, **kwargs)
        
        
    
    def emission(self, netuid = None, **kwargs):
        return self.subnet_state('Emission', netuid=netuid, **kwargs)
        
    
    def dividends(self, netuid = None, **kwargs):
        return self.subnet_state('Dividends', netuid=netuid, **kwargs)
        
        
    
    
    @classmethod
    def get_node_keys(cls, path):
        for key in cls.gen_node_keys(path):
            c.print(key)
        
    
    @classmethod
    def add_keystore(cls,
                     suri = None ,
                     node = 'alice',
                     chain = 'main',
                     key_type = 'gran',
                     schema = 'Ed25519',
                     password_interactive = False,):
        
        
        if suri is None:
            suri = c.module('key').gen().mnemonic
        base_path = cls.resolve_node_keystore_path(node)
        if key_type == 'gran':
            schema = 'Ed25519'
        elif key_type == 'aura':
            schema = 'Sr25519'
        else:
            raise Exception(f'Unknown key type {key_type}')
        cmd  = f'''
        {cls.chain_release_path} key insert --base-path {base_path}\
        --chain {chain} \
        --scheme {schema} \
        --suri "{suri}" \
        --key-type {key_type}
        '''
        
        if password_interactive:
            cmd = cmd + ' --password-interactive'
        
        return c.cmd(cmd, verbose=True)
        


    @classmethod
    def gen_keys(cls, schema = 'Sr25519' , n:int=2, **kwargs):
        for i in range(n):
            cls.gen_key(schema=schema, **kwargs)
        

    @classmethod
    def localnet(cls):
        cls.cmd('chmod +x ./scripts/*', cwd=f'{cls.repo_path}/subtensor', verbose=True)
        cls.cmd('./scripts/', cwd=f'{cls.repo_path}/subtensor', verbose=True)
    

    
    @classmethod
    def sand(cls, user='Alice'):
        self = cls()
        # namespace = self.query_map('Addresses', params=[0]).records
        # addresses = self.query_map('Addresses', params=[0]).records
        # c.print(self.query_map('Addresses', params=[0]).records)
        key = c.module('key').create_from_uri(user)
        # c.print(key, key.ss58_address)
        # key = c.get_key(user)
        c.print(self.get_balance(key.ss58_address).__dict__)

        # # c.print(self.namespace())
        # c.print(self.namespace())
        

    def uids(self, netuid = 0):
        return [v[1].value for v in self.query_map('Uids',None,  [netuid]).records]

  
if __name__ == "__main__":
    Subspace.run()

    