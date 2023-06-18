
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
    token_decimals = 9
    retry_params = dict(delay=2, tries=2, backoff=2, max_delay=4) # retry params for retrying failed RPC calls
    network2url_map = {
        'local': '127.0.0.1:9945',
        'testnet': '127.0.0.1:9945',
        }
    
    chain = 'subspace'
    chain_path = f'{c.repo_path}/{chain}'
    chain_release_path =  f'{c.repo_path}/subspace/target/release/node-{chain}'
    spec_path = f'{chain_path}/specs'
    key_types = ['aura', 'gran']
    supported_schemas = ['Sr25519', 'Ed25519']
    default_netuid = 1

    
    def __init__( 
        self, 
        subspace: str = 'testnet',
        **kwargs,
    ):


        self.set_subspace( subspace)
    @classmethod
    def network2url(cls, network:str) -> str:
        assert isinstance(network, str), f'network must be a string, not {type(network)}'
        return cls.network2url_map.get(network, 'local')
    @classmethod
    def url2network(cls, url:str) -> str:
        return {v: k for k, v in cls.network2url_map.items()}.get(url, None)
    
    @classmethod
    def resolve_subspace_url(cls, network:str ):  
        external_ip = cls.external_ip()      
        url = cls.network2url(network)
        # resolve url ip if it is its own ip
        ip = url.split('/')[-1].split(':')[0]
        port = url.split(':')[-1]
        if ip.strip() == external_ip.strip():
            ip = '0.0.0.0'
            url = f'{ip}:{port}'
            

        # if not url.startswith('wss://') and not url.startswith('ws://'):
        if not url.startswith('ws://'):
            url = f'ws://{url}'
        
        return url
    def set_subspace(self, 
                subspace:str,
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
        
        self.print(f'Connecting to [cyan bold]{subspace.upper()}[/ cyan bold] ')

        url = self.resolve_subspace_url(subspace)
        self.url = self.chain_endpoint = url
        
        self.print(url, 'broooo red')
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
    def resolve_key(self, key: 'c.Key') -> 'c.Key':
        if key == None:
            if not hasattr(self, 'key'):
                self.key = c.get_key()
            key = self.key
        
        return key
    

    def get_netuid_for_network(self, network: str = None) -> int:
        netuid = self.network2netuid[network]
        return netuid
    
    
    def register (
        self,
        network = 'commune',
        name: str = None,
        address: str = None,
        stake : int = 0,
        key: 'c.Key' = None,
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
        # netuid = self.network2netuid(network)



        # convert name to bytes
        name = name.encode('utf-8')
        address = address.encode('utf-8')
        network = network.encode('utf-8')
    
        # Attempt rolling registration.
        attempts = 1
        c.print(":satellite: Registering...({}/{})".format(attempts, max_allowed_attempts))


        with self.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call( 
                call_module='SubspaceModule',  
                call_function='register', 
                call_params={ 
                    'network': network,
                    'address': address,
                    'name': name,
                    'stake': stake,
                } 
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key  )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )
            
            # process if registration successful, try again if pow is still valid
            response.process_events()
            c.print( response.__dict__ )
    
            


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
        key = self.resolve_key(key)


        # Validate destination address.
        print(dest)
        if not is_valid_address_or_public_key( dest ):
            c.print(":cross_mark: [red]Invalid destination address[/red]:[bold white]\n  {}[/bold white]".format(dest))
            return False

        if isinstance( dest, bytes):
            # Convert bytes to hex string.
            dest = "0x" + dest.hex()


        # Check balance.
        with c.status(":satellite: Checking Balance..."):
            account_balance = self.get_balance( key.ss58_address )
            # check existential deposit.
            existential_deposit = self.get_existential_deposit()

        transfer_balance =  Balance(amount)
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

                fee = payment_info['partialFee'] * 10 ** self.token_decimals
        
        if not keep_alive:
            # Check if the transfer should keep_alive the account
            existential_deposit = 0

        # Check if we have enough balance.
        if account_balance < (transfer_balance + fee + existential_deposit):
            c.print(":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance: {}\n  amount: {}\n  for fee: {}[/bold white]".format( account_balance, transfer_balance, fee ))
            return False

        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to transfer:[bold white]\n  amount: {}\n  from: {}\n  to: {}\n  for fee: {}[/bold white]".format( transfer_balance, key.ss58_address, dest, fee )):
                return False

        with c.status(":satellite: Transferring..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Balances',
                    call_function='transfer',
                    call_params={
                        'dest': dest, 
                        'value': transfer_balance.nano
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
                else:
                    c.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

        if response.is_success:
            with c.status(":satellite: Checking Balance..."):
                new_balance = self.get_balance( key.ss58_address )
                c.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(account_balance, new_balance))
                return True
        
        return False
    
    def get_existential_deposit(
        self,
        block: Optional[int] = None,
    ) -> Optional[Balance]:
        """ Returns the existential deposit for the chain. """
        result = self.query_constant(
            module_name='Balances',
            constant_name='ExistentialDeposit',
            block = block,
        )
        
        if result is None:
            return None
        
        
        

    #################
    #### Serving ####
    #################
    def serve (
        self,
        ip: str, 
        port: int, 
        netuid: int = None,
        key: 'c.Key' =  None,
        wait_for_inclusion: bool = False,
        wait_for_finalization = True,
        prompt: bool = False,
    ) -> bool:
        r""" Subscribes an bittensor endpoint to the substensor chain.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
            protocol (int):
                int representation of the protocol 
            netuid (int):
                network uid to serve on.
            placeholder1 (int):
                placeholder for future use.
            placeholder2 (int):
                placeholder for future use.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        params = {
            'ip': ip_to_int(ip),
            'port': port,
            'netuid': netuid,
            'key': wallet.coldkeypub.ss58_address,
        }

        with c.info(":satellite: Checking Axon..."):
            module = self.get_module_for_pubkey_and_subnet( wallet.hotkey.ss58_address, netuid = netuid )
            module_up_to_date = not module.is_null and params == {
                'ip': ip_to_int(module.module_info.ip),
                'port': module.module_info.port,
                'netuid': module.netuid,
                'key': module.coldkey,
            }

        output = params.copy()
        output['key'] = key.ss58_address

        if module_up_to_date:
            c.print(f":white_heavy_check_mark: [green]Axon already Served[/green]\n"
                                        f"[green not bold]- coldkey: [/green not bold][white not bold]{output['key']}[/white not bold] \n"
                                        f"[green not bold]- Status: [/green not bold] |"
                                        f"[green not bold] ip: [/green not bold][white not bold]{int_to_ip(output['ip'])}[/white not bold] |"
                                        f"[green not bold] port: [/green not bold][white not bold]{output['port']}[/white not bold] | "
                                        f"[green not bold] netuid: [/green not bold][white not bold]{output['netuid']}[/white not bold] |"
            )


            return True

        if prompt:
            output = params.copy()
            output['key'] = key.ss58_address
            if not Confirm.ask("Do you want to serve module:\n  [bold white]{}[/bold white]".format(
                json.dumps(output, indent=4, sort_keys=True)
            )):
                return False

        with c.status(":satellite: Serving module on: [white]{}:{}[/white] ...".format(self.network, netuid)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubspaceModule',
                    call_function='serve_module',
                    call_params=params
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key)
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        c.print(':white_heavy_check_mark: [green]Served[/green]\n  [bold white]{}[/bold white]'.format(
                            json.dumps(params, indent=4, sort_keys=True)
                        ))
                        return True
                    else:
                        c.print(':cross_mark: [green]Failed to Serve module[/green] error: {}'.format(response.error_message))
                        return False
                else:
                    return True


    def add_stake(
            self,
            key_ss58: Optional[str] = None,
            amount: Union[Balance, float] = None, 
            key: 'c.Key' = None,
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            wallet (c.wallet):
                Bittensor wallet object.
            hotkey_ss58 (Optional[str]):
                ss58 address of the hotkey account to stake to
                defaults to the wallet's hotkey.
            amount (Union[Balance, float]):
                Amount to stake as commune balance, or float interpreted as joule.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.

        Raises:
            NotRegisteredError:
                If the wallet is not registered on the chain.
            NotDelegateError:
                If the hotkey is not a delegate on the chain.
        """


        # Flag to indicate if we are using the wallet's own hotkey.
        old_balance = self.get_balance( key.ss58_address )
        # Get current stake
        old_stake = self.get_stake_for_key( key_ss58=key.ss58_address )

        # Convert to c.Balance
        if amount == None:
            # Stake it all.
            staking_balance =  old_balance
        elif not isinstance(amount, Balance ):
            staking_balance = Balance.from_token( amount )
        else:
            staking_balance = amount

        # Remove existential balance to keep key alive.
        if staking_balance > Balance.from_nano( 1000 ):
            staking_balance = staking_balance - Balance.from_nano( 1000 )
        else:
            staking_balance = staking_balance

        # Check enough to stake.
        if staking_balance > old_balance:
            c.print(":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  coldkey: {}[/bold white]".format(old_balance, staking_balance, wallet.name))
            return False
                
        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to stake:[bold white]\n  amount: {}\n  to: {}[/bold white]".format( staking_balance, key.ss58_address) ):
                return False

        try:
            with c.status(":satellite: Staking to: [bold white]{}[/bold white] ...".format(self.network)):

                with self.substrate as substrate:
                    call = substrate.compose_call(
                    call_module='SubspaceModule', 
                    call_function='add_stake',
                    call_params={
                        'key': key.ss58_address,
                        'amount_staked': amount.nano
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )


            if response: # If we successfully staked.
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    c.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                c.print(":white_heavy_check_mark: [green]Finalized[/green]")
                with c.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                    new_balance = self.get_balance( address = key.ss58_address )
                    block = self.get_current_block()
                    new_stake = self.get_stake_for_key(key.ss58_address,block=block) # Get current stake

                    c.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                    c.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                    return True
            else:
                c.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                return False

        except NotRegisteredError as e:
            c.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(key.ss58_address))
            return False
        except StakeError as e:
            c.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            return False





    def unstake (
            self,
            amount: float = None, 
            key: 'c.Key' = None,
            wait_for_inclusion:bool = True, 
            wait_for_finalization:bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Removes stake into the wallet coldkey from the specified hotkey uid.
        Args:
            wallet (c.wallet):
                commune wallet object.
            key_ss58 (Optional[str]):
                ss58 address of the hotkey to unstake from.
                by default, the wallet hotkey is used.
            amount (Union[Balance, float]):
                Amount to stake as commune balance, or float interpreted as joule.
            wait_for_inclusion (bool):
                if set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                if set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        with c.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( key.ss58_address )        
            old_stake = self.get_stake_for_key( key_ss58 = key.ss58_address)


        unstaking_balance = amount

        # Check enough to unstake.
        stake_on_uid = old_stake
        if unstaking_balance > stake_on_uid:
            c.print(":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from key: [white]{}[/white]".format(stake_on_uid, unstaking_balance, key.ss58_address))
            return False
        
        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to unstake:\n[bold white]  amount: {} key: [white]{}[/bold white ]\n?".format( unstaking_balance, key.ss58_address) ):
                return False
        try:
            with c.status(":satellite: Unstaking from chain: [white]{}[/white] ...".format(self.network)):


                with self.substrate as substrate:
                    call = substrate.compose_call(
                    call_module='SubspaceModule', 
                    call_function='remove_stake',
                    call_params={
                        'hotkey': key.ss58_address,
                        'amount_unstaked': amount.nano
                        }
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        return True

                    response.process_events()


            if response: # If we successfully unstaked.
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    c.print(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                c.print(":white_heavy_check_mark: [green]Finalized[/green]")
                with c.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                    new_balance = self.get_balance( address = key.ss58_address )
                    new_stake = self.get_stake_for_key( key_ss58 = key.ss58_address ) # Get stake on hotkey.
                    c.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                    c.print("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                    return True
            else:
                c.print(":cross_mark: [red]Failed[/red]: Error unknown.")
                return False

        except NotRegisteredError as e:
            c.print(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(key.ss58_address))
            return False
        except StakeError as e:
            c.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
            return False

    ########################
    #### Standard Calls ####
    ########################

    """ Queries subspace named storage with params and block. """
    def query_subspace( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[object]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(
                    module='SubspaceModule',
                    storage_function = name,
                    params = params,
                    block_hash = None if block == None else substrate.get_block_hash(block)
                )
        return make_substrate_call_with_retry()

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
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("ImmunityPeriod", block, [netuid] ).value


    """ Returns network MinAllowedWeights hyper parameter """
    def min_allowed_weights (self, netuid: int = None, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("MinAllowedWeights", block, [netuid] ).value

    """ Returns network MaxWeightsLimit hyper parameter """
    def max_weight_limit (self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        if not self.subnet_exists( netuid ): return None
        return U16_NORMALIZED_FLOAT( self.query_subspace('MaxWeightsLimit', block, [netuid] ).value )

    """ Returns network SubnetN hyper parameter """
    def subnetwork_n (self, netuid: int = None, block: Optional[int] = None ) -> int:
        netuid = self.resolve_netuid( netuid )
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('SubnetN', block, [netuid] ).value

    """ Returns network MaxAllowedUids hyper parameter """
    def max_n (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('MaxAllowedUids', block, [netuid] ).value

    """ Returns network BlocksSinceLastStep hyper parameter """
    def blocks_since_epoch (self, netuid: int = None, block: Optional[int] = None) -> int:
        netuid = self.resolve_netuid( netuid )
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('BlocksSinceLastStep', block, [netuid] ).value

    """ Returns network Tempo hyper parameter """
    def tempo (self, netuid: int = None, block: Optional[int] = None) -> int:
        netuid = self.resolve_netuid( netuid )
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('Tempo', block, [netuid] ).value

    ##########################
    #### Account functions ###
    ##########################

    """ Returns the stake under a coldkey - hotkey pairing """
    def get_stake_for_key( self, key_ss58: str, block: Optional[int] = None ) -> Optional['Balance']:
        return Balance.from_nano( self.query_subspace( 'Stake', block, [key_ss58] ).value )

    """ Returns a list of stake tuples (coldkey, balance) for each delegating coldkey including the owner"""
    def get_stake( self,  key_ss58: str, block: Optional[int] = None ) -> List[Tuple[str,'Balance']]:
        return [ (r[0].value, Balance.from_nano( r[1].value ))  for r in self.query_map( 'Stake', block, [key_ss58] ) ]

    """ Returns the module information for this key account """
    def get_module_info( self, key_ss58: str, block: Optional[int] = None ) -> Optional[ModuleInfo]:
        result = self.query_subspace( 'Modules', block, [key_ss58] )        
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

    def serving_rate_limit (self, block: Optional[int] = None ) -> Optional[int]:
        return self.query_subspace( "ServingRateLimit", block ).value

    #####################################
    #### Network Parameters ####
    #####################################

    def subnet_exists( self, netuid: int = None, block: Optional[int] = None ) -> bool:
        netuid = self.resolve_netuid( netuid )
        return self.query_subspace( 'SubnetN', block, [netuid] ).value  

    def get_subnets( self, block: Optional[int] = None ) -> List[int]:
        subnet_netuids = []
        result = self.query_map( 'SubnetN', block )
        if result.records:
            for netuid, exists in result:  
                if exists:
                    subnet_netuids.append( netuid.value )
            
        return subnet_netuids

    def get_total_subnets( self, block: Optional[int] = None ) -> int:
        return self.query_subspace( 'TotalSubnets', block ).value      
    
    def get_emission_value_by_subnet( self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return Balance.from_nano( self.query_subspace( 'EmissionValues', block, [ netuid ] ).value )


    def get_subnets( self, block: Optional[int] = None ) -> List[int]:
        subnets = []
        result = self.query_map( 'NetworksAdded', block )
        if result.records:
            for network in result.records:
                subnets.append( network[0].value )
            return subnets
        else:
            return []

    def get_all_subnets_info( self,
                             block: Optional[int] = None ,
                             retry_params: Dict[str, int] = None) -> List[SubnetInfo]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = []
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="subnetInfo_getSubnetsInfo", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []
        
        return SubnetInfo.list_from_vec_u8( result )

    def get_subnet_info( self, netuid: int, block: Optional[int] = None ) -> Optional[SubnetInfo]:
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="subnetInfo_getSubnetInfo", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return None
        
        return SubnetInfo.from_vec_u8( result )


    ########################################
    #### Module information per subnet ####
    ########################################

    def is_key_registered_any( self, key: str = None, block: Optional[int] = None) -> bool:
        key = self.resolve_key( key )
        return len( self.get_netuids_for_key( key.ss58_address, block) ) > 0
    
    def is_key_registered_on_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> bool:
        uid = self.get_uid_for_key_on_subnet( key_ss58, netuid, block ) != None
        return uid != None

    def is_key_registered( self, key: str, netuid: int, block: Optional[int] = None) -> bool:
        if not isinstance( key, str ):
            key = key.ss58_address
        uid = self.get_uid_for_key_on_subnet( key, netuid, block ) 
        
  
        return uid != None

    def get_uid_for_key_on_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> int:
        return self.query_subspace( 'Uids', block, [ netuid, key_ss58 ] ).value  

    def get_all_uids_for_key( self, key_ss58: str, block: Optional[int] = None) -> List[int]:
        return [ self.get_uid_for_key_on_subnet( key_ss58, netuid, block) for netuid in self.get_netuids_for_key( key_ss58, block)]

    def get_netuids_for_key( self, key_ss58: str, block: Optional[int] = None) -> List[int]:
        result = self.query_map( 'IsNetworkMember', block, [ key_ss58 ] )   
        netuids = []
        for netuid, is_member in result.records:
            if is_member:
                netuids.append( netuid.value )
        return netuids

    def get_module_for_pubkey_and_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None ) -> Optional[ModuleInfo]:
        return self.module_for_uid( self.get_uid_for_key_on_subnet(key_ss58, netuid, block=block), netuid, block = block)

    def get_all_modules_for_key( self, key_ss58: str, block: Optional[int] = None ) -> List[ModuleInfo]:
        netuids = self.get_netuids_for_key( key_ss58, block) 
        uids = [self.get_uid_for_key_on_subnet(key_ss58, netuid) for netuid in netuids] 
        return [self.module_for_uid( uid, netuid ) for uid, netuid in list(zip(uids, netuids))]


    def module_for_wallet( self, key: 'c.Key', netuid = int, block: Optional[int] = None ) -> Optional[ModuleInfo]: 
        return self.get_module_for_pubkey_and_subnet ( key.ss58_address, netuid = netuid, block = block )

    def module_for_uid( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[ModuleInfo]: 
        r""" Returns a list of module from the chain. 
        Args:
            uid ( int ):
                The uid of the module to query for.
            netuid ( int ):
                The uid of the network to query for.
            block ( int ):
                The module at a particular block
        Returns:
            module (Optional[ModuleInfo]):
                module metadata associated with uid or None if it does not exist.
        """
        if uid == None: return ModuleInfo._null_module()
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid, uid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="moduleInfo_getModule", # custom rpc method
                    params=params
                )
        json_body = make_substrate_call_with_retry()
        result = json_body['result']
        self.print(result, 'RESULT')
        if result in (None, []):
            return ModuleInfo._null_module()
        return ModuleInfo.from_vec_u8( result ) 

    def modules(self, netuid: int =0 , block: Optional[int] = None ) -> List[ModuleInfo]: 
        r""" Returns a list of module from the chain. 
        Args:
            netuid ( int ):
                The netuid of the subnet to pull modules from.
            block ( Optional[int] ):
                block to sync from.
        Returns:
            module (List[ModuleInfo]):
                List of module metadata objects.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="moduleInfo_getModules", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []
        
        return ModuleInfo.list_from_vec_u8( result )


    

    ################
    #### Legacy ####
    ################


    def get_balance(self, address: str, block: int = None) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        try:
            @retry(delay=2, tries=3, backoff=2, max_delay=4)
            def make_substrate_call_with_retry():
                with self.substrate as substrate:
                    return substrate.query(
                        module='System',
                        storage_function='Account',
                        params=[address],
                        block_hash = None if block == None else substrate.get_block_hash( block )
                    )
            result = make_substrate_call_with_retry()
        except scalecodec.exceptions.RemainingScaleBytesNotEmptyException:
            c.critical("Your key it legacy formatted, you need to run btcli stake --ammount 0 to reformat it." )
            return Balance(1000)
        return Balance( result.value['data']['free'] )


    
    def get_balances(self, block: int = None) -> Dict[str, Balance]:
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
            bal =  Balance.from_nano(int( r[1]['data']['free'].value ) )
            return_dict[r[0].value] = bal
        return return_dict

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

    def get_balances(self, block: int = None) -> Dict[str, Balance]:
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
            bal = Balance( int( r[1]['data']['free'].value ) )
            return_dict[r[0].value] = bal
        return return_dict
    
    def resolve_network(self, network: Optional[int] = None) -> int:
        if network == None:
            network = self.network
        return network
    
    def get_namespace(self, network = None  ):
        network = self.resolve_network(network)
        self.get_storage_map(module='SubspaceModule', storage_map='ModuleNamespace')
        return {self.substrate.ss58_encode(k[len(map_prefix):]): v for k, v in result}


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


    # @classmethod
    # def get_key(cls, name = None):
    #     # Imports
    #     from substrateinterface import Keypair, KeypairType

    #     # Generate the keypair from shortform private key
        
    #     if name == None:
    #         mnemonic = Keypair.generate_mnemonic()

    #         # Generate the keypair from mnemonic
    #         keypair = Keypair.create_from_mnemonic(mnemonic, crypto_type=KeypairType.SR25519)
    #     else:
    #         # get key from uri or name
    #         keypair = Keypair.create_from_uri(name, crypto_type=KeypairType.SR25519)
    #     return keypair
        


    @classmethod
    def test_keys(cls, 
                     keys: List[str]=['Alice', 'Bob', 'Chris']):
        key_map = {}
        key_class = c.module('key')
        for k in keys:
            key_map[k] = key_class.gen(k)
        return key_map
    @property
    def default_subnet(self) -> str:
        for k, v in self.network2netuid.items():
            if v == 0:
                return k
        raise Exception("No default subnet found.")
    
    
    @property
    def network2netuid(self ) -> Dict[str, str]:
        
        # Get the namespace for the netuid.
        records = self.query_map('SubnetNamespace', params=[]).records
        
        network2netuid = {}
        for r in records:
            name = r[0].value
            uid = int(r[1].value)
            network2netuid[name] = int(uid)
        
        return network2netuid
    
    def subnets(self) -> List[str]:
        return list(self.network2netuid.keys())
        
    def subnet_uids(self) -> List[int]:
        return list(self.network2netuid.values())

    def resolve_netuid(cls, netuid: int = None) -> int:
        if netuid == None:
            netuid = cls.default_netuid
        return netuid

        
    def namespace(self, netuid: int = None) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        
        # Get the namespace for the netuid.
        netuid = self.resolve_netuid(netuid)        
        addresses = { r[0].value: r[1].value for r in self.query_map('Address', params=[netuid]).records}
        namespace = { r[0].value: addresses[r[1].value] for r in self.query_map('Namespace', params=[netuid]).records}
        return namespace
    
    
    def name2key(self, netuid: int = None) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        
        # Get the namespace for the netuid.
        netuid = self.resolve_netuid(netuid)        
        keys = { r[0].value: r[1].value for r in self.query_map('Keys', params=[netuid]).records}
        namespace = { r[0].value: keys[r[1].value] for r in self.query_map('Namespace', params=[netuid]).records}
        return namespace
    
    def key2name(self, netuid: int = None) -> Dict[str, str]:
        return {v:k for k,v in self.name2key(netuid=netuid).items()}
    
    @classmethod
    def nodes(cls):
        return c.pm2ls('subspace')
    
    @classmethod
    def query(cls, name,  *params,  block=None):
        self = cls()
        return self.query_map(name=name,params=list(params),block=block).records

 

    @classmethod
    def test(cls):
        subspace = cls()
        keys = cls.test_keys()
        for idx, (username, key) in enumerate(keys.items()):
            port  = c.free_port()
            address = f'{c.external_ip()}:{port}'
            c.print(key)
            subspace.register(key=key, network='commune', address=address, name=f'module{idx}')
        c.print(subspace.query_map('SubnetNamespace', params=[]).records)
        c.print(subspace.uids())
        # for key in keys.values():
        #     subspace.set_weights(key=key, netuid=1, weights=[0.5 for n in modules], uids=[n.uid for n in modules])

            
    def modules(self, network:int = None):
        netuid = self.resolve_network(network)
        return self.query_map('Modules', params=[netuid]).records
    
    
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
                 validator = True,          
                 boot_nodes = '/ip4/127.0.0.1/tcp/30333/p2p/12D3KooWFYXNTRKT7Nc2podN4RzKMTJKZaYmm7xcCX5aE5RvagxV',       
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
                    sleep :int = 2,
                    build: bool = False):
        if build:
            cls.build(verbose=verbose)
        avoid_ports = []

        for user in users:
            node_kwargs = {'chain':chain, 'user':user, 'verbose':verbose}
            for k in ['port','rpc_port','ws_port']:
                port = c.free_port(avoid_ports=avoid_ports)
                avoid_ports.append(port)
                node_kwargs[k] = port
                
            cls.start_node(**node_kwargs)
            cls.sleep(sleep)
       
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
    
    
    @classmethod
    def keys(cls, *args, **kwargs ):
        return c.module('key').keys(*args, **kwargs)
    
    
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
    def sand(cls, user='alice'):
        self = cls()
        # namespace = self.query_map('Addresses', params=[0]).records
        # addresses = self.query_map('Addresses', params=[0]).records
        # c.print(self.query_map('Addresses', params=[0]).records)
        # key = c.module('key').gen(user)
        # c.print(key, key.ss58_address)
        # c.print(self.get_balance(key.ss58_address).__dict__)

        c.print(self.network2netuid)
        # c.print(self.namespace())
        c.print(self.namespace())
        

    def uids(self, netuid = 0):
        return [v[1].value for v in self.query_map('Uids',None,  [netuid]).records]

  
if __name__ == "__main__":
    Subspace.run()

    