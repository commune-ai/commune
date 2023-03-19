# The MIT License (MIT)
# Copyright © 2021 Yuma nano

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
import torch
from rich.prompt import Confirm, Prompt
from typing import List, Dict, Union, Optional
from multiprocessing import Process
# Mocking imports
import os
import random
import scalecodec
import time
from loguru import logger
logger = logger.opt(colors=True)

import commune
from tqdm import tqdm
import commune.utils.network as net
import commune.subspace.utils.weight_utils as weight_utils
from retry import retry
from substrateinterface import SubstrateInterface, Keypair
from commune.subspace.balance import Balance
from commune.subspace.utils import is_valid_address_or_public_key
from types import SimpleNamespace
from balance import Balance



class Subspace(commune.Module):
    """
    Handles interactions with the subtensor chain.
    """
    def __init__( 
        self, 
        network: str = 'local',
        url: str = '127.0.0.1:9944',
        **kwargs,
    ):
        r""" Initializes a subtensor chain interface.
            Args:
                substrate (:obj:`SubstrateInterface`, `required`): 
                    substrate websocket client.
                network (default='local', type=str)
                    The subtensor network flag. The likely choices are:
                            -- local (local running network)
                            -- nobunaga (staging network)
                            -- nakamoto (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the network argument.
        """
        self.set_substrate( network=network, url=url)


    network2url_map = {
        'local': 'ws://127.0.0.1:9944'
        }
    @classmethod
    def network2url(cls, network:str) -> str:
        return cls.network2url_map.get(network, None)
    @classmethod
    def url2network(cls, url:str) -> str:
        return {v: k for k, v in cls.network2url_map.items()}.get(url, None)
    
    def set_substrate(self, 
                url:str="ws://127.0.0.1:9944", 
                network:str = None,
                websocket:str=None, 
                ss58_format:int=42, 
                type_registry:dict=None, 
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

        if url == None:
            assert network != None, "network or url must be set"
            url = self.network2url(network)
        if not url.startswith('ws://'):
            url = f'ws://{url}'
        
        if network == None:
            network = self.url2network(url)
        self.network = network 
        self.url = url
        
        
        
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
        
        
    
        

    

    def __str__(self) -> str:
        if self.network == self.url:
            # Connecting to chain endpoint without network known.
            return "Subspace({})".format( self.url )
        else:
            # Connecting to network with endpoint known.
            return "Subspace({}, {})".format( self.network, self.url )

    def __repr__(self) -> str:
        return self.__str__()
  
    def endpoint_for_network( 
            self,
            blacklist: List[str] = [] 
        ) -> str:
        r""" Returns a chain endpoint based on self.network.
            Returns None if there are no available endpoints.
        """

        # Chain endpoint overrides the --network flag.
        if self.url != None:
            if self.url in blacklist:
                return None
            else:
                return self.url

    def connect( self, timeout: int = 10, failure = True ) -> bool:
        attempted_endpoints = []
        while True:
            def connection_error_message():
                print('''
                        Check that your internet connection is working and the chain endpoints are available: <blue>{}</blue>
                        The subtensor.network should likely be one of the following choices:
                            -- local - (your locally running node)
                            -- nobunaga - (staging)
                            -- nakamoto - (main)
                        Or you may set the endpoint manually using the --subtensor.chain_endpoint flag 
                        To run a local node (See: docs/running_a_validator.md) \n
                              '''.format( attempted_endpoints) )

            # ---- Get next endpoint ----
            ws_chain_endpoint = self.endpoint_for_network( blacklist = attempted_endpoints )
            if ws_chain_endpoint == None:
                logger.error("No more endpoints available for subtensor.network: <blue>{}</blue>, attempted: <blue>{}</blue>".format(self.network, attempted_endpoints))
                connection_error_message()
                if failure:
                    logger.critical('Unable to connect to network:<blue>{}</blue>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.network))
                else:
                    return False
            
            attempted_endpoints.append(ws_chain_endpoint)

            # --- Attempt connection ----
            try:
                with self.substrate:
                    logger.success("Network:".ljust(20) + "<blue>{}</blue>", self.network)
                    logger.success("Endpoint:".ljust(20) + "<blue>{}</blue>", ws_chain_endpoint)
                    return True
            
            except Exception:
                logger.error( "Error while connecting to network:<blue>{}</blue> at endpoint: <blue>{}</blue>".format(self.network, ws_chain_endpoint))
                connection_error_message()
                if failure:
                    raise RuntimeError('Unable to connect to network:<blue>{}</blue>.\nMake sure your internet connection is stable and the network is properly set.'.format(self.network))
                else:
                    return False



    @property
    def total_issuance (self) -> 'Balance':
        r""" Returns the total token issuance.
        Returns:
            total_issuance (int):
                Total issuance as balance.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return Balance.from_nano( substrate.query(  module='SubspaceModule', storage_function = 'TotalIssuance').value )
        return make_substrate_call_with_retry()

    @property
    def immunity_period (self) -> int:
        r""" Returns the chain registration immunity_period
        Returns:
            immunity_period (int):
                Chain registration immunity_period
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubspaceModule', storage_function = 'ImmunityPeriod' ).value
        return make_substrate_call_with_retry()

    @property
    def total_stake (self) -> 'Balance':
        r""" Returns total stake on the chain.
        Returns:
            total_stake (Balance):
                Total stake as balance.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return Balance.from_nano( substrate.query(  module='SubspaceModule', storage_function = 'TotalStake' ).value )
        return make_substrate_call_with_retry()


    @property
    def n (self) -> int:
        r""" Returns total number of modules on the chain.
        Returns:
            n (int):
                Total number of modules on chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubspaceModule', storage_function = 'N' ).value
        return make_substrate_call_with_retry()

    @property
    def max_n (self) -> int:
        r""" Returns maximum number of module positions on the graph.
        Returns:
            max_n (int):
                Maximum number of module positions on the graph.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubspaceModule', storage_function = 'MaxAllowedUids' ).value
        return make_substrate_call_with_retry()

    @property
    def block (self) -> int:
        r""" Returns current chain block.
        Returns:
            block (int):
                Current chain block.
        """
        return self.get_current_block()

    @property
    def blocks_since_epoch (self) -> int:
        r""" Returns blocks since last epoch.
        Returns:
            blocks_since_epoch (int):
                blocks_since_epoch 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubspaceModule', storage_function = 'BlocksSinceLastStep' ).value
        return make_substrate_call_with_retry()

    @property
    def blocks_per_epoch (self) -> int:
        r""" Returns blocks per chain epoch.
        Returns:
            blocks_per_epoch (int):
                blocks_per_epoch 
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query( module='SubspaceModule', storage_function = 'BlocksPerStep' ).value
        return make_substrate_call_with_retry()

    def get_n (self, block: int = None) -> int:
        r""" Returns total number of modules on the chain.
        Returns:
            n (int):
                Total number of modules on chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query(  
                    module='SubspaceModule', 
                    storage_function = 'N',
                    block_hash = None if block == None else substrate.get_block_hash( block )
                ).value
        return make_substrate_call_with_retry()


    def serve_module (
        self,
        module: 'commune.Axon',
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
        key: Keypair = None,
    ) -> bool:
        r""" Serves the axon to the network.
        Args:
            axon (commune.Axon):
                Axon to serve.
            use_upnpc (:type:bool, `optional`): 
                If true, the axon attempts port forward through your router before 
                subscribing.                
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
        """


        if hasattr(module, 'ip'):
            ip = module.ip
        elif hasattr(module, 'external_ip'):
            ip = module.external_ip()
        else:
            ip = commune.external_ip()
            
        if hasattr(module, 'port'):
            port = module.port
            
        
        commune.log(":white_heavy_check_mark: [green]Found external ip: {}[/green]".format( external_ip ))
        commune.logging.success(prefix = 'External IP', sufix = '<blue>{}</blue>'.format( external_ip ))
        
        # ---- Subscribe to chain ----
        serve_success = self.serve(
                key = key,
                ip = ip,
                port = port,
                wait_for_inclusion = wait_for_inclusion,
                wait_for_finalization = wait_for_finalization,
                prompt = prompt
        )
        return serve_success

    def register (
        self,
        key: 'commune.key',
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        r""" Registers the key to chain.
        Args:
            key (Keypair):
                commune key object.
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
        """

        with commune.__console__.status(":satellite: Checking Account..."):
             module = self.module_for_pubkey( key.ss58_address )
             if not module.is_null:
                 commune.log(":white_heavy_check_mark: [green]Already Registered[/green]:\n  uid: [bold white]{}[/bold white]\n  hotkey: [bold white]{}[/bold white]\n  coldkey: [bold white]{}[/bold white]".format(module.uid, module.key, module.coldkey))
                 return True


        with self.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call( 
                call_module='SubspaceModule',  
                call_function='register', 
                call_params={} 
            )
        extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
        response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )
        
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            commune.log(":white_heavy_check_mark: [green]Sent[/green]")
            return True
        
        # process if registration successful, try again if pow is still valid
        response.process_events()
        if not response.is_success:
            if 'key is already registered' in response.error_message:
                # Error meant that the key is already registered.
                commune.log(":white_heavy_check_mark: [green]Already Registered[/green]")
                return True

            commune.log(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))




    def serve (
            self, 
            ip: str, 
            port: int, 
            wait_for_inclusion: bool = False,
            wait_for_finalization = True,
            prompt: bool = False,
            key: 'Keypair' = None,

        ) -> bool:
        r""" Subscribes an commune endpoint to the substensor chain.
        Args:
            key (Keypair):
                commune key object.
            ip (str):
                endpoint host port i.e. 192.122.31.4
            port (int):
                endpoint port number i.e. 9221
            modality (int):
                int encoded endpoint modality i.e 0 for TEXT
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
            'ip': net.ip_to_int(ip),
            'port': port,
            'ip_type': net.ip_version(ip),
            'key': key.ss58_address,
        }

        with commune.__console__.status(":satellite: Checking Axon..."):
            module = self.module_for_pubkey( key.hotkey.ss58_address )
            module_up_to_date = not module.is_null and params == {
                'ip': module.ip,
                'port': module.port,
            }
            if module_up_to_date:
                commune.log(":white_heavy_check_mark: [green]Already Served[/green]\n  [bold white]ip: {}\n  port: {}\n  modality: {}\n  hotkey: {}\n  coldkey: {}[/bold white]".format(ip, port, modality, key.hotkey.ss58_address, key.coldkeypub.ss58_address))
                return True

        if prompt:
            if not Confirm.ask("Do you want to serve axon:\n  [bold white]ip: {}\n  port: {}\n  modality: {}\n  hotkey: {}\n  coldkey: {}[/bold white]".format(ip, port, modality, key.hotkey.ss58_address, key.coldkeypub.ss58_address)):
                return False
        
        with commune.__console__.status(":satellite: Serving axon on: [white]{}[/white] ...".format(self.network)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubspaceModule',
                    call_function='serve_axon',
                    call_params=params
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key.hotkey)
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                if wait_for_inclusion or wait_for_finalization:
                    response.process_events()
                    if response.is_success:
                        commune.log(':white_heavy_check_mark: [green]Served[/green]\n  [bold white]ip: {}\n  port: {}\n  modality: {}\n  hotkey: {}\n  coldkey: {}[/bold white]'.format(ip, port, modality, key.hotkey.ss58_address, key.coldkeypub.ss58_address ))
                        return True
                    else:
                        commune.log(':cross_mark: [green]Failed to Subscribe[/green] error: {}'.format(response.error_message))
                        return False
                else:
                    return True

    def add_stake(
            self, 
            amount: Union[Balance, float] = None, 
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
            key: Keypair = None,

        ) -> bool:
        r""" Adds the specified amount of stake to passed hotkey uid.
        Args:
            key (Keypair):
                Bittensor key object.
            amount (Union[Balance, float]):
                Amount to stake as commune balance, or float interpreted as token.
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
        """


        with commune.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( key.ss58_address )
            module = self.module_for_pubkey( ss58_key = key.ss58_address )
        if module.is_null:
            commune.log(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(key.hotkey_str))
            return False

        # Covert to Balance
        if amount == None:
            # Stake it all.
            staking_balance = Balance.from_token( old_balance.token )
        elif not isinstance(amount, Balance ):
            staking_balance = Balance.from_token( amount )
        else:
            staking_balance = amount

        # Remove existential balance to keep key alive.
        if staking_balance > Balance.from_nano( 1000 ):
            staking_balance = staking_balance - Balance.from_nano( 1000 )
        else:
            staking_balance = staking_balance

        # Estimate transfer fee.
        staking_fee = None # To be filled.
        with commune.__console__.status(":satellite: Estimating Staking Fees..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubspaceModule', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': key.hotkey.ss58_address,
                        'ammount_staked': staking_balance.nano
                    }
                )
                payment_info = substrate.get_payment_info(call = call, keypair = key)
                if payment_info:
                    staking_fee = Balance.from_nano(payment_info['partialFee'])
                    commune.log("[green]Estimated Fee: {}[/green]".format( staking_fee ))
                else:
                    staking_fee = Balance.from_token( 0.2 )
                    commune.log(":cross_mark: [red]Failed[/red]: could not estimate staking fee, assuming base fee of 0.2")

        # Check enough to unstake.
        if staking_balance > old_balance + staking_fee:
            commune.log(":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  fee: {}\n  coldkey: {}[/bold white]".format(old_balance, staking_balance, staking_fee, key.name))
            return False
                
        with commune.__console__.status(":satellite: Staking to: [bold white]{}[/bold white] ...".format(self.network)):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='SubspaceModule', 
                    call_function='add_stake',
                    call_params={
                        'hotkey': key.hotkey.ss58_address,
                        'ammount_staked': staking_balance.nano
                    }
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    commune.log(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                if response.is_success:
                    commune.log(":white_heavy_check_mark: [green]Finalized[/green]")
                else:
                    commune.log(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

        if response.is_success:
            with commune.__console__.status(":satellite: Checking Balance on: [white]{}[/white] ...".format(self.network)):
                new_balance = self.get_balance( key.ss58_address )
                old_stake = Balance.from_token( module.stake )
                new_stake = Balance.from_token( self.module_for_pubkey( ss58_hotkey = key.ss58_address ).stake)
                commune.log("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_balance, new_balance ))
                commune.log("Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format( old_stake, new_stake ))
                return True
        
        return False

    def transfer(
            self, 
            dest: str, 
            amount: Union[Balance, float], 
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = False,
            prompt: bool = False,
            key: Keypair = None,


        ) -> bool:
        r""" Transfers funds from this key to the destination public key address
        Args:
            key (Keypair):
                Bittensor key object to make transfer from.
            dest (str, ss58_address or ed25519):
                Destination public key address of reciever. 
            amount (Union[Balance, int]):
                Amount to stake as commune balance, or float interpreted as token.
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
                Flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        # Validate destination address.
        if not is_valid_address_or_public_key( dest ):
            commune.log(":cross_mark: [red]Invalid destination address[/red]:[bold white]\n  {}[/bold white]".format(dest))
            return False

        if isinstance( dest, bytes):
            # Convert bytes to hex string.
            dest = "0x" + dest.hex()

        # Unlock key coldkey.

        # Convert to Balance
        if not isinstance(amount, Balance ):
            transfer_balance = Balance.from_token( amount )
        else:
            transfer_balance = amount

        # Check balance.
        with commune.__console__.status(":satellite: Checking Balance..."):
            account_balance = self.get_balance( key.ss58_address )

        # Estimate transfer fee.
        with commune.__console__.status(":satellite: Estimating Transfer Fees..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Balances',
                    call_function='transfer',
                    call_params={
                        'dest': dest, 
                        'value': transfer_balance.nano
                    }
                )
                payment_info = substrate.get_payment_info(call = call, keypair = key)
                transfer_fee = "N/A"
                if payment_info:
                    transfer_fee = Balance.from_nano(payment_info['partialFee'])
                    commune.log("[green]Estimated Fee: {}[/green]".format( transfer_fee ))
                else:
                    commune.log(":cross_mark: [red]Failed[/red]: could not estimate transfer fee, assuming base fee of 0.2")
                    transfer_fee = Balance.from_token( 0.2 )

        if account_balance < transfer_balance + transfer_fee:
            commune.log(":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance: {}\n  amount: {} fee: {}[/bold white]".format( account_balance, transfer_balance, transfer_fee ))
            return False

        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to transfer:[bold white]\n  amount: {}\n  from: {}:{}\n  to: {}\n  for fee: {}[/bold white]".format( transfer_balance, key.name, key.coldkey.ss58_address, dest, transfer_fee )):
                return False

        with commune.__console__.status(":satellite: Transferring..."):
            with self.substrate as substrate:
                call = substrate.compose_call(
                    call_module='Balances',
                    call_function='transfer',
                    call_params={
                        'dest': dest, 
                        'value': transfer_balance.nano
                    }
                )
                extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key.coldkey )
                response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                # We only wait here if we expect finalization.
                if not wait_for_finalization and not wait_for_inclusion:
                    commune.log(":white_heavy_check_mark: [green]Sent[/green]")
                    return True

                # Otherwise continue with finalization.
                response.process_events()
                if response.is_success:
                    commune.log(":white_heavy_check_mark: [green]Finalized[/green]")
                    block_hash = response.block_hash
                    commune.log("[green]Block Hash: {}[/green]".format( block_hash ))
                    explorer_url = "https://explorer.nakamoto.opentensor.ai/#/explorer/query/{block_hash}".format( block_hash = block_hash )
                    commune.log("[green]Explorer Link: {}[/green]".format( explorer_url ))
                else:
                    commune.log(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

        if response.is_success:
            with commune.__console__.status(":satellite: Checking Balance..."):
                new_balance = self.get_balance( key.coldkey.ss58_address )
                commune.log("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(account_balance, new_balance))
                return True
        
        return False

    def unstake (
            self, 
            key: 'Keypair',
            amount: Union[Balance, float] = None, 
            wait_for_inclusion:bool = True, 
            wait_for_finalization:bool = False,
            prompt: bool = False,
        ) -> bool:
        r""" Removes stake into the key coldkey from the specified hotkey uid.
        Args:
            key (Keypair):
                commune key object.
            amount (Union[Balance, float]):
                Amount to stake as commune balance, or float interpreted as token.
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
        # Decrypt keys,

        with commune.__console__.status(":satellite: Syncing with chain: [white]{}[/white] ...".format(self.network)):
            old_balance = self.get_balance( key.ss58_address )
            module = self.module_for_pubkey( ss58_hotkey = key.ss58_address )
        if module.is_null:
            commune.log(":cross_mark: [red]Hotkey: {} is not registered.[/red]".format( key.hotkey_str ))
            return False

        # Covert to Balance
        if amount == None:
            # Unstake it all.
            unstaking_balance = Balance.from_token( module.stake )
        elif not isinstance(amount, Balance ):
            unstaking_balance = Balance.from_token( amount )
        else:
            unstaking_balance = amount

        # Check enough to unstake.
        stake_on_uid = Balance.from_token( module.stake )
        if unstaking_balance > stake_on_uid:
            commune.log(":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from hotkey: [white]{}[/white]".format(stake_on_uid, unstaking_balance, key.hotkey_str))
            return False

        # Estimate unstaking fee.
        unstake_fee = None # To be filled.

        with self.substrate as substrate:
            call = substrate.compose_call(
                call_module='SubspaceModule', 
                call_function='remove_stake',
                call_params={
                    'ammount_unstaked': unstaking_balance.nano
                }
            )
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            response.process_events()

        
        return False
         
    def set_weights(
            self, 
            uids: Union[torch.LongTensor, list],
            weights: Union[torch.FloatTensor, list],
            wait_for_inclusion:bool = False,
            wait_for_finalization:bool = False,
            prompt:bool = False,
            key: Keypair = None,

        ) -> bool:
        r""" Sets the given weights and values on chain for key hotkey account.
        Args:
            key (Keypair):
                commune key object.
            uids (Union[torch.LongTensor, list]):
                uint64 uids of destination modules.
            weights ( Union[torch.FloatTensor, list]):
                weights to set which must floats and correspond to the passed uids.
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
        # First convert types.
        if isinstance( uids, list ):
            uids = torch.tensor( uids, dtype = torch.int64 )
        if isinstance( weights, list ):
            weights = torch.tensor( weights, dtype = torch.float32 )

        # Reformat and normalize.
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit( uids, weights )

        # Ask before moving on.
        if prompt:
            if not Confirm.ask("Do you want to set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]?".format( [float(v/4294967295) for v in weight_vals], weight_uids) ):
                return False

        with commune.__console__.status(":satellite: Setting weights on [white]{}[/white] ...".format(self.network)):
            try:
                with self.substrate as substrate:
                    call = substrate.compose_call(
                        call_module='SubspaceModule',
                        call_function='set_weights',
                        call_params = {'dests': weight_uids, 'weights': weight_vals}
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key.hotkey )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        commune.log(":white_heavy_check_mark: [green]Sent[/green]")
                        return True

                    response.process_events()
                    if response.is_success:
                        commune.log(":white_heavy_check_mark: [green]Finalized[/green]")
                        commune.logging.success(  prefix = 'Set weights', sufix = '<green>Finalized: </green>' + str(response.is_success) )
                    else:
                        commune.log(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                        commune.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(response.error_message) )

            except Exception as e:
                commune.log(":cross_mark: [red]Failed[/red]: error:{}".format(e))
                commune.logging.warning(  prefix = 'Set weights', sufix = '<red>Failed: </red>' + str(e) )
                return False

        if response.is_success:
            commune.log("Set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]".format( [float(v/4294967295) for v in weight_vals], weight_uids ))
            message = '<green>Success: </green>' + f'Set {len(uids)} weights, top 5 weights' + str(list(zip(uids.tolist()[:5], [round (w,4) for w in weights.tolist()[:5]] )))
            logger.debug('Set weights:'.ljust(20) +  message)
            return True
        
        return False

    def get_balance(self, address: str, block: int = None) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (commune.utils.balance.Balance):
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
            logger.critical("Your key it legacy formatted, you need to run btcli stake --ammount 0 to reformat it." )
            return Balance(1000)
        return Balance( result.value['data']['free'] )

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

    def modules(self, block: int = None ) -> List[SimpleNamespace]: 
        r""" Returns a list of module from the chain. 
        Args:
            block (int):
                block to sync from.
        Returns:
            module (List[SimpleNamespace]):
                List of module objects.
        """
        modules = []
        for id in tqdm(range(self.get_n( block ))): 
            try:
                module = self.module_for_uid(id, block)
                modules.append( module )
            except Exception as e:
                logger.error('Exception encountered when pulling module {}: {}'.format(id, e))
                break
        return modules

    @staticmethod
    def _null_module() -> SimpleNamespace:
        module = SimpleNamespace()
        module.active = 0   
        module.stake = 0
        module.incentive = 0
        module.dividends = 0
        module.emission = 0
        module.weights = []
        module.bonds = []
        module.uid = 0
        module.last_update = 0
        module.ip = 0
        module.port = 0
        module.key = "000000000000000000000000000000000000000000000000"
        return module

    @staticmethod
    def _module_dict_to_namespace(module_dict) -> SimpleNamespace:

        U64MAX = 18446744073709551615
        module = SimpleNamespace( **module_dict )
        module.rank = module.rank / U64MAX
        module.incentive = module.incentive / U64MAX
        module.dividends = module.dividends / U64MAX
        module.is_null = False
        return module

    def module_for_uid( self, uid: int, block: int = None ) -> Union[ dict, None ]: 
        r""" Returns a list of module from the chain. 
        Args:
            uid ( int ):
                The uid of the module to query for.
            block ( int ):
                The module at a particular block
        Returns:
            module (dict(moduleMetadata)):
                module object associated with uid or None if it does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                result = dict( substrate.query( 
                    module='SubspaceModule',  
                    storage_function='modules', 
                    params = [ uid ], 
                    block_hash = None if block == None else substrate.get_block_hash( block )
                ).value )
            return result
        result = make_substrate_call_with_retry()
        module = Subspace._module_dict_to_namespace( result )
        return module

    def get_uid_for_key( self, ss58_key: str, block: int = None) -> int:
        r""" Returns true if the passed is registered on the chain.
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a module.
        Returns:
            uid ( int ):
                UID of passed hotkey or -1 if it is non-existent.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query (
                    module='SubspaceModule',
                    storage_function='Keys',
                    params = [ ss58_key ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        # Process the result.
        uid = int(result.value)
        
        module = self.module_for_uid( uid, block )
        if module.key != ss58_key:
            return -1
        else:
            return uid


    def is_key_registered( self, ss58_key: str, block: int = None) -> bool:
        r""" Returns true if the passed hotkey is registered on the chain.
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a module.
        Returns:
            is_registered ( bool):
                True if the passed hotkey is registered on the chain.
        """
        uid = self.get_uid_for_key( ss58_key = ss58_key, block = block)
        if uid == -1:
            return False
        else:
            return True

    def module_for_pubkey( self, ss58_key: str, block: int = None ) -> SimpleNamespace: 
        r""" Returns a list of module from the chain. 
        Args:
            ss58_hotkey ( str ):
                The hotkey to query for a module.

        Returns:
            module ( dict(moduleMetadata) ):
                module object associated with uid or None if it does not exist.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return substrate.query (
                    module='SubspaceModule',
                    storage_function='Keys',
                    params = [ ss58_key ],
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
        result = make_substrate_call_with_retry()
        # Get response uid. This will be zero if it doesn't exist.
        uid = int(result.value)
        module = self.module_for_uid( uid, block )
        if module.key != ss58_key:
            return Subspace._null_module()
        else:
            return module

    def get_n( self, block: int = None ) -> int: 
        r""" Returns the number of modules on the chain at block.
        Args:
            block ( int ):
                The block number to get the module count from.

        Returns:
            n ( int ):
                the number of modules subscribed to the chain.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                return int(substrate.query(  module='SubspaceModule', storage_function = 'N', block_hash = None if block == None else substrate.get_block_hash( block ) ).value)
        return make_substrate_call_with_retry()

    def module_for_key( self, key: 'commune.Wallet', block: int = None ) -> SimpleNamespace: 
        r""" Returns a list of module from the chain. 
        Args:
            key ( `commune.Wallet` ):
                Checks to ensure that the passed key is subscribed.
        Returns:
            module ( dict(moduleMetadata) ):
                module object associated with uid or None if it does not exist.
        """
        return self.module_for_pubkey ( key.ss58_address, block = block )


if __name__ == "__main__":
    import streamlit as st
    self = Subspace()
    
    st.write(self.modules())
    
