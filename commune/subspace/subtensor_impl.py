# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

# Imports
import torch
import bittensor
import scalecodec
from retry import retry
from typing import List, Dict, Union, Optional, Tuple
from substrateinterface import SubstrateInterface
from bittensor.utils.balance import Balance
from bittensor.utils import U16_NORMALIZED_FLOAT, U64_MAX, RAOPERTAO, U16_MAX

# Local imports.
from .chain_data import NeuronInfo, AxonInfo, DelegateInfo, PrometheusInfo, SubnetInfo, NeuronInfoLite
from .errors import *
from .extrinsics.staking import add_stake_extrinsic, add_stake_multiple_extrinsic
from .extrinsics.unstaking import unstake_extrinsic, unstake_multiple_extrinsic
from .extrinsics.serving import serve_extrinsic, serve_axon_extrinsic
from .extrinsics.registration import register_extrinsic, burned_register_extrinsic
from .extrinsics.transfer import transfer_extrinsic
from .extrinsics.set_weights import set_weights_extrinsic
from .extrinsics.prometheus import prometheus_extrinsic

# Logging
from loguru import logger
logger = logger.opt(colors=True)

class Subspace:
    """
    Handles interactions with the subspace chain.
    """
    
    def __init__( 
        self, 
        network: str = 'local',
        url: str = '127.0.0.1:9944',
        **kwargs,
    ):
        r""" Initializes a subspace chain interface.
            Args:
                substrate (:obj:`SubstrateInterface`, `required`): 
                    substrate websocket client.
                network (default='local', type=str)
                    The subspace network flag. The likely choices are:
                            -- local (local running network)
                            -- nobunaga (staging network)
                            -- nakamoto (main network)
                    If this option is set it overloads subspace.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subspace endpoint flag. If set, overrides the network argument.
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
                type_registry:dict=__type_registery__, 
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
        wallet: 'bittensor.wallet',
        netuid: int,
        uids: Union[torch.LongTensor, list],
        weights: Union[torch.FloatTensor, list],
        version_key: int = bittensor.__version_as_int__,
        wait_for_inclusion:bool = False,
        wait_for_finalization:bool = False,
        prompt:bool = False
    ) -> bool:
        return set_weights_extrinsic( 
            subspace=self,
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            prompt=prompt,
        )


    @classmethod
    def name2subnet(cls, name:str) -> int:
        name2subnet = {
            'commune': 0,
            'text': 1,
            # 'image': 2,
            # 'audio': 3,
            # 'image2text': 3,
            # 'text2image': 4,
            # 'speech2text': 5,
            # 'text2speech': 6,
            # 'video': 7,
            # 'video2text': 7,
            # 'text2video': 8,
            # 'video2image': 9,
        }
        subnet = name2subnet.get(name, None)
        
        assert subnet != None, f'Invalid name: {name}, your name must be one of {name2subnet.keys()}'
        
        return subnet
    ######################
    #### Registration ####
    ######################

    def resolve_key(self, key: 'commune.Key') -> 'commune.Key':
        if key == None:
            if not hasattr(self, 'key'):
                self.key = commune.key()
            key = self.key
        
        return key
    
    @classmethod
    def subnets(cls) -> List[int]:
        return self.name2subnet.keys()
    
    
    def register (
        self
        name: str = 'commune',
        key: 'commune.Key' = None,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
        max_allowed_attempts: int = 3,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,

    ) -> bool:

        r""" Registers the wallet to chain.
        Args:
            wallet (bittensor.wallet):
                bittensor wallet object.
            netuid (int):
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
            cuda (bool):
                If true, the wallet should be registered using CUDA device(s).
            dev_id (Union[List[int], int]):
                The CUDA device id to use, or a list of device ids.
            TPB (int):
                The number of threads per block (CUDA).
            num_processes (int):
                The number of processes to use to register.
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
        neduid = self.name2subnet(name)

        
        if not subspace.subnet_exists( netuid ):
            commune.print(":cross_mark: [red]Failed[/red]: error: [bold white]subnet:{}[/bold white] does not exist.".format(netuid))
            return False

        with commune.status(f":satellite: Checking Account on [bold]subnet:{netuid}[/bold]..."):
            neuron = subspace.get_neuron_for_pubkey_and_subnet( key.ss58_address, netuid = netuid )
            if not neuron.is_null:
                commune.print(
                ':white_heavy_check_mark: [green]Already Registered[/green]:\n'\
                'uid: [bold white]{}[/bold white]\n' \
                'netuid: [bold white]{}[/bold white]\n' \
                'hotkey: [bold white]{}[/bold white]\n' \
                'coldkey: [bold white]{}[/bold white]' 
                .format(neuron.uid, neuron.netuid, neuron.hotkey, neuron.coldkey))
                return True


        # Attempt rolling registration.
        attempts = 1
        while True:
            commune.print(":satellite: Registering...({}/{})".format(attempts, max_allowed_attempts))

            # pow failed
            # might be registered already on this subnet
            if (self.is_key_registered(key=key, , netuid = netuid, subspace = subspace, )):
                commune.print(f":white_heavy_check_mark: [green]Already registered on netuid:{netuid}[/green]")
                return True

                with subspace.substrate as substrate:
                    # create extrinsic call
                    call = substrate.compose_call( 
                        call_module='SubspaceModule',  
                        call_function='register', 
                        call_params={ 
                            'netuid': netuid,
                        } 
                    )
                    extrinsic = substrate.create_signed_extrinsic( call = call, keypair = key  )
                    response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization )
                    
                    # We only wait here if we expect finalization.
                    if not wait_for_finalization and not wait_for_inclusion:
                        commune.print(":white_heavy_check_mark: [green]Sent[/green]")
                        return True
                    
                    # process if registration successful, try again if pow is still valid
                    response.process_events()
                    if not response.is_success:
                        if 'key is already registered' in response.error_message:
                            # Error meant that the key is already registered.
                            commune.print(f":white_heavy_check_mark: [green]Already Registered on [bold]subnet:{netuid}[/bold][/green]")
                            return True

                        commune.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))
                        time.sleep(0.5)
                    
                    # Successful registration, final check for neuron and pubkey
                    else:
                        commune.print(":satellite: Checking Balance...")
                        is_registered = self.is_key_registered( key=key, subspace = subspace, netuid = netuid )
                        if is_registered:
                            commune.print(":white_heavy_check_mark: [green]Registered[/green]")
                            return True
                        else:
                            # neuron not found, try again
                            commune.print(":cross_mark: [red]Unknown error. Neuron not found.[/red]")
                            continue
            
                    
            if attempts < max_allowed_attempts:
                #Failed registration, retry pow
                attempts += 1
                commune.print( ":satellite: Failed registration, retrying pow ...({}/{})".format(attempts, max_allowed_attempts))
            else:
                # Failed to register after max attempts.
                commune.print( "[red]No more attempts.[/red]" )
                return False 

            
        
    def burned_register (
        self,
        wallet: 'bittensor.Wallet',
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False
    ) -> bool:
        """ Registers the wallet to chain by recycling TAO."""
        return burned_register_extrinsic( 
            subspace = self, 
            wallet = wallet, 
            netuid = netuid, 
            wait_for_inclusion = wait_for_inclusion, 
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt
        )

    ##################
    #### Transfer ####
    ##################
    def transfer(
        self,
        wallet: 'bittensor.wallet',
        dest: str, 
        amount: Union[Balance, float], 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Transfers funds from this wallet to the destination public key address"""
        return transfer_extrinsic(
            subspace = self,
            wallet = wallet,
            dest = dest,
            amount = amount,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
            prompt = prompt
        )
    
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
        
        return Balance.from_rao(result.value)

    #################
    #### Serving ####
    #################
    def serve (
        self,
        wallet: 'bittensor.wallet',
        ip: str, 
        port: int, 
        protocol: int, 
        netuid: int,
        placeholder1: int = 0,
        placeholder2: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization = True,
        prompt: bool = False,
    ) -> bool:
        return serve_extrinsic( self, wallet, ip, port, protocol, netuid , placeholder1, placeholder2, wait_for_inclusion, wait_for_finalization)

    def serve_axon (
        self,
        axon: 'bittensor.Axon',
        use_upnpc: bool = False,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        prompt: bool = False,
    ) -> bool:
        return serve_axon_extrinsic( self, axon, use_upnpc, wait_for_inclusion, wait_for_finalization)

    def serve_prometheus (
        self,
        wallet: 'bittensor.wallet',
        port: int,
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return prometheus_extrinsic( self, wallet = wallet, port = port, netuid = netuid, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization)
    #################
    #### Staking ####
    #################
    def add_stake(
        self, 
        wallet: 'bittensor.wallet',
        hotkey_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Adds the specified amount of stake to passed hotkey uid. """
        return add_stake_extrinsic( 
            subspace = self, 
            wallet = wallet,
            hotkey_ss58 = hotkey_ss58, 
            amount = amount, 
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt
        )

    def add_stake_multiple (
        self, 
        wallet: 'bittensor.wallet',
        hotkey_ss58s: List[str],
        amounts: List[Union[Balance, float]] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Adds stake to each hotkey_ss58 in the list, using each amount, from a common coldkey."""
        return add_stake_multiple_extrinsic( self, wallet, hotkey_ss58s, amounts, wait_for_inclusion, wait_for_finalization, prompt)

    ###################
    #### Unstaking ####
    ###################
    def unstake_multiple (
        self,
        wallet: 'bittensor.wallet',
        hotkey_ss58s: List[str],
        amounts: List[Union[Balance, float]] = None, 
        wait_for_inclusion: bool = True, 
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Removes stake from each hotkey_ss58 in the list, using each amount, to a common coldkey. """
        return unstake_multiple_extrinsic( self, wallet, hotkey_ss58s, amounts, wait_for_inclusion, wait_for_finalization, prompt)

   

    def unstake (
        self,
        wallet: 'bittensor.wallet',
        hotkey_ss58: Optional[str] = None,
        amount: Union[Balance, float] = None, 
        wait_for_inclusion:bool = True, 
        wait_for_finalization:bool = False,
        prompt: bool = False,
    ) -> bool:
        """ Removes stake into the wallet coldkey from the specified hotkey uid."""
        return unstake_extrinsic( self, wallet, hotkey_ss58, amount, wait_for_inclusion, wait_for_finalization, prompt )


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
    def query_map_subspace( self, name: str, block: Optional[int] = None, params: Optional[List[object]] = [] ) -> Optional[object]:
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

    """ Returns network Rho hyper parameter """
    def rho (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace( "Rho", block, [netuid] ).value

    """ Returns network Kappa hyper parameter """
    def kappa (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid ): return None
        return U16_NORMALIZED_FLOAT( self.query_subspace( "Kappa", block, [netuid] ).value )

    """ Returns network Difficulty hyper parameter """
    def difficulty (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace( "Difficulty", block, [netuid] ).value
    
    """ Returns network Burn hyper parameter """
    def burn (self, netuid: int, block: Optional[int] = None ) -> Optional[bittensor.Balance]:
        if not self.subnet_exists( netuid ): return None
        return bittensor.Balance.from_rao( self.query_subspace( "Burn", block, [netuid] ).value )

    """ Returns network ImmunityPeriod hyper parameter """
    def immunity_period (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("ImmunityPeriod", block, [netuid] ).value

    """ Returns network ValidatorBatchSize hyper parameter """
    def validator_batch_size (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("ValidatorBatchSize", block, [netuid] ).value

    """ Returns network ValidatorPruneLen hyper parameter """
    def validator_prune_len (self, netuid: int, block: Optional[int] = None ) -> int:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("ValidatorPruneLen", block, [netuid] ).value

    """ Returns network ValidatorLogitsDivergence hyper parameter """
    def validator_logits_divergence (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid ): return None
        return U16_NORMALIZED_FLOAT(self.query_subspace("ValidatorLogitsDivergence", block, [netuid]).value)

    """ Returns network ValidatorSequenceLength hyper parameter """
    def validator_sequence_length (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("ValidatorSequenceLength", block, [netuid] ).value

    """ Returns network ValidatorEpochsPerReset hyper parameter """
    def validator_epochs_per_reset (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("ValidatorEpochsPerReset", block, [netuid] ).value

    """ Returns network ValidatorEpochLen hyper parameter """
    def validator_epoch_length (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("ValidatorEpochLen", block, [netuid] ).value

    """ Returns network ValidatorEpochLen hyper parameter """
    def validator_exclude_quantile (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid ): return None
        return U16_NORMALIZED_FLOAT( self.query_subspace("ValidatorExcludeQuantile", block, [netuid] ).value )

    """ Returns network MaxAllowedValidators hyper parameter """
    def max_allowed_validators(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace( 'MaxAllowedValidators', block, [netuid] ).value
        
    """ Returns network MinAllowedWeights hyper parameter """
    def min_allowed_weights (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace("MinAllowedWeights", block, [netuid] ).value

    """ Returns network MaxWeightsLimit hyper parameter """
    def max_weight_limit (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid ): return None
        return U16_NORMALIZED_FLOAT( self.query_subspace('MaxWeightsLimit', block, [netuid] ).value )

    """ Returns network ScalingLawPower hyper parameter """
    def scaling_law_power (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('ScalingLawPower', block, [netuid] ).value / 100.

    """ Returns network SynergyScalingLawPower hyper parameter """
    def synergy_scaling_law_power (self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('SynergyScalingLawPower', block, [netuid] ).value / 100.

    """ Returns network SubnetworkN hyper parameter """
    def subnetwork_n (self, netuid: int, block: Optional[int] = None ) -> int:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('SubnetworkN', block, [netuid] ).value

    """ Returns network MaxAllowedUids hyper parameter """
    def max_n (self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('MaxAllowedUids', block, [netuid] ).value

    """ Returns network BlocksSinceLastStep hyper parameter """
    def blocks_since_epoch (self, netuid: int, block: Optional[int] = None) -> int:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('BlocksSinceLastStep', block, [netuid] ).value

    """ Returns network Tempo hyper parameter """
    def tempo (self, netuid: int, block: Optional[int] = None) -> int:
        if not self.subnet_exists( netuid ): return None
        return self.query_subspace('Tempo', block, [netuid] ).value

    ##########################
    #### Account functions ###
    ##########################

    """ Returns the total stake held on a coldkey across all hotkeys including delegates"""
    def get_total_stake_for_key( self, ss58_address: str, block: Optional[int] = None ) -> Optional['bittensor.Balance']:
        return bittensor.Balance.from_rao( self.query_subspace( 'TotalKeyStake', block, [ss58_address] ).value )

    """ Returns the stake under a coldkey - hotkey pairing """
    def get_stake_for_key( self, key_ss58: str, block: Optional[int] = None ) -> Optional['bittensor.Balance']:
        return bittensor.Balance.from_rao( self.query_subspace( 'Stake', block, [key_ss58] ).value )

    """ Returns a list of stake tuples (coldkey, balance) for each delegating coldkey including the owner"""
    def get_stake( self,  key_ss58: str, block: Optional[int] = None ) -> List[Tuple[str,'bittensor.Balance']]:
        return [ (r[0].value, bittensor.Balance.from_rao( r[1].value ))  for r in self.query_map_subspace( 'Stake', block, [key_ss58] ) ]

    """ Returns the axon information for this key account """
    def get_axon_info( self, key_ss58: str, block: Optional[int] = None ) -> Optional[AxonInfo]:
        result = self.query_subspace( 'Axons', block, [key_ss58 ] )        
        if result != None:
            return AxonInfo(
                ip = commune.utils.networking.ip_from_int( result.value.ip ),
                port = result.value.port,
            )
        else:
            return None


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

    def total_issuance (self, block: Optional[int] = None ) -> 'bittensor.Balance':
        return bittensor.Balance.from_rao( self.query_subspace( 'TotalIssuance', block ).value )

    def total_stake (self,block: Optional[int] = None ) -> 'bittensor.Balance':
        return bittensor.Balance.from_rao( self.query_subspace( "TotalStake", block ).value )

    def serving_rate_limit (self, block: Optional[int] = None ) -> Optional[int]:
        return self.query_subspace( "ServingRateLimit", block ).value

    #####################################
    #### Network Parameters ####
    #####################################

    def subnet_exists( self, netuid: int, block: Optional[int] = None ) -> bool:
        return self.query_subspace( 'NetworksAdded', block, [netuid] ).value  

    def get_all_subnet_netuids( self, block: Optional[int] = None ) -> List[int]:
        subnet_netuids = []
        result = self.query_map_subspace( 'NetworksAdded', block )
        if result.records:
            for netuid, exists in result:  
                if exists:
                    subnet_netuids.append( netuid.value )
            
        return subnet_netuids

    def get_total_subnets( self, block: Optional[int] = None ) -> int:
        return self.query_subspace( 'TotalNetworks', block ).value      

    def get_subnet_modality( self, netuid: int, block: Optional[int] = None ) -> Optional[int]:
        return self.query_subspace( 'NetworkModality', block, [netuid] ).value   

    def get_subnet_connection_requirement( self, netuid_0: int, netuid_1: int, block: Optional[int] = None) -> Optional[int]:
        return self.query_subspace( 'NetworkConnect', block, [netuid_0, netuid_1] ).value

    def get_emission_value_by_subnet( self, netuid: int, block: Optional[int] = None ) -> Optional[float]:
        return bittensor.Balance.from_rao( self.query_subspace( 'EmissionValues', block, [ netuid ] ).value )

    def get_subnet_connection_requirements( self, netuid: int, block: Optional[int] = None) -> Dict[str, int]:
        result = self.query_map_subspace( 'NetworkConnect', block, [netuid] )
        if result.records:
            requirements = {}
            for tuple in result.records:
                requirements[str(tuple[0].value)] = tuple[1].value
        else:
            return {}

    def get_subnets( self, block: Optional[int] = None ) -> List[int]:
        subnets = []
        result = self.query_map_subspace( 'NetworksAdded', block )
        if result.records:
            for network in result.records:
                subnets.append( network[0].value )
            return subnets
        else:
            return []

    def get_all_subnets_info( self, block: Optional[int] = None ) -> List[SubnetInfo]:
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
    #### Neuron information per subnet ####
    ########################################

    def is_key_registered_any( self, key: str = None, block: Optional[int] = None) -> bool:
        key = self.resolve_key( key )
        return len( self.get_netuids_for_key( key.ss58_address, block) ) > 0
    
    def is_key_registered_on_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> bool:
        return self.get_uid_for_key_on_subnet( key_ss58, netuid, block ) != None

    def is_key_registered( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> bool:
        return self.get_uid_for_key_on_subnet( key_ss58, netuid, block ) != None

    def get_uid_for_key_on_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> int:
        return self.query_subspace( 'Uids', block, [ netuid, key_ss58 ] ).value  

    def get_all_uids_for_key( self, key_ss58: str, block: Optional[int] = None) -> List[int]:
        return [ self.get_uid_for_key_on_subnet( key_ss58, netuid, block) for netuid in self.get_netuids_for_key( key_ss58, block)]

    def get_netuids_for_key( self, key_ss58: str, block: Optional[int] = None) -> List[int]:
        result = self.query_map_subspace( 'IsNetworkMember', block, [ key_ss58 ] )   
        netuids = []
        for netuid, is_member in result.records:
            if is_member:
                netuids.append( netuid.value )
        return netuids

    def get_neuron_for_pubkey_and_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfo]:
        return self.neuron_for_uid( self.get_uid_for_key_on_subnet(key_ss58, netuid, block=block), netuid, block = block)

    def get_all_neurons_for_key( self, key_ss58: str, block: Optional[int] = None ) -> List[NeuronInfo]:
        netuids = self.get_netuids_for_key( key_ss58, block) 
        uids = [self.get_uid_for_key_on_subnet(key_ss58, net) for net in netuids] 
        return [self.neuron_for_uid( uid, net ) for uid, net in list(zip(uids, netuids))]

    def neuron_has_validator_permit( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[bool]:
        return self.query_subspace( 'ValidatorPermit', block, [ netuid, uid ] ).value

    def neuron_for_wallet( self, key: 'commune.Key', netuid = int, block: Optional[int] = None ) -> Optional[NeuronInfo]: 
        return self.get_neuron_for_pubkey_and_subnet ( key.ss58_address, netuid = netuid, block = block )

    def neuron_for_uid( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfo]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            uid ( int ):
                The uid of the neuron to query for.
            netuid ( int ):
                The uid of the network to query for.
            block ( int ):
                The neuron at a particular block
        Returns:
            neuron (Optional[NeuronInfo]):
                neuron metadata associated with uid or None if it does not exist.
        """
        if uid == None: return NeuronInfo._null_neuron()
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid, uid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="neuronInfo_getNeuron", # custom rpc method
                    params=params
                )
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return NeuronInfo._null_neuron()
        
        return NeuronInfo.from_vec_u8( result ) 

    def neurons(self, netuid: int, block: Optional[int] = None ) -> List[NeuronInfo]: 
        r""" Returns a list of neuron from the chain. 
        Args:
            netuid ( int ):
                The netuid of the subnet to pull neurons from.
            block ( Optional[int] ):
                block to sync from.
        Returns:
            neuron (List[NeuronInfo]):
                List of neuron metadata objects.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="neuronInfo_getNeurons", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []
        
        return NeuronInfo.list_from_vec_u8( result )
    
    def neuron_for_uid_lite( self, uid: int, netuid: int, block: Optional[int] = None ) -> Optional[NeuronInfoLite]: 
        r""" Returns a list of neuron lite from the chain. 
        Args:
            uid ( int ):
                The uid of the neuron to query for.
            netuid ( int ):
                The uid of the network to query for.
            block ( int ):
                The neuron at a particular block
        Returns:
            neuron (Optional[NeuronInfoLite]):
                neuron metadata associated with uid or None if it does not exist.
        """
        if uid == None: return NeuronInfoLite._null_neuron()
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid, uid]
                if block_hash:
                    params = params + [block_hash] 
                return substrate.rpc_request(
                    method="neuronInfo_getNeuronLite", # custom rpc method
                    params=params
                )
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return NeuronInfoLite._null_neuron()
        
        return NeuronInfoLite.from_vec_u8( result ) 

    def neurons_lite(self, netuid: int, block: Optional[int] = None ) -> List[NeuronInfoLite]: 
        r""" Returns a list of neuron lite from the chain. 
        Args:
            netuid ( int ):
                The netuid of the subnet to pull neurons from.
            block ( Optional[int] ):
                block to sync from.
        Returns:
            neuron (List[NeuronInfoLite]):
                List of neuron lite metadata objects.
        """
        @retry(delay=2, tries=3, backoff=2, max_delay=4)
        def make_substrate_call_with_retry():
            with self.substrate as substrate:
                block_hash = None if block == None else substrate.get_block_hash( block )
                params = [netuid]
                if block_hash:
                    params = params + [block_hash]
                return substrate.rpc_request(
                    method="neuronInfo_getNeuronsLite", # custom rpc method
                    params=params
                )
        
        json_body = make_substrate_call_with_retry()
        result = json_body['result']

        if result in (None, []):
            return []
        
        return NeuronInfoLite.list_from_vec_u8( result )

    def metagraph( self, netuid: int, block: Optional[int] = None, lite: bool = True ) -> 'bittensor.Metagraph':
        r""" Returns the metagraph for the subnet.
        Args:
            netuid ( int ):
                The network uid of the subnet to query.
            block (Optional[int]):
                The block to create the metagraph for.
                Defaults to latest.
            lite (bool, default=True):
                If true, returns a metagraph using the lite sync (no weights, no bonds)
        Returns:
            metagraph ( `bittensor.Metagraph` ):
                The metagraph for the subnet at the block.
        """
        status: Optional['rich.console.Status'] = None
        if bittensor.__use_console__:
            status = commune.status("Synchronizing Metagraph...", spinner="earth")
            status.start()
        
        # Get neurons.
        if lite:
            neurons = self.neurons_lite( netuid = netuid, block = block )
        else:
            neurons = self.neurons( netuid = netuid, block = block )
        
        # Get subnet info.
        subnet_info: Optional[bittensor.SubnetInfo] = self.get_subnet_info( netuid = netuid, block = block )
        if subnet_info == None:
            status.stop() if status else ...
            raise ValueError('Could not find subnet info for netuid: {}'.format(netuid))

        status.stop() if status else ...

        # Create metagraph.
        block_number = self.block
        
        metagraph = bittensor.metagraph.from_neurons( network = self.network, netuid = netuid, info = subnet_info, neurons = neurons, block = block_number )
        print("Metagraph subspace: ", self.network)
        return metagraph

    ################
    #### Transfer ##
    ################


    

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
            logger.critical("Your wallet it legacy formatted, you need to run btcli stake --ammount 0 to reformat it." )
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
            bal = bittensor.Balance( int( r[1]['data']['free'].value ) )
            return_dict[r[0].value] = bal
        return return_dict

    @staticmethod
    def _null_neuron() -> NeuronInfo:
        neuron = NeuronInfo(
            uid = 0,
            netuid = 0,
            active =  0,
            stake = '0',
            rank = 0,
            emission = 0,
            incentive = 0,
            consensus = 0,
            trust = 0,
            dividends = 0,
            last_update = 0,
            weights = [],
            bonds = [],
            is_null = True,
            key = "000000000000000000000000000000000000000000000000",
        )
        return neuron




# Substrate ss58_format
__ss58_format__ = 42

# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import argparse
import copy
import os

import bittensor
from loguru import logger
from substrateinterface import SubstrateInterface
from torch.cuda import is_available as is_cuda_available

from bittensor.utils import strtobool_with_default
from .naka_subspace_impl import Subspace as Nakamoto_subspace
from . import subspace_impl, subspace_mock

logger = logger.opt(colors=True)

GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME = 'node-subspace'

class subspace:
    """Factory Class for both bittensor.Subspace and Mock_Subspace Classes

    The Subspace class handles interactions with the substrate subspace chain.
    By default, the Subspace class connects to the Nakamoto which serves as the main bittensor network.
    
    """
    
    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            network: str = None,
            chain_endpoint: str = None,
            _mock: bool = None,
        ) -> 'bittensor.Subspace':
        r""" Initializes a subspace chain interface.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.subspace.config()
                network (default='local', type=str)
                    The subspace network flag. The likely choices are:
                            -- local (local running network)
                            -- finney (main network)
                            -- mock (mock network for testing.)
                    If this option is set it overloads subspace.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subspace endpoint flag. If set, overrides the network argument.
                _mock (bool, `optional`):
                    Returned object is mocks the underlying chain connection.
        """
        if config == None: config = subspace.config()
        config = copy.deepcopy( config )

        # Returns a mocked connection with a background chain connection.
        config.subspace._mock = _mock if _mock != None else config.subspace._mock
        if config.subspace._mock == True or network == 'mock' or config.subspace.get('network', bittensor.defaults.subspace.network) == 'mock':
            config.subspace._mock = True
            return subspace_mock.mock_subspace.mock()
        
        # Determine config.subspace.chain_endpoint and config.subspace.network config.
        # If chain_endpoint is set, we override the network flag, otherwise, the chain_endpoint is assigned by the network.
        # Argument importance: chain_endpoint > network > config.subspace.chain_endpoint > config.subspace.network
       
        # Select using chain_endpoint arg.
        if chain_endpoint != None:
            config.subspace.chain_endpoint = chain_endpoint
            if network != None:
                config.subspace.network = network
            else:
                config.subspace.network = config.subspace.get('network', bittensor.defaults.subspace.network)
            
        # Select using network arg.
        elif network != None:
            config.subspace.chain_endpoint = subspace.determine_chain_endpoint( network )
            config.subspace.network = network
            
        # Select using config.subspace.chain_endpoint
        elif config.subspace.chain_endpoint != None:
            config.subspace.chain_endpoint = config.subspace.chain_endpoint
            config.subspace.network = config.subspace.get('network', bittensor.defaults.subspace.network)
         
        # Select using config.subspace.network
        elif config.subspace.get('network', bittensor.defaults.subspace.network) != None:
            config.subspace.chain_endpoint = subspace.determine_chain_endpoint( config.subspace.get('network', bittensor.defaults.subspace.network) )
            config.subspace.network = config.subspace.get('network', bittensor.defaults.subspace.network)
            
        # Fallback to defaults.
        else:
            config.subspace.chain_endpoint = subspace.determine_chain_endpoint( bittensor.defaults.subspace.network )
            config.subspace.network = bittensor.defaults.subspace.network
        
        # make sure it's wss:// or ws://
        # If it's bellagene (parachain testnet) then it has to be wss
        endpoint_url: str = config.subspace.chain_endpoint
        
        # make sure formatting is good
        endpoint_url = bittensor.utils.networking.get_formatted_ws_endpoint_url(endpoint_url)
        
        

        subspace.check_config( config )
        network = config.subspace.get('network', bittensor.defaults.subspace.network)
        if network == 'nakamoto':
            substrate = SubstrateInterface(
                ss58_format = bittensor.__ss58_format__,
                use_remote_preset=True,
                url = endpoint_url,
            )
            # Use nakamoto-specific subspace.
            return Nakamoto_subspace( 
                substrate = substrate,
                network = config.subspace.get('network', bittensor.defaults.subspace.network),
                chain_endpoint = config.subspace.chain_endpoint,
            )
        else:
            substrate = SubstrateInterface(
                ss58_format = bittensor.__ss58_format__,
                use_remote_preset=True,
                url = endpoint_url,
                type_registry=bittensor.__type_registry__
            )
            return subspace_impl.Subspace( 
                substrate = substrate,
                network = config.subspace.get('network', bittensor.defaults.subspace.network),
                chain_endpoint = config.subspace.chain_endpoint,
            )

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        subspace.add_args( parser )
        return bittensor.config( parser )

    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'subspace.network', default = bittensor.defaults.subspace.network, type=str,
                                help='''The subspace network flag. The likely choices are:
                                        -- finney (main network)
                                        -- local (local running network)
                                        -- mock (creates a mock connection (for testing))
                                    If this option is set it overloads subspace.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
            parser.add_argument('--' + prefix_str + 'subspace.chain_endpoint', default = bittensor.defaults.subspace.chain_endpoint, type=str, 
                                help='''The subspace endpoint flag. If set, overrides the --network flag.
                                    ''')       
            parser.add_argument('--' + prefix_str + 'subspace._mock', action='store_true', help='To turn on subspace mocking for testing purposes.', default=bittensor.defaults.subspace._mock)
            # registration args. Used for register and re-register and anything that calls register.
            parser.add_argument('--' + prefix_str + 'subspace.register.num_processes', '-n', dest=prefix_str + 'subspace.register.num_processes', help="Number of processors to use for registration", type=int, default=bittensor.defaults.subspace.register.num_processes)
            parser.add_argument('--' + prefix_str + 'subspace.register.update_interval', '--' + prefix_str + 'subspace.register.cuda.update_interval', '--' + prefix_str + 'cuda.update_interval', '-u', help="The number of nonces to process before checking for next block during registration", type=int, default=bittensor.defaults.subspace.register.update_interval)
            parser.add_argument('--' + prefix_str + 'subspace.register.no_output_in_place', '--' + prefix_str + 'no_output_in_place', dest="subspace.register.output_in_place", help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.", action='store_false', required=False, default=bittensor.defaults.subspace.register.output_in_place)
            parser.add_argument('--' + prefix_str + 'subspace.register.verbose', help="Whether to ouput the registration statistics verbosely.", action='store_true', required=False, default=bittensor.defaults.subspace.register.verbose)
            
            ## Registration args for CUDA registration.
            parser.add_argument( '--' + prefix_str + 'subspace.register.cuda.use_cuda', '--' + prefix_str + 'cuda', '--' + prefix_str + 'cuda.use_cuda', default=argparse.SUPPRESS, help='''Set flag to use CUDA to register.''', action="store_true", required=False )
            parser.add_argument( '--' + prefix_str + 'subspace.register.cuda.no_cuda', '--' + prefix_str + 'no_cuda', '--' + prefix_str + 'cuda.no_cuda', dest=prefix_str + 'subspace.register.cuda.use_cuda', default=argparse.SUPPRESS, help='''Set flag to not use CUDA for registration''', action="store_false", required=False )

            parser.add_argument( '--' + prefix_str + 'subspace.register.cuda.dev_id', '--' + prefix_str + 'cuda.dev_id',  type=int, nargs='+', default=argparse.SUPPRESS, help='''Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).''', required=False )
            parser.add_argument( '--' + prefix_str + 'subspace.register.cuda.TPB', '--' + prefix_str + 'cuda.TPB', type=int, default=bittensor.defaults.subspace.register.cuda.TPB, help='''Set the number of Threads Per Block for CUDA.''', required=False )

            parser.add_argument('--netuid', type=int, help='netuid for subnet to serve this neuron on', default=argparse.SUPPRESS)        
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def add_defaults(cls, defaults ):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.subspace = bittensor.Config()
        defaults.subspace.network = os.getenv('BT_SUBTENSOR_NETWORK') if os.getenv('BT_SUBTENSOR_NETWORK') != None else 'finney'
        defaults.subspace.chain_endpoint = os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') if os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') != None else None
        defaults.subspace._mock = os.getenv('BT_SUBTENSOR_MOCK') if os.getenv('BT_SUBTENSOR_MOCK') != None else False

        defaults.subspace.register = bittensor.Config()
        defaults.subspace.register.num_processes = os.getenv('BT_SUBTENSOR_REGISTER_NUM_PROCESSES') if os.getenv('BT_SUBTENSOR_REGISTER_NUM_PROCESSES') != None else None # uses processor count by default within the function
        defaults.subspace.register.update_interval = os.getenv('BT_SUBTENSOR_REGISTER_UPDATE_INTERVAL') if os.getenv('BT_SUBTENSOR_REGISTER_UPDATE_INTERVAL') != None else 50_000
        defaults.subspace.register.output_in_place = True
        defaults.subspace.register.verbose = False

        defaults.subspace.register.cuda = bittensor.Config()
        defaults.subspace.register.cuda.dev_id = [0]
        defaults.subspace.register.cuda.use_cuda = False
        defaults.subspace.register.cuda.TPB = 256

        

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.subspace
        #assert config.subspace.network != None
        if config.subspace.get('register') and config.subspace.register.get('cuda'):
            assert all((isinstance(x, int) or isinstance(x, str) and x.isnumeric() ) for x in config.subspace.register.cuda.get('dev_id', []))

            if config.subspace.register.cuda.get('use_cuda', bittensor.defaults.subspace.register.cuda.use_cuda):
                try:
                    import cubit
                except ImportError:
                    raise ImportError('CUDA registration is enabled but cubit is not installed. Please install cubit.')

                if not is_cuda_available():
                    raise RuntimeError('CUDA registration is enabled but no CUDA devices are detected.')


    @staticmethod
    def determine_chain_endpoint(network: str):
        if network == "nakamoto":
            # Main network.
            return bittensor.__nakamoto_entrypoint__
        elif network == "finney": 
            # Kiru Finney stagin network.
            return bittensor.__finney_entrypoint__
        elif network == "nobunaga": 
            # Staging network.
            return bittensor.__nobunaga_entrypoint__
        elif network == "bellagene":
            # Parachain test net
            return bittensor.__bellagene_entrypoint__
        elif network == "local":
            # Local chain.
            return bittensor.__local_entrypoint__
        elif network == 'mock':
            return bittensor.__mock_entrypoint__
        else:
            return bittensor.__local_entrypoint__
