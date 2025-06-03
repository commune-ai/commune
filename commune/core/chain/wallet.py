
import requests
import json
import os
import queue
import re
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Mapping, TypeVar, cast, List, Dict, Optional
from collections import defaultdict
from .substrate.storage import StorageKey
from .substrate.key import  Keypair# type: ignore
from .substrate.base import ExtrinsicReceipt, SubstrateInterface
from .substrate.types import (ChainTransactionError,
                    NetworkQueryError, 
                    SubnetParamsMaps, 
                    SubnetParamsWithEmission,
                    BurnConfiguration, 
                    GovernanceConfiguration,
                    Ss58Address,  
                    NetworkParams, 
                    SubnetParams, 
                    Chunk)
from typing import Any, Callable, Optional, Union, Mapping
import commune as c

U16_MAX = 2**16 - 1
MAX_REQUEST_SIZE = 9_000_000
IPFS_REGEX = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")
T1 = TypeVar("T1")
T2 = TypeVar("T2")



    def transfer(
        self,
        key: Keypair = None,
        amount: int = None,
        dest: Ss58Address = None,
        safety: bool = True,
        multisig: Optional[str] = None
    ) -> ExtrinsicReceipt:
        """
        Transfers a specified amount of tokens from the signer's account to the
        specified account.

        Args:
            key: The keypair associated with the sender's account.
            amount: The amount to transfer, in nanotokens.
            dest: The SS58 address of the recipient.

        Returns:
            A receipt of the transaction.

        Raises:
            InsufficientBalanceError: If the sender's account does not have
              enough balance.
            ChainTransactionError: If the transaction fails.
        """
        if self.is_float(dest):
            dest = amount
            amount = float(str(dest).replace(',', ''))
        if key == None:
            key = input('Enter key: ')
        key = self.get_key(key)
        if dest == None:
            dest = input('Enter destination address: ')
        dest = self.get_key_address(dest)
        if amount == None:
            amount = input('Enter amount: ')
        amount = float(str(amount).replace(',', ''))

        params = {"dest": dest, "value":int(self.to_nanos(amount))}
        if safety:
            address2key = c.address2key()
            from_name = address2key.get(key.ss58_address, key.ss58_address)
            to_name = address2key.get(dest, dest)
            c.print(f'Transfer({from_name} --({params["value"]/(10**9)}c)--> {to_name})')
            if input(f'Are you sure you want to transfer? (y/n): ') != 'y':
                return False
        return self.call( module="Balances", fn="transfer_keep_alive", params=params, key=key, multisig=multisig)

    def transfer_multiple(
        self,
        key: Keypair,
        destinations: list[Ss58Address],
        amounts: list[int],
    ) -> ExtrinsicReceipt:
        """
        Transfers multiple tokens to multiple addresses at once
        Args:
            key: The keypair associated with the sender's account.
            destinations: A list of SS58 addresses of the recipients.
            amounts: Amount to transfer to each recipient, in nanotokens.
        Returns:
            A receipt of the transaction.
        Raises:
            InsufficientBalanceError: If the sender's account does not have
              enough balance for all transfers.
            ChainTransactionError: If the transaction fails.
        """

        assert len(destinations) == len(amounts)

        # extract existential deposit from amounts
        amounts = [self.to_nanos(a)  for a in amounts]

        params = {
            "destinations": destinations,
            "amounts": amounts,
        }

        return self.call(module="SubspaceModule", fn="transfer_multiple", params=params, key=key )

    def wallets(self,  max_age=None, update=False, mode='df'):
        """
        an overview of your wallets
        """
        my_stake = self.my_stake(update=update, max_age=max_age)
        my_balance = self.my_balance(update=update, max_age=max_age)
        key2address = c.key2address()
        wallets = []
        wallet_names = set(list(my_stake) + list(my_balance))
        for k in wallet_names:
            if not k in key2address:
                continue
            address = key2address[k]
            balance = my_balance.get(k, 0)
            stake = my_stake.get(k, 0)
            total = balance + stake
            wallets.append({'name': k , 'address': address, 'balance': balance, 'stake': stake, 'total': total})

        # add total balance to each wallet
        wallets = c.df(wallets)
        wallets = wallets.sort_values(by='total', ascending=False)
        wallets = wallets.reset_index(drop=True)
        wallets = wallets.to_dict(orient='records')
        wallets.append({'name': '--', 'address': 'total', 'balance': sum([w['balance'] for w in wallets]), 'stake': sum([w['stake'] for w in wallets]), 'total': sum([w['total'] for w in wallets])})
        if mode == 'df':
            wallets = c.df(wallets)
        elif mode == 'list':
            wallets = wallets
        else:
            raise ValueError(f'Invalid mode {mode}. Use "df" or "list".')
        return wallets
     

    def my_tokens(self, min_value=0):
        my_stake = self.my_stake()
        my_balance = self.my_balance()
        my_tokens =  {k:my_stake.get(k,0) + my_balance.get(k,0) for k in set(my_stake)}
        return dict(sorted({k:v for k,v in my_tokens.items() if v > min_value}.items(), key=lambda x: x[1], reverse=True))
   
    def my_total(self):
        return sum(self.my_tokens().values())

    def update_module(
        self,
        key: str,
        name: str=None,
        url: str = None,
        metadata: str = None,
        delegation_fee: int = None,
        validator_weight_fee = None,
        subnet = 2,
        min_balance = 10,
        public = False,

    ) -> ExtrinsicReceipt:
        assert isinstance(key, str) or name != None
        name = name or key
        key = self.get_key(key)
        balance = self.balance(key.ss58_address)
        if balance < min_balance:
            raise ValueError(f'Key {key.ss58_address} has insufficient balance {balance} < {min_balance}')
        subnet = self.get_subnet(subnet)
        if url == None:
            url = c.namespace().get(name, '0.0.0.0:8888')
        url = url if public else ('0.0.0.0:' + url.split(':')[-1])
        module = self.module(key.ss58_address, subnet=subnet)
        validator_weight_fee = validator_weight_fee or module.get('validator_weight_fee', 10)
        delegation_fee = delegation_fee or module.get('stake_delegation_fee', 10)
        params = {
            "name": name,
            "address": url,
            "stake_delegation_fee": delegation_fee,
            "metadata": metadata,
            'validator_weight_fee': validator_weight_fee,
            'netuid': subnet,
        }
        return self.call("update_module", params=params, key=key) 
    
    def update_vali(
        self,
        key: str,
        name: str=None,
        url: str = None,
        metadata: str = None,
        delegation_fee: int = None,
        validator_weight_fee = None,
        subnet = 0,
        min_balance = 10,
        public = False,

    ) -> ExtrinsicReceipt:

        return self.update_module(
            key=key,
            name=name,
            url=url,
            metadata=metadata,
            delegation_fee=delegation_fee,
            validator_weight_fee=validator_weight_fee,
            subnet=subnet,
            min_balance=min_balance,
            public=public
        )
    

    updatemod = upmod = update_module

    def reg(self, name='compare', metadata=None, url='0.0.0.0:8888', module_key=None, key=None, subnet=2, net=None):
        return self.register(name=name, metadata=metadata, url=url, module_key=module_key, key=key, subnet=subnet, net=net)

    def register(
        self,
        name: str,
        url: str = 'NA',
        key : Optional[str] = None , 
        metadata: Optional[str] = None, code : Optional[str] = None, # either code or metadata
        subnet: Optional[str] = 2,
        net = None,
        wait_for_finalization = False,
        public = False,
        stake = 0,
        safety = False,
        payer: Optional[Union[str, Keypair]] = None,
        **kwargs
    ) -> ExtrinsicReceipt:
        """
        Registers a new module in the network.

        Args:
            key: The keypair used for registering the module.
            name: The name of the module.
            url: The url of the module. 
            key_address : The ss58_address of the module
            subnet: The network subnet to register the module in.
                If None, a default value is used.
        """
        name = name or key
        key =  c.get_key(key or name)
        if url == None:
            self.get_module_url(name, public=public)

        params = {
            "network_name": self.get_subnet_name(net or subnet),
            "address":  url,
            "name": name,
            "module_key": c.get_key(key or name).ss58_address,
            "metadata": metadata or code or 'NA',
        }

        return  self.call("register", params=params, key=payer or key, wait_for_finalization=wait_for_finalization, safety=safety)

    def deregister(self, key: Keypair, subnet: int=0) -> ExtrinsicReceipt:
        """
        Deregisters a module from the network.

        Args:
            key: The keypair associated with the module's account.
            subnet: The network identifier.

        Returns:
            A receipt of the module deregistration transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """
        subnet = self.get_subnet(subnet)
        
        params = {"netuid": subnet}

        response = self.call("deregister", params=params, key=key)

        return response
    
    def dereg(self, key: Keypair, subnet: int=0):
        return self.deregister(key=key, subnet=subnet)


    def dereg_many(self, *key: Keypair, subnet: int = 0):
        futures = [c.submit(self.deregister, dict(key=k, subnet=subnet)) for k in key ]
        results = []
        for f in c.as_completed(futures):
            results += [f.result()]
        return results

    def register_subnet(self, name: str, metadata: str = None,  key: Keypair=None) -> ExtrinsicReceipt:
        """
        Registers a new subnet in the network.

        Args:
            key (Keypair): The keypair used for registering the subnet.
            name (str): The name of the subnet to be registered.
            metadata (str, optional): Additional metadata for the subnet. Defaults to None.

        Returns:
            ExtrinsicReceipt: A receipt of the subnet registration transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """
        key = c.get_key(key or name)

        params = {
            "name": name,
            "metadata": metadata,
        }
        response = self.call("register_subnet", params=params, key=key)
        return response
    
    regnet = register_subnet

    def vote(
        self,
        key: Keypair,
        modules: list[int] = None, # uids, keys or names
        weights: list[int] = None, # any value, relative is takens
        subnet = 0,
    ) -> ExtrinsicReceipt:
        """
        Casts votes on a list of module UIDs with corresponding weights.

        The length of the UIDs list and the weights list should be the same.
        Each weight corresponds to the UID at the same index.

        Args:
            key: The keypair used for signing the vote transaction.
            uids: A list of module UIDs to vote on.
            weights: A list of weights corresponding to each UID.
            subnet: The network identifier.

        Returns:
            A receipt of the voting transaction.

        Raises:
            InvalidParameterError: If the lengths of UIDs and weights lists
                do not match.
            ChainTransactionError: If the transaction fails.
        """
        if modules == None:
            modules_str = input('Enter modules (space separated): ')
            modules = [int(m.strip()) for m in modules_str.split(' ')]
        if weights == None:
            weights_str = input('Enter weights (space separated): ')
            weights = [int(w.strip()) for w in weights_str.split(' ')]

        subnet = self.get_subnet(subnet)
        assert len(modules) == len(weights)
        key2uid = self.key2uid(subnet)
        uids = [key2uid.get(m, m) for m in modules]
        params = {"uids": uids,"weights": weights,"netuid": subnet}
        response = self.call("set_weights", params=params, key=key, module="SubnetEmissionModule")
        return response
    
    def set_weights(
        self,
        modules: list[int], # uids, keys or names
        weights: list[int], # any value, relative is takens
        key: Keypair,
        subnet = 0,
    ) -> ExtrinsicReceipt:
        return self.vote(modules, weights, key, subnet=subnet)

    def update_subnet(
        self,
        subnet,
        params: SubnetParams = None,
        **extra_params
    ) -> ExtrinsicReceipt:
        """
        Update a subnet's configuration.

        It requires the founder key for authorization.

        Args:
            key: The founder keypair of the subnet.
            params: The new parameters for the subnet.
            subnet: The network identifier.

        Returns:
            A receipt of the subnet update transaction.

        Raises:
            AuthorizationError: If the key is not authorized.
            ChainTransactionError: If the transaction fails.
        """
        original_params = self.subnet_params(subnet)
        subnet = self.get_subnet(subnet)

        # ensure founder key
        address2key = c.address2key()
        assert original_params['founder'] in address2key, f'No key found for {original_params["founder"]}'
        key = c.get_key(address2key[original_params['founder']])
        print('Updating subnet', subnet, 'with', params, key)

        params = {**(params or {}), **extra_params} 
        if 'founder' in params:
            params['founder'] = self.get_key_address(params['founder'])
        params = {**original_params, **params} # update original params with params
        assert any([k in original_params for k in params.keys()]), f'Invalid params {params.keys()}'
        params["netuid"] = subnet
        params['vote_mode'] = params.pop('governance_configuration')['vote_mode']
        params["metadata"] = params.pop("metadata", None)
        params["use_weights_encryption"] = params.pop("use_weights_encryption", False)
        params["copier_margin"] = params.pop("copier_margin", 0)
        params["max_encryption_period"] = params.pop("max_encryption_period", 360)
        return self.call(fn="update_subnet",params=params,key=key)



    # def topup_miners(self, subnet):
    
    def transfer_stake(
        self,
        key: Keypair,
        from_module_key: Ss58Address,
        dest_module_address: Ss58Address,
        amount: int,
    ) -> ExtrinsicReceipt:
        """
        Realocate staked tokens from one staked module to another module.

        Args:
            key: The keypair associated with the account that is delegating the tokens.
            amount: The amount of staked tokens to transfer, in nanotokens.
            from_module_key: The SS58 address of the module you want to transfer from (currently delegated by the key).
            dest_module_address: The SS58 address of the destination (newly delegated key).

        Returns:
            A receipt of the stake transfer transaction.

        Raises:
            InsufficientStakeError: If the source module key does not have
            enough staked tokens. ChainTransactionError: If the transaction
            fails.
        """

        amount = amount - self.existential_deposit()

        params = {
            "amount": self.format_amount(amount, fmt='nano'),
            "module_key": from_module_key,
            "new_module_key": dest_module_address,
        }

        response = self.call("transfer_stake", key=key, params=params)

        return response
    
    stake_transfer = transfer_stake 

    def multiunstake(
        self,
        key: Keypair,
        keys: list[Ss58Address],
        amounts: list[int],
    ) -> ExtrinsicReceipt:
        """
        Unstakes tokens from multiple module keys.

        And the lists `keys` and `amounts` must be of the same length. Each
        amount corresponds to the module key at the same index.

        Args:
            key: The keypair associated with the unstaker's account.
            keys: A list of SS58 addresses of the module keys to unstake from.
            amounts: A list of amounts to unstake from each module key,
              in nanotokens.

        Returns:
            A receipt of the multi-unstaking transaction.

        Raises:
            MismatchedLengthError: If the lengths of keys and amounts lists do
            not match. InsufficientStakeError: If any of the module keys do not
            have enough staked tokens. ChainTransactionError: If the transaction
            fails.
        """

        assert len(keys) == len(amounts)

        params = {"module_keys": keys, "amounts": amounts}

        response = self.call("remove_stake_multiple", params=params, key=key)

        return response

    def multistake(
        self,
        key: Keypair,
        keys: list[Ss58Address],
        amounts: list[int],
    ) -> ExtrinsicReceipt:
        """
        Stakes tokens to multiple module keys.

        The lengths of the `keys` and `amounts` lists must be the same. Each
        amount corresponds to the module key at the same index.

        Args:
            key: The keypair associated with the staker's account.
            keys: A list of SS58 addresses of the module keys to stake to.
            amounts: A list of amounts to stake to each module key,
                in nanotokens.

        Returns:
            A receipt of the multi-staking transaction.

        Raises:
            MismatchedLengthError: If the lengths of keys and amounts lists
                do not match.
            ChainTransactionError: If the transaction fails.
        """

        assert len(keys) == len(amounts)

        params = {
            "module_keys": keys,
            "amounts": amounts,
        }

        response = self.call("add_stake_multiple", params=params, key=key)

        return response

    def add_profit_shares(
        self,
        key: Keypair,
        keys: list[Ss58Address],
        shares: list[int],
    ) -> ExtrinsicReceipt:
        """
        Allocates profit shares to multiple keys.

        The lists `keys` and `shares` must be of the same length,
        with each share amount corresponding to the key at the same index.

        Args:
            key: The keypair associated with the account
                distributing the shares.
            keys: A list of SS58 addresses to allocate shares to.
            shares: A list of share amounts to allocate to each key,
                in nanotokens.

        Returns:
            A receipt of the profit sharing transaction.

        Raises:
            MismatchedLengthError: If the lengths of keys and shares
                lists do not match.
            ChainTransactionError: If the transaction fails.
        """

        assert len(keys) == len(shares)

        params = {"keys": keys, "shares": shares}

        response = self.call("add_profit_shares", params=params, key=key)

        return response

    def add_subnet_proposal(
        self, key: Keypair,
        params: dict[str, Any],
        ipfs: str,
        subnet: int = 0
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for creating or modifying a subnet within the
        network.

        The proposal includes various parameters like the name, founder, share
        allocations, and other subnet-specific settings.

        Args:
            key: The keypair used for signing the proposal transaction.
            params: The parameters for the subnet proposal.
            subnet: The network identifier.

        Returns:
            A receipt of the subnet proposal transaction.

        Raises:
            InvalidParameterError: If the provided subnet
                parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """
        subnet = self.get_subnet(subnet)
        general_params = dict(params)
        general_params["netuid"] = subnet
        general_params["data"] = ipfs
        if "metadata" not in general_params:
            general_params["metadata"] = None

        # general_params["burn_config"] = json.dumps(general_params["burn_config"])
        response = self.call(
            fn="add_params_proposal",
            params=general_params,
            key=key,
            module="GovernanceModule",
        )

        return response

    def add_custom_proposal(
        self,
        key: Keypair,
        cid: str,
    ) -> ExtrinsicReceipt:

        params = {"data": cid}

        response = self.call(
            fn="add_global_custom_proposal",
            params=params,
            key=key,
            module="GovernanceModule",
        )
        return response

    def add_custom_subnet_proposal(
        self,
        key: Keypair,
        cid: str,
        subnet: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for creating or modifying a custom subnet within the
        network.

        The proposal includes various parameters like the name, founder, share
        allocations, and other subnet-specific settings.c

        Args:
            key: The keypair used for signing the proposal transaction.
            params: The parameters for the subnet proposal.
            subnet: The network identifier.

        Returns:
            A receipt of the subnet proposal transaction.
        """

        subnet = self.get_subnet(subnet)
        params = {
            "data": cid,
            "netuid": subnet,
        }

        response = self.call(
            fn="add_subnet_custom_proposal",
            params=params,
            key=key,
            module="GovernanceModule",
        )

        return response

    def add_global_proposal(
        self,
        key: Keypair,
        params: NetworkParams,
        cid: str,
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for altering the global network parameters.

        Allows for the submission of a proposal to
        change various global parameters
        of the network, such as emission rates, rate limits, and voting
        thresholds. It is used to
        suggest changes that affect the entire network's operation.

        Args:
            key: The keypair used for signing the proposal transaction.
            params: A dictionary containing global network parameters
                    like maximum allowed subnets, modules,
                    transaction rate limits, and others.

        Returns:
            A receipt of the global proposal transaction.

        Raises:
            InvalidParameterError: If the provided network
                parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """
        general_params = cast(dict[str, Any], params)
        cid = cid or ""
        general_params["data"] = cid

        response = self.call(
            fn="add_global_params_proposal",
            params=general_params,
            key=key,
            module="GovernanceModule",
        )

        return response

    def vote_on_proposal(
        self,
        key: Keypair,
        proposal_id: int,
        agree: bool,
    ) -> ExtrinsicReceipt:
        """
        Casts a vote on a specified proposal within the network.

        Args:
            key: The keypair used for signing the vote transaction.
            proposal_id: The unique identifier of the proposal to vote on.

        Returns:
            A receipt of the voting transaction in nanotokens.

        Raises:
            InvalidProposalIDError: If the provided proposal ID does not
                exist or is invalid.
            ChainTransactionError: If the transaction fails.
        """

        params = {"proposal_id": proposal_id, "agree": agree}

        response = self.call(
            "vote_proposal",
            key=key,
            params=params,
            module="GovernanceModule",
        )

        return response

    def unvote_on_proposal(
        self,
        key: Keypair,
        proposal_id: int,
    ) -> ExtrinsicReceipt:
        """
        Retracts a previously cast vote on a specified proposal.

        Args:
            key: The keypair used for signing the unvote transaction.
            proposal_id: The unique identifier of the proposal to withdraw the
                vote from.

        Returns:
            A receipt of the unvoting transaction in nanotokens.

        Raises:
            InvalidProposalIDError: If the provided proposal ID does not
                exist or is invalid.
            ChainTransactionError: If the transaction fails to be processed, or
                if there was no prior vote to retract.
        """

        params = {"proposal_id": proposal_id}

        response = self.call(
            "remove_vote_proposal",
            key=key,
            params=params,
            module="GovernanceModule",
        )

        return response

    def enable_vote_power_delegation(self, key: Keypair) -> ExtrinsicReceipt:
        """
        Enables vote power delegation for the signer's account.

        Args:
            key: The keypair used for signing the delegation transaction.

        Returns:
            A receipt of the vote power delegation transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        response = self.call(
            "enable_vote_power_delegation",
            params={},
            key=key,
            module="GovernanceModule",
        )

        return response

    def disable_vote_power_delegation(self, key: Keypair) -> ExtrinsicReceipt:
        """
        Disables vote power delegation for the signer's account.

        Args:
            key: The keypair used for signing the delegation transaction.

        Returns:
            A receipt of the vote power delegation transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        response = self.call(
            "disable_vote_power_delegation",
            params={},
            key=key,
            module="GovernanceModule",
        )

        return response

    def add_dao_application(
        self, key: Keypair, application_key: Ss58Address, data: str
    ) -> ExtrinsicReceipt:
        """
        Submits a new application to the general subnet DAO.

        Args:
            key: The keypair used for signing the application transaction.
            application_key: The SS58 address of the application key.
            data: The data associated with the application.

        Returns:
            A receipt of the application transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        params = {"application_key": application_key, "data": data}

        response = self.call(
            "add_dao_application", module="GovernanceModule", key=key,
            params=params
        )

        return response



    sudo_multisig_data = {'keys': [
            '5H47pSknyzk4NM5LyE6Z3YiRKb3JjhYbea2pAUdocb95HrQL', # sudo
            '5FZsiAJS5WMzsrisfLWosyzaCEQ141rncjv55VFLHcUER99c', # krishna
            '5DPSqGAAy5ze1JGuSJb68fFPKbDmXhfMqoNSHLFnJgUNTPaU', # sentinal
            '5CMNEDouxNdMUEM6NE9HRYaJwCSBarwr765jeLdHvWEE15NH', # liaonu
            '5CwXN5zQFQNoFRaycsiE29ibDDp2mXwnof228y76fMbs2jHd', # huck
        ],
    'threshold': 3
    }
    sudo_multisig_threshold = 3

    def sudo_multisig(self) -> List[str]:
        return self.get_multisig(sudo_multisig_data)

    def sudo_transfer(self,
        key: Keypair,
        dest: Ss58Address,
        amount: int,
        data: str = None,
    ):
        """
        Transfer funds to a specific address using the sudo key.
        """

        key = self.get_key(key)
        return self.call_multisig(
            key=key,
            multisig=self.multisig('sudo'),
            dest=dest,
            amount=amount,
            data=data,
        )

    def multisig(self, keys=None, threshold=3):
        if isinstance(keys, str) or isinstance(keys, dict):
            multisig_data = self.get_multisig_data(keys)
            keys = multisig_data['keys']
            threshold = multisig_data['threshold']
    
        keys = keys or self.sudo_multisig_data['keys']
        keys = [self.get_key_address(k) for k in keys]
        with self.get_conn(init=True) as substrate:
        
            multisig_acc = substrate.generate_multisig_account(  # type: ignore
                keys, threshold
            )

        return multisig_acc

    def compose_call_multisig(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair,
        signatories: list[Ss58Address],
        threshold: int,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = None,
        sudo: bool = False,
        era: dict[str, int] = None,
    ) -> ExtrinsicReceipt:
        """
        Composes and submits a multisignature call to the network node.

        This method allows the composition and submission of a call that
        requires multiple signatures for execution, known as a multisignature
        call. It supports specifying signatories, a threshold of signatures for
        the call's execution, and an optional era for the call's mortality. The
        call can be a standard extrinsic, a sudo extrinsic for elevated
        permissions, or a multisig extrinsic if multiple signatures are
        required. Optionally, the method can wait for the call's inclusion in a
        block and/or its finalization. Make sure to pass all keys,
        that are part of the multisignature.

        Args:
            fn: The function name to call on the network. params: A dictionary
            of parameters for the call. key: The keypair for signing the
            extrinsic. signatories: List of SS58 addresses of the signatories.
            Include ALL KEYS that are part of the multisig. threshold: The
            minimum number of signatories required to execute the extrinsic.
            module: The module containing the function to call.
            wait_for_inclusion: Whether to wait for the call's inclusion in a
            block. wait_for_finalization: Whether to wait for the transaction's
            finalization. sudo: Execute the call as a sudo (superuser)
            operation. era: Specifies the call's mortality in terms of blocks in
            the format
                {'period': amount_blocks}. If omitted, the extrinsic is
                immortal.

        Returns:
            The receipt of the submitted extrinsic if `wait_for_inclusion` is
            True. Otherwise, returns a string identifier of the extrinsic.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        # getting the call ready
        with self.get_conn() as substrate:
            if wait_for_finalization is None:
                wait_for_finalization = self.wait_for_finalization

            substrate.reload_type_registry()


            # prepares the `GenericCall` object
            call = substrate.compose_call(  # type: ignore
                call_module=module, call_function=fn, call_params=params
            )
            if sudo:
                call = substrate.compose_call(  # type: ignore
                    call_module="Sudo",
                    call_function="sudo",
                    call_params={
                        "call": call.value,  # type: ignore
                    },
                )
            multisig_acc = substrate.generate_multisig_account(  # type: ignore
                signatories, threshold
            )

            # send the multisig extrinsic
            extrinsic = substrate.create_multisig_extrinsic(  # type: ignore
                call=call,  # type: ignore
                keypair=key,
                multisig_account=multisig_acc,  # type: ignore
                era=era,  # type: ignore
            )  # type: ignore

            response = substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if wait_for_inclusion:
            if not response.is_success:
                raise ChainTransactionError(
                    response.error_message, response  # type: ignore
                )

        return response



    def call(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        multisig = None,
        sudo: bool = False,
        tip = 0,
        safety: bool = False,
        nonce=None,
    ) -> ExtrinsicReceipt:
        """
        Composes and submits a call to the network node.

        Composes and signs a call with the provided keypair, and submits it to
        the network. The call can be a standard extrinsic or a sudo extrinsic if
        elevated permissions are required. The method can optionally wait for
        the call's inclusion in a block and/or its finalization.

        Args:
            fn: The function name to call on the network.
            params: A dictionary of parameters for the call.
            key: The keypair for signing the extrinsic.
            module: The module containing the function.
            wait_for_inclusion: Wait for the call's inclusion in a block.
            wait_for_finalization: Wait for the transaction's finalization.
            sudo: Execute the call as a sudo (superuser) operation.

        Returns:
            The receipt of the submitted extrinsic, if
              `wait_for_inclusion` is True. Otherwise, returns a string
              identifier of the extrinsic.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        key = self.get_key(key)
        
        info_call = {
            "module": module,
            "fn": fn,
            "params": params,
            "key": key.ss58_address,
            "network": self.network,
        }

        key_name = self.get_key_name(key.ss58_address)
        c.print(f"Call(\nmodule={info_call['module']} \nfn={info_call['fn']} \nkey={key.ss58_address} ({key_name}) \nparams={info_call['params']}) \n)", color='cyan')

        if safety:
            if input('Are you sure you want to send this transaction? (y/n) --> ') != 'y':
                raise Exception('Transaction cancelled by user')

        with self.get_conn() as substrate:

    

            call = substrate.compose_call(  # type: ignore
                call_module=module, 
                call_function=fn, 
                call_params=params
            )
            if sudo:
                call = substrate.compose_call(call_module="Sudo", call_function="sudo", call_params={"call": call.value})

            if multisig != None:
                multisig = self.get_multisig(multisig)
                # send the multisig extrinsic
                extrinsic = substrate.create_multisig_extrinsic(  
                                                                call=call,   
                                                                keypair=key, 
                                                                multisig_account=multisig, 
                                                                era=None,  # type: ignore
                )
            else:
                extrinsic = substrate.create_signed_extrinsic(call=call, keypair=key, nonce=nonce, tip=tip)  # type: ignore


            response = substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        if wait_for_inclusion:
            if not response.is_success:
                raise ChainTransactionError(
                    response.error_message, response  # type: ignore
                )
            else:
                return {'success': True, 'tx_hash': response.extrinsic_hash, 'module': module, 'fn':fn, 'url': self.url,  'network': self.network, 'key':key.ss58_address }
            
        if wait_for_finalization:
            response.process_events()
            if response.is_success:
                response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'module': module, 'fn':fn, 'url': self.url,  'network': self.network, 'key':key.ss58_address }
            else:
                response =  {'success': False, 'error': response.error_message, 'module': module, 'fn':fn, 'url': self.url,  'network': self.network, 'key':key.ss58_address }
        return response

    def my_valis(self, subnet=0, min_stake=0, features=['name', 'key','weights', 'stake']):
        return c.df(self.my_modules(subnet, features=features ))

    def my_keys(self, subnet=0):
        return [m['key'] for m in self.my_modules(subnet)]

    def name2key(self, subnet=0, **kwargs) -> dict[str, str]:
        """
        Returns a mapping of names to keys for the specified subnet.
        """
        modules = self.modules(subnet=subnet, features=['name', 'key'], **kwargs)
        return {m['name']: m['key'] for m in modules}

    def call_multisig(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair,
        multisig = None,
        signatories: list[Ss58Address]=None,
        threshold: int = None,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = None,
        sudo: bool = False,
        era: dict[str, int] = None,
    ) -> ExtrinsicReceipt:
        """
        Composes and submits a multisignature call to the network node.

        This method allows the composition and submission of a call that
        requires multiple signatures for execution, known as a multisignature
        call. It supports specifying signatories, a threshold of signatures for
        the call's execution, and an optional era for the call's mortality. The
        call can be a standard extrinsic, a sudo extrinsic for elevated
        permissions, or a multisig extrinsic if multiple signatures are
        required. Optionally, the method can wait for the call's inclusion in a
        block and/or its finalization. Make sure to pass all keys,
        that are part of the multisignature.

        Args:
            fn: The function name to call on the network. params: A dictionary
            of parameters for the call. key: The keypair for signing the
            extrinsic. signatories: List of SS58 addresses of the signatories.
            Include ALL KEYS that are part of the multisig. threshold: The
            minimum number of signatories required to execute the extrinsic.
            module: The module containing the function to call.
            wait_for_inclusion: Whether to wait for the call's inclusion in a
            block. wait_for_finalization: Whether to wait for the transaction's
            finalization. sudo: Execute the call as a sudo (superuser)
            operation. era: Specifies the call's mortality in terms of blocks in
            the format
                {'period': amount_blocks}. If omitted, the extrinsic is
                immortal.

        Returns:
            The receipt of the submitted extrinsic if `wait_for_inclusion` is
            True. Otherwise, returns a string identifier of the extrinsic.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        # getting the call ready
        with self.get_conn() as substrate:
            # prepares the `GenericCall` object
            
            call = substrate.call(  # type: ignore
                call_module=module, call_function=fn, call_params=params
            )
            if sudo:
                call = substrate.call(  # type: ignore
                    call_module="Sudo",
                    call_function="sudo",
                    call_params={
                        "call": call.value,  # type: ignore
                    },
                )

            # create the multisig account
            if multisig != None :
                # send the multisig extrinsic
                extrinsic = substrate.create_multisig_extrinsic(  # type: ignore
                    call=call,  # type: ignore
                    keypair=key,
                    multisig_account=multisig,  # type: ignore
                    era=era,  # type: ignore
                )  # type: ignore

            response = substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if wait_for_inclusion:
            if not response.is_success:
                raise ChainTransactionError(
                    response.error_message, response  # type: ignore
                )

        return response

    def get_multisig_path(self, multisig):
        return self.get_path(f'multisig/{multisig}')

    def get_multisig_data(self, multisig):
        if multisig == 'sudo':
            return self.sudo_multisig_data
        if isinstance(multisig, str):
            multisig = c.get(self.get_multisig_path(multisig))
            assert isinstance(multisig, dict)
        return multisig

    def get_multisig(self, multisig):
        if isinstance(multisig, str):
            multisig = self.multisigs().get(multisig)
        if isinstance(multisig, dict):
            return self.multisig(multisig.get('keys'),  multisig.get('threshold'))

        return multisig

    def check_multisig(self, multisig):
        if isinstance(multisig, str):
            multisig = self.get_multisig(multisig)
        if isinstance(multisig, dict):
            keys = multisig.get('signatories', multisig.get('keys'))
            threshold = multisig.get('threshold')
            assert len(keys) >= threshold
            assert len(keys) > 0
            return True
        return False

    def add_multisig(self, name='multisig',  keys=None, threshold=None):
        assert not self.multisig_exists(name)
        if keys == None:
            keys = input('Enter keys (comma separated): ')
            keys = [ k.strip() for k in keys.split(',') ]
        if threshold == None:
            threshold = input('Enter threshold: ')
            threshold = int(threshold)
            assert threshold <= len(keys)

        multisig = {
            'keys': keys,
            'threshold': threshold,
        }
        assert self.check_multisig(multisig)
        path = self.get_multisig_path(name)
        return c.put(path, multisig)

    put_multiisg = add_multisig
    def multisig_exists(self, multisig):
        if isinstance(multisig, str):
            multisig = self.get_multisig(multisig)
        if isinstance(multisig, dict):
            self.check_multisig(multisig)
        return False

    def multisigs(self):
        path = self.get_path(f'multisig')
        paths = c.ls(path)
        multisigs = {}
        for p in paths:
            multisig = c.get(p, None)
            if multisig != None:
                multisigs[p.split('/')[-1].split('.')[-2]] = self.get_multisig_data(multisig)

        # add sudo multisig
        multisigs['sudo'] = self.sudo_multisig_data

        for k, v in multisigs.items():
            if isinstance(v, dict):
                multisig_address = self.multisig(v).ss58_address
                multisigs[k]['address'] = multisig_address
        return multisigs

    mss = multisigs
    
    def transfer(
        self,
        key: Keypair = None,
        amount: int = None,
        dest: Ss58Address = None,
        safety: bool = True,
        multisig: Optional[str] = None
    ) -> ExtrinsicReceipt:
        """
        Transfers a specified amount of tokens from the signer's account to the
        specified account.

        Args:
            key: The keypair associated with the sender's account.
            amount: The amount to transfer, in nanotokens.
            dest: The SS58 address of the recipient.

        Returns:
            A receipt of the transaction.

        Raises:
            InsufficientBalanceError: If the sender's account does not have
              enough balance.
            ChainTransactionError: If the transaction fails.
        """
        if self.is_float(dest):
            dest = amount
            amount = float(str(dest).replace(',', ''))
        if key == None:
            key = input('Enter key: ')
        key = self.get_key(key)
        if dest == None:
            dest = input('Enter destination address: ')
        dest = self.get_key_address(dest)
        if amount == None:
            amount = input('Enter amount: ')
        amount = float(str(amount).replace(',', ''))

        params = {"dest": dest, "value":int(self.to_nanos(amount))}
        if safety:
            address2key = c.address2key()
            from_name = address2key.get(key.ss58_address, key.ss58_address)
            to_name = address2key.get(dest, dest)
            c.print(f'Transfer({from_name} --({params["value"]/(10**9)}c)--> {to_name})')
            if input(f'Are you sure you want to transfer? (y/n): ') != 'y':
                return False
        return self.call( module="Balances", fn="transfer_keep_alive", params=params, key=key, multisig=multisig)
    
    def send(
        self, key, amount, dest, multisig=None, safety=True
    ) -> ExtrinsicReceipt:
        return self.transfer(key=key, amount=amount, dest=dest)



    def send_my_modules( self,  amount=1, subnet=0, key='module'):
        destinations = self.my_keys(subnet)
        amounts = [amount] * len(destinations)
        return self.transfer_multiple(key=key, destinations=destinations,amounts=amounts)

    def my_staketo(self, update=False, max_age=None):
        staketo = self.staketo(update=update, max_age=max_age)
        key2address = c.key2address()
        my_stakefrom = {}
        for key, address in key2address.items():
            if address in staketo:
                my_stakefrom[key] = staketo[address]
        return my_stakefrom

    def my_stake(self, update=False, max_age=None):
        my_stake =  {k:sum(v.values()) for k,v in self.my_staketo(update=update, max_age=max_age).items()}
        my_stake =  dict(sorted(my_stake.items(), key=lambda x: x[1], reverse=True))
        return {k:v for k,v in my_stake.items() if v > 0}

    def unstake(
        self,
        key: Keypair,
        amount: int = None,
        dest: Ss58Address=None ,
        safety: bool = True,

    ) -> ExtrinsicReceipt:
        """
        Unstakes the specified amount of tokens from a module key address.
        """
        if self.is_float(dest):
            dest = amount
            amount = float(dest)
        if amount == None:
            amount = input('Enter amount to unstake: ')
            amount = float(str(amount).replace(',', ''))
            
        if dest == None:
            staketo = self.staketo(key)
            idx2key_options = {i: k for i, (k, v) in enumerate(staketo.items()) if v > amount}
            if len(idx2key_options) == 1:
                dest = list(idx2key_options.values())[0]
            elif len(idx2key_options) > 1:
                c.print(f'Unstake {amount}c from which module key? {idx2key_options}')
                idx = input(f'')
                dest = idx2key_options[int(idx)]
            else:
                raise ValueError(f'No module key found with enough stake to unstake {amount}')
        params = {"amount":  amount * 10**9, "module_key": self.get_key_address(dest)}
        return self.call(fn="remove_stake", params=params, key=key, safety=safety)


    def stake(
        self,
        key: Keypair,
        amount: int = None,
        dest: Ss58Address=None ,
        safety: bool = True,


    ) -> ExtrinsicReceipt:
        """
        stakes the specified amount of tokens from a module key address.
        """
        if self.is_float(dest):
            dest = amount
            amount = float(dest)
        if amount == None:
            amount = input('Enter amount to unstake: ')
            amount = float(str(amount).replace(',', ''))
        
        if dest == None:
            staketo = self.staketo(key)
            # if there is only one module key, use it
            dest = {i: k for i, (k, v) in enumerate(staketo.items()) if v > amount}
            if len(dest) == 0:
                raise ValueError(f'No module key found with enough stake to unstake {amount}')
            else:
                c.print(f'Unstake {amount}c from which module key? {dest}')
                idx = input(f'')
                dest = dest[int(idx)]
        else:
            name2key = self.name2key()
            dest = name2key.get(dest, dest)
        params = {"amount":  amount * 10**9, "module_key": self.get_key_address(dest)}
        return self.call(fn="add_stake", params=params, key=key, safety=safety)
    
