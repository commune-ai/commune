
from substrateinterface import SubstrateInterface, Keypair, ContractCode, ContractInstance
import shutil
import streamlit as st
# from commune.substrate.account import SubstrateAccount
import os
from typing import *
from glob import glob
import os, sys
import commune



class SubstrateContract(commune.Module):
    
    def __init__(self, keypair:Keypair = None, 
                 substrate:'SubstrateInterface' = None,
                 contracts_dir_path:str = None):
        
        self.set_contract_dir_path(contracts_dir_path)
        self.set_keypair(keypair)
        self.set_substrate(substrate)
        

    def set_contract_dir_path(self, contract_dir_path:str = None):
        self.contracts_dir_path = contract_dir_path if contract_dir_path else self.default_contract_dir()
        return self.contracts_dir_path
   
    @classmethod
    def default_contract_dir(cls, contract:str):
        contracts_dir_path = f'{cls.pwd}/contracts/ink'

        
    def set_substrate(self, substrate=None):
        
        if substrate == None:
            substrate = SubstrateInterface(
                url = "ws://0.0.0.0:9944",
                type_registry_preset='canvas'
            )
        
        commune.print(f'Connected to {substrate.url}','purple')
        
        self.substrate = substrate

    def set_keypair(self, keypair:Keypair = None):
        if keypair == None:
            keypair = Keypair.create_from_uri('//Alice')
        self.keypair = keypair


    @property
    def contract_paths(self):
        return [f for f in glob(self.contracts_dir_path+'/**') if os.path.isdir(f)]
    


    @property
    def contract_file_info(self):
        compiled_contract_paths = []
        contract_file_info = {}
        for f in self.contract_paths:
            contract_info = {}
            target_path = os.path.join(f, 'target')
            contract_name = f.split('/')[-1]

            compiled = False
            if os.path.isdir(target_path):
                build_path_dict = {}
                build_path_dict['metadata'] = f'{target_path}/ink/metadata.json'
                build_path_dict['wasm'] =  f'{target_path}/ink/{contract_name}.wasm'
                build_path_dict['contract'] =  f'{target_path}/ink/{contract_name}.contract'
                compiled =all([os.path.exists(v) and os.path.isfile(v) for v in build_path_dict.values()])
                if compiled:
                    contract_info.update(build_path_dict)
                wasm_path = contract_name

            
            contract_info['path'] = f
            contract_info['compiled'] = compiled

            contract_file_info[contract_name] = contract_info

        return contract_file_info
    @property
    def contract_names(self):
        return [ f.split('/')[-1] for f in self.contract_paths   ]


    default_tag = 'base'
    def deploy(self, 
            contract:str,
            endowment:int=0,
            deployment_salt: str=None,
            gas_limit:int=1000000000000,
            constructor:str="new",
            args:dict=None,
            upload_code:bool=True,
            refresh:bool = False,
            compile:bool=False,
            tag: str = None):
        args = args if args!= None else {}
        '''
        Deploy a contract to the chain
        
        Args:
            contract (str): The name of the contract to deploy
            endowment (int, optional): The amount of tokens to send to the contract. Defaults to 0.
            deployment_salt (str, optional): A salt to use for the deployment. Defaults to None.
            gas_limit (int, optional): The gas limit to use for the deployment. Defaults to 1000000000000.
            constructor (str, optional): The name of the constructor to use. Defaults to "new".
            args: The arguments to pass to the constructor. Defaults to {'total_supply': 100000}.
            upload_code (bool, optional): Whether or not to upload the code to the chain. Defaults to True.
            refresh (bool, optional): Whether or not to refresh the contract. Defaults to True.
            compile (bool, optional): Whether or not to compile the contract. Defaults to False.
            tag (str, optional): A tag to use for the contract. Defaults to None.
        '''
        
        tag = tag if tag else self.default_tag

        # If refresh is false, lets see if the contract exists
        if compile:
            self.compile(contract)
        
        
        
        # if you do not want ot refresh the contract, lets see if it exists
        if refresh == False:
            contract_instance = self.get_contract(contract)
            if contract_instance != None:
                self.contract = contract_instance
                return self.contract

        deployment_salt = deployment_salt if deployment_salt else str(time.time())

        contract_file_info = self.contract_file_info[contract]
        if contract_file_info['compiled'] == False:
            contract_file_info = self.compile(contract=contract)

        code = ContractCode.create_from_contract_files(
                    metadata_file=contract_file_info['metadata'],
                    wasm_file= contract_file_info['wasm'],
                    substrate=self.substrate
                )
        
        
        deploy_params = dict(
            endowment=endowment,
            gas_limit=gas_limit,
            deployment_salt=deployment_salt,
            constructor=constructor,
            args=args,
            upload_code=upload_code
        ) 
        
        self.contract = code.deploy(
            keypair=self.keypair,
            **deploy_params
        )
        
        self.register_contract(contract=self.contract, name=contract, tag=tag)


        return self.contract


    def get_contract_info(self, contract):
        
        contract_info = {'address':contract.contract_address}
        return contract_info
        
    def register_contract(self, contract:str, name: str, tag:str):
        contract_info = self.get_contract_info(contract)
        
        self.put_json(f'deployed_contracts/{name}/{tag}', contract_info )

        return contract_info
    @property
    def contract_address(self):
        return self.contract.contract_address

    @property
    def contracts(self):
        return self.list_contracts()

    def list_contracts(self, compiled_only=False):

        contracts = []
        for contract_name,contract_info in self.contract_file_info.items():
            if compiled_only:
                if contract_info['compiled']:
                    contracts.append(contract_name)
            else:
                contracts.append(contract_name)

        return contracts



    def set_contract(self, contract:Union[str, ContractInstance], deployment_salt:str=None) -> ContractInstance:
        if isinstance(contract, str):
            contract = self.get_contract(contract=contract,deployment_salt=deployment_salt)
        elif isinstance(contract, ContractInstance):
            contract = contract
        self.contract = contract
        return self.contract

    def get_contract(self, contract:str, tag:str=None) -> Union['Contract', 'contract_addresses']:
        '''
        Get the contract from an existing chain
        '''
        tag = tag if tag != None else self.default_tag
        # Check if contract is on chain
        contract_info = self.contract_file_info[contract]
        contract_addresses = self.deployed_contracts.get(contract, None)
        if contract_addresses == None:
            return None

        
        tag = tag if tag else list(contract_addresses.keys())[0]
        contract_address = contract_addresses[tag]['address']
        
        
        contract_substrate_info = self.substrate.query("Contracts", "ContractInfoOf", [contract_address])
        if contract_substrate_info.value:

            commune.print(f'Found contract on chain: {contract_substrate_info.value}', 'green')

            # Create contract instance from deterministic address
            contract = ContractInstance.create_from_address(
                contract_address=contract_address,
                metadata_file=contract_info['metadata'],
                substrate=self.substrate
            )

        else:
            raise NotImplemented


        return contract



    @property
    def deployed_contracts(self):
        deployed_paths = self.glob('deployed_contracts/**')
        deployed_contracts = {}
        for path in deployed_paths:
            name = path.split('/')[-2]
            tag = path.split('/')[-1]
            
            if name not in deployed_contracts:
                deployed_contracts[name] = {}
            if tag not in deployed_contracts[name]:
                deployed_contracts[name][tag] = self.get_json(path)
                
        return deployed_contracts


    @deployed_contracts.setter
    def deployed_contracts(self, deployed_contracts:dict):
        contract_file_info = self.contract_file_info
        assert isinstance(deployed_contracts, dict)

        for k,v in deployed_contracts.items():
            assert k in contract_file_info, f'{k} does not exist as a contract name'

        self.put_json('deployed_contracts', deployed_contracts)
        return deployed_contracts


    
    
    command_dict = {
        "build": 'cargo +nightly contract build'
    }
    def compile(self, contract:str, rebuild:bool=False):
        ''
        assert contract in self.contract_file_info, f'available is {self.contract_names}'
        
        contract_info = self.contract_file_info[contract]
        contract_path = contract_info['path']
        commune.run_command(self.command_dict['build'], cwd=contract_path)

        new_contract_info = self.contract_file_info[contract]
        assert new_contract_info['compiled'] == True
        return new_contract_info['compiled']

    def refresh_deployed_contracts(self):
        # self.deployed_contracts = {}
        return self.rm_json('deployed_contracts')


    def rm_contract (self, contract:str):
        return shutil.rmtree(self.contract_file_info[contract]['path'])

    def new_contract(self, contract:str, compile:bool=True, refresh:bool=False):
        contract_file_info = self.contract_file_info.get(contract,{})
        contract_path = contract_file_info.get('path', ' This is hundo p not a file')
        contract_project_exists = lambda : os.path.exists(contract_path)

        if contract_project_exists():
            if refresh:
                os.rmdir(self.rm_contract(contract))
                
        
        # if not contract_project_exists():
        commune.run_command(f'cargo contract new {contract}', cwd=self.contracts_dir_path)
        
        if compile:
            self.compile(contract)




    def read(self, method:str, args:dict={}, keypair: Keypair = None, contract = None) -> Dict:
        contract = contract if contract != None else self.contract
        keypair = keypair if keypair != None else self.keypair
        # Do a gas estimation of the message
        result = contract.read(self.keypair, method, args=args)
        
        return result




    def exec(self,  method:str, args:dict={}, keypair: Keypair = None, contract = None):
        contract = contract if contract != None else self.contract
        keypair = keypair if keypair != None else self.keypair
        # Do a gas estimation of the message
        gas_predit_result = contract.read(self.keypair, method, args=args)

        # print('Result of dry-run: ', gas_predit_result.value)
        # print('Gas estimate: ', gas_predit_result.gas_required)

        # Do the actual call
        # print('Executing contract call...')
        contract_receipt = contract.exec(keypair, method, args=args, gas_limit=gas_predit_result.gas_required)

        if contract_receipt.is_success:
            print(f'Events triggered in contract: {contract_receipt.contract_events}')
        else:
            raise Exception(f'Error message: {contract_receipt.error_message}')

        result = contract.read(self.keypair, method, args=args)
        return result
    
    @classmethod
    def sandbox(cls):
        self = cls()
        # self.refresh_deployed_contracts()
        contract = self.deploy('erc20', args={'total_supply': 100000})
        # test_accounts = commune.get_module('web3.substrate.account').test_accounts()
        
        # # self.set_contract('erc20')

        # print(test_accounts)
        
        print(self.exec(method='set_ip', args={'owner': self.keypair.ss58_address,'ip': '0.0.0.0:5001'}).value)
        print(self.read(method='get_ip', args={'owner': self.keypair.ss58_address}).value)
        # print(self.contract.call())
import time
if __name__ == "__main__":

    SubstrateContract.sandbox()

    # print(self.contract.metadata.__dict__)
    

    # st.write(self.call('flip'))
    # st.write(self.compile('fam'))
    # st.write(self.contract.__dict__)
    # st.write(self.contract)
    # st.write(self.deploy(contract='fam', args={'init_value': True}, refresh=False))
    # st.write(self.put_json('contract_file_info',self.contract_file_info))
    # st.write(self.get_json('contract_file_info'))
    # st.write(subprocess.r('ls', shell=False, cwd='./commune/substrate'))