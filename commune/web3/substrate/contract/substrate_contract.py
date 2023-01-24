
from substrateinterface import SubstrateInterface, Keypair, ContractCode, ContractInstance
import shutil
import streamlit as st
# from commune.substrate.account import SubstrateAccount
import os
from typing import *
from glob import glob
import os, sys
import commune
class SubstrateContract:
    
    dir_file_path = os.path.dirname(__file__)
    contracts_dir_path = f'{dir_file_path}/data/ink'
    default_url = "ws://0.0.0.0:9944"
    def __init__(self, keypair:Keypair = None, substrate:'SubstrateInterface' = None):
        self.set_keypair(keypair)
        self.set_substrate(substrate)
    tmp_dir = '/tmp/'+os.path.dirname(__file__)

    @classmethod
    def put_json(cls, path:str, data):
        path = os.path.join(cls.tmp_dir, path)
        commune.put_json(path=path, data=data)
        return path
    @classmethod
    def get_json(cls, path:str):
        path = os.path.join(cls.tmp_dir, path)
        return commune.get_json(path=path)

    def set_substrate(self, substrate=None):
        
        if substrate == None:
            substrate = SubstrateInterface(
                url = self.default_url,
                type_registry_preset='canvas'
            )
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

    def deploy(self, 
            contract:str,
            endowment:int=0,
            deployment_salt: str=None,
            gas_limit:int=1000000000000,
            constructor:str="new",
            args:dict={'total_supply': 100000},
            upload_code:bool=True,
            refresh:bool = False,
            compile:bool=False):
        # Deploy contract

        # If refresh is false, lets see if the contract exists
        if compile:
            self.compile(contract)
        
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

        self.contract = code.deploy(
            keypair=self.keypair,
            endowment=endowment,
            gas_limit=gas_limit,
            deployment_salt=deployment_salt,
            constructor=constructor,
            args=args,
            upload_code=upload_code
        )
        deployed_contracts = self.deployed_contracts


        if contract not in deployed_contracts:
            deployed_contracts[contract] = {}
        deployed_contracts[contract][deployment_salt] = self.contract.contract_address
        self.deployed_contracts = deployed_contracts

        return self.contract

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

    def get_contract(self, contract:str, deployment_salt:str=None) -> Union['Contract', 'contract_addresses']:

        # Check if contract is on chain
        st.write(contract)
        contract_info = self.contract_file_info[contract]
        contract_addresses = self.deployed_contracts.get(contract, None)
        if contract_addresses == None:
            return None

        
        deployment_salt = deployment_salt if deployment_salt else list(contract_addresses.keys())[0]
        contract_address = contract_addresses[deployment_salt]
        contract_substrate_info = self.substrate.query("Contracts", "ContractInfoOf", [contract_address])

        if contract_substrate_info.value:

            print(f'Found contract on chain: {contract_substrate_info.value}')

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
        deployed_contracts = self.get_json('deployed_contracts')

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
        print(new_contract_info)
        st.write(new_contract_info)
        assert new_contract_info['compiled'] == True
        return new_contract_info['compiled']

    def refresh_deployed_contracts(self):
        self.deployed_contracts = {}
        return self.deployed_contracts


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


    def read_contract_value(self):
        # Read current value
        result = self.contract.read(self.keypair, 'get')
        return result.contract_result_data

    def call(self,  method:str, args:dict={}):
        # Do a gas estimation of the message
        gas_predit_result = self.contract.read(self.keypair, method)

        # print('Result of dry-run: ', gas_predit_result.value)
        # print('Gas estimate: ', gas_predit_result.gas_required)

        # Do the actual call
        # print('Executing contract call...')
        contract_receipt = self.contract.exec(self.keypair, method, args={

        }, gas_limit=gas_predit_result.gas_required)

        if contract_receipt.is_success:
            print(f'Events triggered in contract: {contract_receipt.contract_events}')
        else:
            raise Exception(f'Error message: {contract_receipt.error_message}')

        result = self.contract.read(self.keypair, 'get')

        print('Current value of "get":', result.contract_result_data)
        return result

import time
if __name__ == "__main__":
    self = SubstrateContract()
    self.compile('subspace')
    
    # self.new_contract('flipper', compile=True)
    # st.write(self.deploy('flipper', args={'init_value': True}, compile=True))
    # st.write(self.call('flip'))
    # st.write(self.compile('fam'))
    # self.deploy(contract='fam', args={'init_value': True})
    # st.write(self.contract.__dict__)
    # st.write(self.contract)
    # st.write(self.deploy(contract='fam', args={'init_value': True}, refresh=False))
    # st.write(self.put_json('contract_file_info',self.contract_file_info))
    # st.write(self.get_json('contract_file_info'))
    # st.write(subprocess.r('ls', shell=False, cwd='./commune/substrate'))