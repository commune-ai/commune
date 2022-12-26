
from substrateinterface import SubstrateInterface, Keypair, ContractCode

import streamlit as st
# from commune.substrate.account import SubstrateAccount
import os
from glob import glob
import os, sys
sys.path.append(os.getenv('PWD'))
import commune

class SubstrateContract:
    contracts_dir_path = f'{os.getenv("PWD")}/commune/substrate/contract/contracts'
    default_url = "ws://127.0.0.1:9944"
    def __init__(self, keypair:Keypair = None, substrate:'SubstrateInterface' = None):

        self.set_keypair(keypair)
        self.set_substrate(substrate)

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
    def contract2info(self):
        compiled_contract_paths = []
        contract2info = {}
        for f in self.contract_paths:
            contract_info = {}
            target_path = os.path.join(f, 'target')
            contract_name = f.split('/')[-1]

            compiled = False
            if os.path.isdir(target_path):
                file_list = os.listdir(target_path+ '/ink')
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

            contract2info[contract_name] = contract_info

        return contract2info
    @property
    def contract_names(self):
        return [ f.split('/')[-1] for f in self.contract_paths   ]
    


    def deploy(self, 
            contract:str,
            endowment:int=0,
            deployment_salt: str='bro',
            gas_limit:int=1000000000000,
            constructor:str="new",
            args:dict={'total_supply': 100000},
            upload_code:bool=True):
        # Deploy contract

        st.write(contract, 'contract')
        contract_info = self.contract2info[contract]
        if contract_info['compiled'] == False:
            contract_info = self.compile(contract=contract)

        code = ContractCode.create_from_contract_files(
                    metadata_file=contract_info['metadata'],
                    wasm_file= contract_info['wasm'],
                    substrate=self.substrate
                )

        st.write(code)
        st.write(self.keypair)
        self.contract = code.deploy(
            keypair=self.keypair,
            endowment=endowment,
            gas_limit=gas_limit,
            deployment_salt=deployment_salt,
            constructor=constructor,
            args=args,
            upload_code=upload_code
        )
        return self.contract

    @property
    def contract_address(self):
        return self.contract.contract_address

    @property
    def contracts(self):
        return self.list_contracts()

    def list_contracts(self, compiled_only=False):

        contracts = []
        for contract_name,contract_info in self.contract2info.items():
            if compiled_only:
                if contract_info['compiled']:
                    contracts.append(contract_name)
            else:
                contracts.append(contract_name)

        return contracts


    
    
    command_dict = {
        "build": 'cargo +nightly contract build'
    }
    def compile(self, contract:str, rebuild:bool=False):
        ''
        assert contract in self.contract2info, f'available is {self.contract_names}'
        
        contract_info = self.contract2info[contract]
        contract_path = contract_info['path']
        commune.run_command(self.command_dict['build'], cwd=contract_path)

        new_contract_info = self.contract2info[contract]
        st.write(new_contract_info)
        assert new_contract_info['compiled'] == True
        return new_contract_info['compiled']




if __name__ == "__main__":
    contract = SubstrateContract()
    st.write(contract.deploy(contract='erc20', deployment_salt='broaod').__dict__)
    # st.write(subprocess.r('ls', shell=False, cwd='./commune/substrate'))