
from substrateinterface import SubstrateInterface, Keypair, ContractCode

import streamlit as st
# from commune.substrate.account import SubstrateAccount
import os
from glob import glob

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
    def compiled_contract_paths(self):
        compiled_contract_paths = []
        for f in self.contract_paths:
            wasm_target_path = os.path.join(f, 'target')
            if os.path.isdir(wasm_target_path):
                compiled_contract_paths.append(f)
                st.write(os.listdir(wasm_target_path+ '/ink'))
        return compiled_contract_paths
    @property
    def contract_names(self):
        return [ f.split('/')[-1] for f in self.contract_paths   ]
    


    def deploy(self, 
            contract_path:str,
            endowment:int=0,
            gas_limit:int=1000000000000,
            constructor:str="new",
            args:dict={'init_value': True},
            upload_code:bool=True):
        # Deploy contract
        code = ContractCode.create_from_contract_files(
                    metadata_file=contract_path + '.json',
                    wasm_file= contract_path + '.wasm',
                    substrate=self.substrate
                )

        self.contract = code.deploy(
            keypair=self.keypair,
            endowment=endowment,
            gas_limit=gas_limit,
            constructor=constructor,
            args=args,
            upload_code=upload_code
        )

    @property
    def contract_address(self):
        return self.contract.contract_address







if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getenv('PWD'))
    import commune


    contract = SubstrateContract()

    st.write(contract.compiled_contract_paths)