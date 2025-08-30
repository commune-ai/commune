import os
from copy import deepcopy
from typing import *
from solcx import compile_standard, install_solc, compile_solc
import commune as c
class Contract(c.Module):
    base_dir = '/'.join(__file__.split('/')[:-2])
    contracts_path = os.path.join(c.pwd(), 'contracts')
    def __init__(self,  solc_version: str = '0.8.0'):
        self.solc_version = solc_version
    def compile(self, contract_path):
        with open(contract_path, 'r') as file:
            contract_source = file.read()

        filename = contract_path.split('/')[-1]
        # Compile the contract
        compiled_sol = compile_standard(
            {
                "language": "Solidity",
                "sources": {filename: {"content": contract_source}},
                "settings": {
                    "outputSelection": {
                        "*": {
                            "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                        }
                    }
                },
            },
            solc_version="0.8.0",
        )

        return compiled_sol

    @property
    def interfaces(self):
        return [f.replace('.sol', '') for f in os.listdir(self.interfaces_path) if f.endswith('.sol')]

    def contract_paths(self):
        return [os.path.join(root, f) for root, _, files in os.walk(self.contracts_path) for f in files if f.endswith('.sol')]

    def contracts(self):
        return [os.path.basename(path).replace('.sol', '') for path in self.contract_paths()]

    def get_artifact(self, contract_name):
        contract_path = next(path for path in self.contract_paths() if contract_name in path)
        compiled_contract = self.compile(contract_path)
        return compiled_contract['contracts'][contract_path][contract_name]

    def get_abi(self, contract_name):
        return self.get_artifact(contract_name)['abi']

    def call(self, function, args=[]):
        if len(args) == 0:
            args.append({'from': self.account})
        output = getattr(self.contract, function)(*args)
        return self.parseOutput(function=function, outputs=output)

    def parseOutput(self, function, outputs):
        output_abi_list = self.function_abi_map['outputs']
        
        parsedOutputs = {}
        for i, output_abi in enumerate(output_abi_list):
            output_key = i if not output_abi['name'] else output_abi['name']
            
            parsedOutputs[output_key] = outputs
            if 'components' in output_abi:
                component_names = [c['name'] for c in output_abi['components']]
                
                parseStruct = lambda o: dict(zip(component_names, deepcopy(o)))
                if isinstance(outputs, (list, tuple, set)):
                    parsedOutputs[output_key] = list(map(parseStruct, outputs[i]))
                else:
                    parsedOutputs[output_key] = parseStruct(outputs[i])
        
        return parsedOutputs

    def deploy_contract(self, contract, args, network=None, key=None, **kwargs):
        client = self.get_client(network)
        key = self.resolve_key(key)
        gas_price = self.gas_price()
        nonce = self.get_transaction_count(key)

        params = {
            'from': key.address,
            'nonce': nonce,
            'gasPrice': gas_price,
        }

        contract_artifact = self.get_artifact(contract)
        contract_class = client.eth.contract(abi=contract_artifact['abi'], bytecode=contract_artifact['evm']['bytecode']['object'])
        contract_construct = contract_class.constructor(*args)
        construct_txn = contract_construct.buildTransaction(params)
        signed_tx = self.sign_tx(construct_txn, key=key)
        tx_hash = self.client.eth.send_raw_transaction(signed_tx)
        tx_receipt = self.client.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt
    
    @property
    def interfaces(self):
        interfaces = list(filter(lambda f: f.startswith('interfaces'), self.artifacts()))
        return list(map(lambda f:os.path.dirname(f.replace('interfaces/', '')), interfaces))