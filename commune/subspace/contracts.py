# Python Substrate Interface Library
#
# Copyright 2018-2020 Stichting Polkascan (Polkascan Foundation).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from hashlib import blake2b
from typing import Optional

from .utils import version_tuple

from .exceptions import ExtrinsicFailedException, DeployContractFailedException, \
    ContractReadFailedException, ContractMetadataParseException, StorageFunctionNotFound
from scalecodec.base import ScaleBytes, ScaleType
from scalecodec.types import GenericContractExecResult
from .base import SubstrateInterface, Keypair, ExtrinsicReceipt

__all__ = ['ContractExecutionReceipt', 'ContractMetadata', 'ContractCode', 'ContractInstance', 'ContractEvent']


class ContractMetadata:

    def __init__(self, metadata_dict: dict, substrate: SubstrateInterface):
        """
        Class using the generated metadata.json file to represent the metadata of a contract. The metadata_dict is
        parsed and the used types are extracted, composed and added to the type registry of the runtime

        Parameters
        ----------
        metadata_dict
        substrate
        """
        self.metadata_version = None
        self.metadata_dict = metadata_dict
        self.substrate = substrate
        self.type_registry = {}

        self.__type_offset = 0

        self.__parse_metadata()

    @classmethod
    def create_from_file(cls, metadata_file: str, substrate: SubstrateInterface) -> "ContractMetadata":
        """
        Create a new ContractMetadata object using the provided metadata_file, usually generated by the command
        "cargo +nightly contract generate-metadata" in an ink! project

        Parameters
        ----------
        metadata_file
        substrate

        Returns
        -------
        ContractMetadata
        """
        with open(os.path.abspath(metadata_file), 'r') as fp:
            metadata_string = fp.read()
        return cls(json.loads(metadata_string), substrate)

    def __getattr__(self, item):
        if item in self.metadata_dict:
            return self.metadata_dict[item]
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, item))

    def __convert_to_latest_metadata(self):

        # Determine version
        if 'metadataVersion' in self.metadata_dict:
            self.metadata_version = 0
        elif 'V1' in self.metadata_dict:
            self.metadata_version = 1
        elif 'V2' in self.metadata_dict:
            self.metadata_version = 2
        elif 'V3' in self.metadata_dict:
            self.metadata_version = 3
        elif 'version' in self.metadata_dict:
            self.metadata_version = int(self.metadata_dict['version'])

        if self.metadata_version is None or self.metadata_version > 5:
            raise ContractMetadataParseException("Unsupported metadata version")

        if 1 <= self.metadata_version <= 3:
            version_key = f"V{self.metadata_version}"
            self.metadata_dict['spec'] = self.metadata_dict[version_key]['spec']
            self.metadata_dict['storage'] = self.metadata_dict[version_key]['storage']
            self.metadata_dict['types'] = self.metadata_dict[version_key]['types']
            del self.metadata_dict[version_key]

        # Version converters

        # V1 -> V2: name becomes label; no longer array
        if self.metadata_version <= 1:
            def replace_name_with_label(obj):
                if 'name' in obj:
                    if type(obj['name']) is list:
                        obj['label'] = '::'.join(obj.pop('name'))
                    else:
                        obj['label'] = obj.pop('name')

                return obj

            for section in ['constructors', 'events', 'messages']:

                for idx, c in enumerate(self.metadata_dict['spec'][section]):
                    self.metadata_dict['spec'][section][idx]['args'] = [
                        replace_name_with_label(a) for a in c['args']
                    ]
                    replace_name_with_label(c)

        # V2 -> V3: new payable flags for constructors: default to true
        if self.metadata_version <= 2:
            for idx, c in enumerate(self.metadata_dict['spec']['constructors']):
                c["payable"] = True

        # V4 -> V5: new event fields: module_path and signature_topic
        if self.metadata_version <= 4:
            for idx, c in enumerate(self.metadata_dict['spec']['events']):
                c["module_path"] = None
                c["signature_topic"] = None


    def __parse_metadata(self):

        self.__convert_to_latest_metadata()

        # Check requirements
        if 'types' not in self.metadata_dict:
            raise ContractMetadataParseException("No 'types' directive present in metadata file")

        if 'spec' not in self.metadata_dict:
            raise ContractMetadataParseException("'spec' directive not present in metadata file")

        if 'constructors' not in self.metadata_dict['spec']:
            raise ContractMetadataParseException("No constructors present in metadata file")

        if 'messages' not in self.metadata_dict['spec']:
            raise ContractMetadataParseException("No messages present in metadata file")

        if 'source' not in self.metadata_dict:
            raise ContractMetadataParseException("'source' directive not present in metadata file")

        # check Metadata version
        if 'V0' in self.metadata_dict and version_tuple(self.metadata_dict['metadataVersion']) < (0, 7, 0):
            # Type indexes are 1-based before 0.7.0
            self.__type_offset = 1

        self.type_string_prefix = f"ink::{self.metadata_dict['source']['hash']}"

        if self.metadata_version == 0:

            for idx, metadata_type in enumerate(self.metadata_dict['types']):

                idx += self.__type_offset

                if idx not in self.type_registry:
                    self.type_registry[idx] = self.get_type_string_for_metadata_type(idx)

        else:
            self.substrate.init_runtime()
            portable_registry = self.substrate.runtime_config.create_scale_object('PortableRegistry')
            portable_registry.encode({"types": self.metadata_dict["types"]})

            self.substrate.runtime_config.update_from_scale_info_types(
                portable_registry['types'], prefix=self.type_string_prefix
            )

    def generate_constructor_data(self, name, args: dict = None) -> ScaleBytes:
        """
        Compose the data field used in the "Contracts.instantiate" call, finding the selectors and encoded the args
        of given constructor

        Parameters
        ----------
        name
        args

        Returns
        -------
        ScaleBytes
        """
        if not args:
            args = {}

        for constructor in self.metadata_dict['spec']['constructors']:
            if name == constructor['label']:
                data = ScaleBytes(constructor['selector'])

                for arg in constructor['args']:
                    if arg['label'] not in args:
                        raise ValueError(f"Argument \"{arg['label']}\" is missing")
                    else:
                        data += self.substrate.encode_scale(
                            type_string=self.get_type_string_for_metadata_type(arg['type']['type']),
                            value=args[arg['label']]
                        )
                return data

        raise ValueError(f'Constructor "{name}" not found')

    def get_type_string_for_metadata_type(self, type_id: int) -> str:
        """
        Adds a type included in the metadata (represented by an index in the type list) to the type registry and
        produces a type string that can be used in the scope of the `RuntimeConfigurationObject`.

        Parameters
        ----------
        type_id 1-based index of type locating in the metadata types dict

        Returns
        -------
        str
        """

        if self.metadata_version >= 1:

            if type_id > len(self.metadata_dict['types']):
                raise ValueError(f'type_id {type_id} not found in metadata')

            return f'{self.type_string_prefix}::{type_id}'

        if self.metadata_version == 0:
            # Legacy type parsing

            # Check if already processed
            if type_id in self.type_registry:
                return self.type_registry[type_id]

            if type_id > len(self.metadata_dict['types']):
                raise ValueError(f'type_id {type_id} not found in metadata')

            arg_type = self.metadata_dict['types'][type_id - 1]

            if 'path' in arg_type:

                # Option field
                if arg_type['path'] == ['Option']:

                    # Examine the fields in the 'Some' variant
                    options_fields = arg_type['def']['variant']['variants'][1]['fields']

                    if len(options_fields) == 1:
                        sub_type = self.get_type_string_for_metadata_type(options_fields[0]['type'])
                    else:
                        raise NotImplementedError('Tuples in Option field not yet supported')

                    return f"Option<{sub_type}>"

                # Predefined types defined in crate ink_env
                if arg_type['path'][0:2] == ['ink_env', 'types']:

                    if arg_type['path'][2] == 'Timestamp':
                        return 'Moment'

                    elif arg_type['path'][2] in ['AccountId', 'Hash', 'Balance', 'BlockNumber']:
                        return arg_type['path'][2]

                    else:
                        raise NotImplementedError(f"Unsupported ink_env type '{arg_type['path'][2]}'")

            # RUST primitives
            if 'primitive' in arg_type['def']:
                return arg_type['def']['primitive']

            elif 'array' in arg_type['def']:
                array_type = self.get_type_string_for_metadata_type(arg_type['def']['array']['type'])
                # Generate unique type string
                return f"[{array_type}; {arg_type['def']['array']['len']}]"

            elif 'variant' in arg_type['def']:
                # Create Enum
                type_definition = {
                  "type": "enum",
                  "type_mapping": []
                }
                for variant in arg_type['def']['variant']['variants']:

                    if 'fields' in variant:
                        if len(variant['fields']) > 1:
                            raise NotImplementedError('Tuples as field of enums not supported')

                        enum_value = self.get_type_string_for_metadata_type(variant['fields'][0]['type'])

                    else:
                        enum_value = 'Null'

                    type_definition['type_mapping'].append(
                        [variant['name'], enum_value]
                    )

                # Add to type registry
                self.substrate.runtime_config.update_type_registry_types(
                    {f'{self.type_string_prefix}::{type_id}': type_definition}
                )
                # Generate unique type string
                self.type_registry[type_id] = f'{self.type_string_prefix}::{type_id}'

                return f'{self.type_string_prefix}::{type_id}'

            elif 'composite' in arg_type['def']:
                # Create Struct
                type_definition = {
                    "type": "struct",
                    "type_mapping": []
                }

                for field in arg_type['def']['composite']['fields']:
                    type_definition['type_mapping'].append(
                        [field['name'], self.get_type_string_for_metadata_type(field['type'])]
                    )

                # Add to type registry
                self.substrate.runtime_config.update_type_registry_types(
                    {f'{self.type_string_prefix}::{type_id}': type_definition}
                )

                # Generate unique type string
                self.type_registry[type_id] = f'{self.type_string_prefix}::{type_id}'

                return f'{self.type_string_prefix}::{type_id}'
            elif 'tuple' in arg_type['def']:
                # Create tuple
                elements = [self.get_type_string_for_metadata_type(element) for element in arg_type['def']['tuple']]
                return f"({','.join(elements)})"

            raise NotImplementedError(f"Type '{arg_type}' not supported")

    def get_return_type_string_for_message(self, name) -> str:
        for message in self.metadata_dict['spec']['messages']:
            if name == message['label']:
                if message['returnType'] is None:
                    return 'Null'
                else:
                    return self.get_type_string_for_metadata_type(message['returnType']['type'])

        raise ValueError(f'Message "{name}" not found')

    def generate_message_data(self, name, args: dict = None) -> ScaleBytes:
        """
        Compose the data field used in the "Contracts.call" call, finding the selector and encoded the args
        of provided message name

        Parameters
        ----------
        name: name of message in contract
        args: arguments required by message, in format: `{'name': value}`

        Returns
        -------
        ScaleBytes
        """
        if not args:
            args = {}

        for message in self.metadata_dict['spec']['messages']:
            if name == message['label']:
                data = ScaleBytes(message['selector'])

                for arg in message['args']:
                    if arg['label'] not in args:
                        raise ValueError(f"Argument \"{arg['label']}\" is missing")
                    else:

                        data += self.substrate.encode_scale(
                            type_string=self.get_type_string_for_metadata_type(arg['type']['type']),
                            value=args[arg['label']]
                        )
                return data

        raise ValueError(f'Message "{name}" not found')

    def get_event_data(self, event_id: int) -> dict:
        """
        Looks up the event data for given 0-based event_id

        Parameters
        ----------
        event_id

        Returns
        -------

        """
        if event_id > len(self.metadata_dict['spec']['events']):
            raise ValueError(f'Event ID {event_id} not found')

        return self.metadata_dict['spec']['events'][event_id]

    def get_event_id_by_topic(self, topic: str) -> Optional[int]:
        for event_id, event in enumerate(self.metadata_dict['spec']['events']):
            if topic == event['signature_topic']:
                return event_id

        # raise ValueError(f'Contract event for topic "{topic}" not found')


class ContractEvent(ScaleType):

    def __init__(self, *args, contract_metadata: ContractMetadata = None, **kwargs):
        """
        ScaleType class containing information about a specific Contract Event, it decodes the "data" field in
        the generic "Contracts.ContractExecution" event that is triggered after a successfull "Contracts.call" call.
        """

        self.contract_metadata = contract_metadata
        self.event_id = None
        self.name = None
        self.docs = None
        self.args = []
        super().__init__(*args, **kwargs)

    def process(self):
        self.event_id = self.process_type('u8').value

        event_data = self.contract_metadata.get_event_data(self.event_id)

        self.name = event_data['label']
        self.docs = event_data['docs']
        self.args = event_data['args']

        for arg in self.args:
            # Decode value of event arg with type_string registered in contract
            arg_type_string = self.contract_metadata.get_type_string_for_metadata_type(arg['type']['type'])
            arg['value'] = self.process_type(arg_type_string).value

        return {
            'name': self.name,
            'docs': self.docs,
            'args': self.args
        }

    def process_encode(self, value):
        raise NotImplementedError()


class ContractExecutionReceipt(ExtrinsicReceipt):

    def __init__(self, *args, **kwargs):
        """
        Object extending the `ExtrinsicReceipt` containing more information about the result after submitting a
        "Contracts.call" extrinsic.

        Parameters
        ----------
        args
        kwargs
        """
        self.__contract_events = None
        self.contract_metadata = kwargs.pop('contract_metadata')
        self.contract_address = kwargs.pop('contract_address')
        super(ContractExecutionReceipt, self).__init__(*args, **kwargs)

    @classmethod
    def create_from_extrinsic_receipt(cls, receipt: ExtrinsicReceipt,
                                      contract_metadata: ContractMetadata, contract_address: str = None) -> "ContractExecutionReceipt":
        """
        Promotes a ExtrinsicReceipt object to a ContractExecutionReceipt. It uses the provided ContractMetadata to
        decode "ContractExecution" events

        Parameters
        ----------
        receipt
        contract_metadata

        Returns
        -------
        ContractExecutionReceipt
        """
        return cls(
            substrate=receipt.substrate,
            extrinsic_hash=receipt.extrinsic_hash,
            block_hash=receipt.block_hash,
            finalized=receipt.finalized,
            contract_metadata=contract_metadata,
            contract_address=contract_address
        )

    def process_events(self):
        super().process_events()

        if self.triggered_events:

            self.__contract_events = []

            for event in self.triggered_events:

                if self.substrate.implements_scaleinfo():
                    if event.value['module_id'] == 'Contracts' and event.value['event_id'] == 'ContractEmitted' and event.value['attributes']['contract'] == self.contract_address:

                        contract_data = event['event'][1][1]['data'].value_object

                        if self.contract_metadata.metadata_version >= 5:
                            # Find corresponding contract event by event topic
                            for topic in event.value['topics']:
                                event_id = self.contract_metadata.get_event_id_by_topic(topic)
                                if event_id is not None:
                                    event_bytes = self.substrate.create_scale_object("U8").encode(event_id).data
                                    contract_data = event_bytes + contract_data

                        # Create contract event
                        contract_event_obj = ContractEvent(
                            data=ScaleBytes(contract_data),
                            runtime_config=self.substrate.runtime_config,
                            contract_metadata=self.contract_metadata
                        )

                        contract_event_obj.decode()

                        self.__contract_events.append(contract_event_obj)
                else:

                    if event.event_module.name == 'Contracts' and event.event.name == 'ContractEmitted':

                        # Create contract event
                        contract_event_obj = ContractEvent(
                            data=ScaleBytes(event.params[1]['value']),
                            runtime_config=self.substrate.runtime_config,
                            contract_metadata=self.contract_metadata
                        )

                        contract_event_obj.decode()

                        self.__contract_events.append(contract_event_obj)

    @property
    def contract_events(self):
        if self.__contract_events is None:
            self.process_events()

        return self.__contract_events


class ContractCode:

    def __init__(self, code_hash: bytes = None, metadata: ContractMetadata = None, wasm_bytes: bytes = None,
                 substrate: SubstrateInterface = None):
        """
        Object representing the blueprint of the contract, combining either the code hash and metadata of a contract, or
        the WASM bytes and metadata

        Parameters
        ----------
        code_hash: code hash of an already uploaded contract WASM binary
        metadata
        wasm_bytes: WASM binary
        substrate
        """
        self.code_hash = code_hash
        self.metadata = metadata
        self.wasm_bytes = wasm_bytes
        self.substrate = substrate

    @classmethod
    def create_from_contract_files(cls, wasm_file: str, metadata_file: str,
                                   substrate: SubstrateInterface) -> "ContractCode":
        """
        Create a ContractCode providing paths for the WASM binary file and metadata JSON file generated by the
        ink! project

        Parameters
        ----------
        wasm_file
        metadata_file
        substrate

        Returns
        -------
        ContractCode
        """

        with open(os.path.abspath(wasm_file), 'rb') as fp:
            wasm_bytes = fp.read()
            code_hash = blake2b(wasm_bytes, digest_size=32).digest()

        metadata = ContractMetadata.create_from_file(metadata_file, substrate=substrate)

        return cls(code_hash=code_hash, metadata=metadata, wasm_bytes=wasm_bytes, substrate=substrate)

    @classmethod
    def create_from_code_hash(cls, code_hash: bytes, metadata_file: str,
                              substrate: SubstrateInterface) -> "ContractCode":
        """
        Create a ContractCode providing an existing contract code hash and a path to the metadata JSON file

        Parameters
        ----------
        code_hash
        metadata_file
        substrate

        Returns
        -------
        ContractCode
        """

        metadata = ContractMetadata.create_from_file(metadata_file, substrate=substrate)

        return cls(code_hash=code_hash, metadata=metadata, substrate=substrate)

    def upload_wasm(self, keypair: Keypair, storage_deposit_limit: int = None) -> ExtrinsicReceipt:
        """
        Created and submits an "Contracts.upload_code" extrinsic containing the WASM binary

        Parameters
        ----------
        keypair: Keypair used to sign the extrinsic
        storage_deposit_limit:T he maximum amount of balance that can be charged to pay for the storage consumed

        Returns
        -------
        ExtrinsicReceipt
        """
        if not self.wasm_bytes:
            raise ValueError("No WASM bytes to upload")

        call_function = self.substrate.get_metadata_call_function('Contracts', 'upload_code')

        if not call_function:
            # Try to fall back on legacy `put_code`
            call_function = self.substrate.get_metadata_call_function('Contracts', 'put_code')

        if not call_function:
            raise NotImplementedError("Couldn't find method in Contracts pallet to upload the WASM binary")

        call = self.substrate.compose_call(
            call_module="Contracts",
            call_function=call_function.name,
            call_params={
                'code': '0x{}'.format(self.wasm_bytes.hex()),
                'storage_deposit_limit': storage_deposit_limit
            }
        )

        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=keypair)

        return self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

    def deploy(self, keypair: Keypair, constructor: str, args: dict = None, value: int = 0, gas_limit: dict = None,
               deployment_salt: str = None, upload_code: bool = False, storage_deposit_limit: int = None
               ) -> "ContractInstance":
        """
        Deploys a new instance of the contract after it has been uploaded on-chain, with provided constructor and
        constructor arguments

        Parameters
        ----------
        keypair
        constructor: name of the constructor to use, provided in the metadata
        args: arguments for the constructor
        value: Value sent to created contract address
        gas_limit: Gas limit as WeightV2 type. Will default to {'ref_time': 25990000000, 'proof_size': 11990383647911208550}.
        deployment_salt: optional string or hex-string that acts as a salt for this deployment
        upload_code: When True the WASM blob itself will be uploaded with the deploy, False if the WASM is already present on-chain
        storage_deposit_limit: The maximum amount of balance that can be charged to pay for the storage consumed.

        Returns
        -------
        ContractInstance
        """

        # Lookup constructor
        data = self.metadata.generate_constructor_data(name=constructor, args=args)

        if gas_limit is None:
            gas_limit = {'ref_time': 25990000000, 'proof_size': 11990383647911208550}

        if upload_code is True:

            if not self.wasm_bytes:
                raise ValueError("No WASM bytes to upload")

            call = self.substrate.compose_call(
                call_module='Contracts',
                call_function='instantiate_with_code',
                call_params={
                    'value': value,
                    'gas_limit': gas_limit,
                    'storage_deposit_limit': storage_deposit_limit,
                    'code': '0x{}'.format(self.wasm_bytes.hex()),
                    'data': data.to_hex(),
                    'salt': deployment_salt or ''
                }
            )
        else:

            call = self.substrate.compose_call(
                call_module='Contracts',
                call_function='instantiate',
                call_params={
                    'value': value,
                    'gas_limit': gas_limit,
                    'storage_deposit_limit': storage_deposit_limit,
                    'code_hash': f'0x{self.code_hash.hex()}',
                    'data': data.to_hex(),
                    'salt': deployment_salt or ''
                }
            )

        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=keypair)

        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

        if not result.is_success:
            raise ExtrinsicFailedException(result.error_message)

        for event in result.triggered_events:

            if self.substrate.implements_scaleinfo():

                if event.value['event']['event_id'] == 'Instantiated':
                    return ContractInstance(
                        contract_address=event.value['event']['attributes']['contract'],
                        metadata=self.metadata,
                        substrate=self.substrate
                    )
            else:

                if event.event.name == 'Instantiated':
                    return ContractInstance(
                        contract_address=event.params[1]['value'],
                        metadata=self.metadata,
                        substrate=self.substrate
                    )

        raise DeployContractFailedException()


class ContractInstance:

    def __init__(self, contract_address: str, metadata: ContractMetadata = None, substrate: SubstrateInterface = None):
        self.substrate = substrate
        self.contract_address = contract_address
        self.metadata = metadata

        self.init()

    def init(self):
        # Determine ContractExecResult according to PalletVersion
        try:
            pallet_version = self.substrate.query("Contracts", "PalletVersion")

            if pallet_version.value <= 9:
                self.substrate.runtime_config.update_type_registry_types(
                    {"ContractExecResult": "ContractExecResultTo267"}
                )
            else:
                self.substrate.runtime_config.update_type_registry_types(
                    {"ContractExecResult": "ContractExecResultTo269"}
                )
        except StorageFunctionNotFound:
            pass

    @classmethod
    def create_from_address(cls, contract_address: str, metadata_file: str,
                            substrate: SubstrateInterface = None) -> "ContractInstance":
        """
        Create a ContractInstance object that already exists on-chain providing a SS58-address and the path to the
        metadata JSON of that contract

        Parameters
        ----------
        contract_address: SS58-address of contract
        metadata_file: path to metadata JSON generated for contract
        substrate

        Returns
        -------
        ContractInstance
        """

        metadata = ContractMetadata.create_from_file(metadata_file, substrate=substrate)

        return cls(contract_address=contract_address, metadata=metadata, substrate=substrate)

    def read(self, keypair: Keypair, method: str, args: dict = None,
             value: int = 0, gas_limit: int = None, block_hash: str = None) -> GenericContractExecResult:
        """
        Used to execute non-mutable messages to for example read data from the contract using getters. Can also be used
        to predict gas limits and 'dry-run' the execution when a mutable message is used.
        This method does not submit an extrinsic.

        Parameters
        ----------
        keypair
        method: name of message to execute
        args: arguments of message in {'name': value} format
        value: value to send when executing the message
        gas_limit: dict repesentation of `WeightV2` type
        block_hash: hash of the block to execute the message on

        Returns
        -------
        GenericContractExecResult
        """

        input_data = self.metadata.generate_message_data(name=method, args=args)

        # Execute runtime call in ContractsApi
        call_result = self.substrate.runtime_call("ContractsApi", "call", {
            'dest': self.contract_address,
            'gas_limit': gas_limit,
            'input_data': input_data.to_hex(),
            'origin': keypair.ss58_address,
            'value': value,
            'storage_deposit_limit': None
        }, block_hash)
        if 'Error' in call_result['result']:
            raise ContractReadFailedException(call_result.value['result']['Error'])

        if 'Ok' in call_result['result']:

            try:
                return_type_string = self.metadata.get_return_type_string_for_message(method)
                result_scale_obj = self.substrate.create_scale_object(return_type_string)
                result_scale_obj.decode(ScaleBytes(call_result['result'][1]['data'].value_object))
                call_result.value_object['result'].value_object[1].value_object['data'] = result_scale_obj
                call_result.value['result']['Ok']['data'] = result_scale_obj.value

            except NotImplementedError:
                pass

        return call_result

    def exec(self, keypair: Keypair, method: str, args: dict = None,
             value: int = 0, gas_limit: Optional[dict] = None, storage_deposit_limit: int = None,
             wait_for_inclusion: bool = True, wait_for_finalization: bool = False
             ) -> ContractExecutionReceipt:
        """
        Executes provided message by creating and submitting an extrinsic. To get a gas prediction or perform a
        'dry-run' of executing this message, see `ContractInstance.read`.

        Parameters
        ----------
        keypair
        method: name of message to execute
        args: arguments of message in {'name': value} format
        value: value to send when executing the message
        gas_limit: dict repesentation of `WeightV2` type. When omited the gas limit will be calculated with a `read()`
        storage_deposit_limit: The maximum amount of balance that can be charged to pay for the storage consumed
        wait_for_inclusion: wait until extrinsic is included in a block (only works for websocket connections)
        wait_for_finalization: wait until extrinsic is finalized (only works for websocket connections)

        Returns
        -------
        ContractExecutionReceipt
        """

        if gas_limit is None:
            gas_predit_result = self.read(keypair, method, args, value)
            gas_limit = gas_predit_result.gas_required

        input_data = self.metadata.generate_message_data(name=method, args=args)

        call = self.substrate.compose_call(
            call_module='Contracts',
            call_function='call',
            call_params={
                'dest': self.contract_address,
                'value': value,
                'gas_limit': gas_limit,
                'storage_deposit_limit': storage_deposit_limit,
                'data': input_data.to_hex()
            }
        )

        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=keypair)

        receipt = self.substrate.submit_extrinsic(
            extrinsic, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization
        )

        return ContractExecutionReceipt.create_from_extrinsic_receipt(receipt, self.metadata, self.contract_address)