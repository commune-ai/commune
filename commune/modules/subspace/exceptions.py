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


class SubstrateRequestException(Exception):
    pass


class StorageFunctionNotFound(ValueError):
    pass


class ConfigurationError(Exception):
    pass


class ExtrinsicFailedException(Exception):
    pass


class DeployContractFailedException(Exception):
    pass


class ContractMetadataParseException(ValueError):
    pass


class ContractReadFailedException(Exception):
    pass


class ContractExecFailedException(Exception):
    pass


class BlockNotFound(Exception):
    pass


class ExtrinsicNotFound(Exception):
    pass


class ExtensionCallNotFound(AttributeError):
    pass
