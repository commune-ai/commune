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
import pandas as pd
import commune as c


U16_MAX = 2**16 - 1
MAX_REQUEST_SIZE = 9_000_000
IPFS_REGEX = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

class Network:

    def __init__(
        self,
        network:str='main',
        url:str=None,
        mode = 'wss',
        num_connections: int = 1,
        wait_for_finalization: bool = False,
        test = False,
        ws_options = {},
        archive = False,
        timeout: int  = None,
        net = None,
        folder = os.path.expanduser(f'~/.commune/chain'),
        urls = {
        "main":  {"lite": ["api.communeai.net"],  "archive": ["archive-node-0.communeai.net", "archive-node-1.communeai.net"]},
        "test": {"lite": ["testnet.api.communeai.net"]}
    }

    ):
        self.folder = folder
        self.urls = urls
        self.networks = list(self.urls.keys())
        self.set_network(network=network or net or self.default_network, # add a little shortcut,
                         mode=mode,
                         url=url,  
                         test = test,
                         num_connections=num_connections,  
                         ws_options=ws_options,
                         archive=archive,
                         wait_for_finalization=wait_for_finalization, 
                         timeout=timeout)
        
    def switch(self, network=None):
        og_network = self.network
        if network == None:
            if self.network == 'main':
                network = 'test'
            else:
                network = 'main'
        path = __file__
        code = c.get_text(path)
        replace_str = f"network : str = '{self.network}' # network name [main, test]"
        new_str = f"network : str = '{network}' # network name [main, test]"
        code =  code.replace(replace_str, new_str)
        c.put_text(path, code)
        self.network = network
        return {'network': self.network, 'og_network': og_network}

    def set_url(self, url: str = None, mode: str = 'wss'):
        """
        Sets the URL for the chain.

        Args:
            url: The URL to set for the chain.
        """
        self.mode = mode or 'wss'
        if url is None:
            url = self.get_url()
        mode_prefix = mode + '://'
        if not url.startswith(mode_prefix):
            url = mode_prefix + url
        self.url = url
        return {'url': self.url}

    def resolve_network(self, network: str , test: bool = False):
        network = network or 'main'
        assert network in self.networks, f'Network {network} not in {self.networks}'
        if network in ['chain']:
            network = 'main'
        if test: 
            network = 'test'
        return network

    def set_network(self, 
                        network='main',
                        mode = 'wss',
                        url = None,
                        test = False,
                        archive = False,
                        num_connections: int = 1,
                        ws_options: dict[str, int] = {},
                        wait_for_finalization: bool = False,
                        timeout: int  = None ):
        t0 = c.time()

        self.network = self.net = self.resolve_network(network=network, test=test)
        if timeout != None:
            ws_options["timeout"] = timeout
        self.ws_options = ws_options
        self.archive = archive
        self.set_url(url, mode=mode)
        self.num_connections = num_connections
        self.wait_for_finalization = wait_for_finalization

        return {
                'network': self.network, 
                'url': self.url, 
                'mode': self.mode, 
                'num_connections': self.num_connections, 
                'wait_for_finalization': self.wait_for_finalization
                }
        
    def get_url(self,  mode=None, **kwargs):
        mode = mode or self.mode
        sub_key = 'archive' if self.archive else 'lite'
        url = c.choice(self.urls[self.network].get(sub_key))
        if not url.startswith(mode):
            url = mode + '://' + url
        return url    

    @contextmanager
    def get_conn(self, timeout: float = None, init: bool = False):
        """
        Context manager to get a connection from the pool.

        Tries to get a connection from the pool queue. If the queue is empty,
        it blocks for `timeout` seconds until a connection is available. If
        `timeout` is None, it blocks indefinitely.

        Args:
            timeout: The maximum time in seconds to wait for a connection.

        Yields:
            The connection object from the pool.

        Raises:
            QueueEmptyError: If no connection is available within the timeout
              period.
        """

        if not hasattr(self, 'connections_queue'):
            t0 = c.time()
            self.connections_queue = queue.Queue(self.num_connections)
            for _ in range(self.num_connections):
                self.connections_queue.put(SubstrateInterface(self.url, ws_options=self.ws_options, use_remote_preset=True ))
            self.connection_latency = round(c.time() - t0, 2)
            c.print(f'Chain(network={self.network} url={self.url} connections={self.num_connections} latency={self.connection_latency}s)', color='blue') 
        conn = self.connections_queue.get(timeout=timeout)
        if init:
            conn.init_runtime()  # type: ignore
        try:
            if conn.websocket and conn.websocket.connected:  # type: ignore
                yield conn
            else:
                conn = SubstrateInterface(self.url, ws_options=self.ws_options)
                yield conn
        finally:
            self.connections_queue.put(conn)

    def get_storage_keys(
        self,
        storage: str,
        queries: list[tuple[str, list[Any]]],
        block_hash: str,
    ):

        send: list[tuple[str, list[Any]]] = []
        prefix_list: list[Any] = []

        key_idx = 0
        with self.get_conn(init=True) as substrate:
            for function, params in queries:
                storage_key = StorageKey.create_from_storage_function(  # type: ignore
                    storage, function, params, runtime_config=substrate.runtime_config, metadata=substrate.metadata  # type: ignore
                )

                prefix = storage_key.to_hex()
                prefix_list.append(prefix)
                send.append(("state_getKeys", [prefix, block_hash]))
                key_idx += 1
        return send, prefix_list

    def get_lists(
        self,
        storage_module: str,
        queries: list[tuple[str, list[Any]]],
        substrate: SubstrateInterface,
    ) -> list[tuple[Any, Any, Any, Any, str]]:
        """
        Generates a list of tuples containing parameters for each storage function based on the given functions and substrate interface.

        Args:
            functions (dict[str, list[query_call]]): A dictionary where keys are storage module names and values are lists of tuples.
                Each tuple consists of a storage function name and its parameters.
            substrate: An instance of the SubstrateInterface class used to interact with the substrate.

        Returns:
            A list of tuples in the format `(value_type, param_types, key_hashers, params, storage_function)` for each storage function in the given functions.

        Example:
            >>> get_lists(
                    functions={'storage_module': [('storage_function', ['param1', 'param2'])]},
                    substrate=substrate_instance
                )
            [('value_type', 'param_types', 'key_hashers', ['param1', 'param2'], 'storage_function'), ...]
        """

        function_parameters: list[tuple[Any, Any, Any, Any, str]] = []

        metadata_pallet = substrate.metadata.get_metadata_pallet(  # type: ignore
            storage_module
        )
        for storage_function, params in queries:
            storage_item = metadata_pallet.get_storage_function(  # type: ignore
                storage_function
            )

            value_type = storage_item.get_value_type_string()  # type: ignore
            param_types = storage_item.get_params_type_string()  # type: ignore
            key_hashers = storage_item.get_param_hashers()  # type: ignore
            function_parameters.append(
                (
                    value_type,
                    param_types,
                    key_hashers,
                    params,
                    storage_function,
                )  # type: ignore
            )
        return function_parameters

    def _send_batch(
        self,
        batch_payload: list[Any],
        request_ids: list[int],
        extract_result: bool = True,
    ):
        """
        Sends a batch of requests to the substrate and collects the results.

        Args:
            substrate: An instance of the substrate interface.
            batch_payload: The payload of the batch request.
            request_ids: A list of request IDs for tracking responses.
            results: A list to store the results of the requests.
            extract_result: Whether to extract the result from the response.

        Raises:
            NetworkQueryError: If there is an `error` in the response message.

        Note:
            No explicit return value as results are appended to the provided 'results' list.
        """
        results: list[str ] = []
        with self.get_conn(init=True) as substrate:
            try:

                substrate.websocket.send(  #  type: ignore
                    json.dumps(batch_payload)
                )  # type: ignore
            except NetworkQueryError:
                pass
            while len(results) < len(request_ids):
                received_messages = json.loads(
                    substrate.websocket.recv()  # type: ignore
                )  # type: ignore
                if isinstance(received_messages, dict):
                    received_messages: list[dict[Any, Any]] = [received_messages]

                for message in received_messages:
                    if message.get("id") in request_ids:
                        if extract_result:
                            try:
                                results.append(message["result"])
                            except Exception:
                                raise (
                                    RuntimeError(
                                        f"Error extracting result from message: {message}"
                                    )
                                )
                        else:
                            results.append(message)
                    if "error" in message:
                        raise NetworkQueryError(message["error"])

            return results

    def _make_request_smaller(
        self,
        batch_request: list[tuple[T1, T2]],
        prefix_list: list[list[str]],
        fun_params: list[tuple[Any, Any, Any, Any, str]],
    ) -> tuple[list[list[tuple[T1, T2]]], list[Chunk]]:
        """
        Splits a batch of requests into smaller batches, each not exceeding the specified maximum size.

        Args:
            batch_request: A list of requests to be sent in a batch.
            max_size: Maximum size of each batch in bytes.

        Returns:
            A list of smaller request batches.

        Example:
            >>> _make_request_smaller(batch_request=[('method1', 'params1'), ('method2', 'params2')], max_size=1000)
            [[('method1', 'params1')], [('method2', 'params2')]]
        """
        assert len(prefix_list) == len(fun_params) == len(batch_request)

        def estimate_size(request: tuple[T1, T2]):
            """Convert the batch request to a string and measure its length"""
            return len(json.dumps(request))

        # Initialize variables
        result: list[list[tuple[T1, T2]]] = []
        current_batch = []
        current_prefix_batch = []
        current_params_batch = []
        current_size = 0
        chunk_list: list[Chunk] = []

        # Iterate through each request in the batch
        for request, prefix, params in zip(batch_request, prefix_list, fun_params):
            request_size = estimate_size(request)

            # Check if adding this request exceeds the max size
            if current_size + request_size > MAX_REQUEST_SIZE:
                # If so, start a new batch

                # Essentiatly checks that it's not the first iteration
                if current_batch:
                    chunk = Chunk(
                        current_batch, current_prefix_batch, current_params_batch
                    )
                    chunk_list.append(chunk)
                    result.append(current_batch)

                current_batch = [request]
                current_prefix_batch = [prefix]
                current_params_batch = [params]
                current_size = request_size
            else:
                # Otherwise, add to the current batch
                current_batch.append(request)
                current_size += request_size
                current_prefix_batch.append(prefix)
                current_params_batch.append(params)

        # Add the last batch if it's not empty
        if current_batch:
            result.append(current_batch)
            chunk = Chunk(current_batch, current_prefix_batch, current_params_batch)
            chunk_list.append(chunk)

        return result, chunk_list

    def _are_changes_equal(self, change_a: Any, change_b: Any):
        for (a, b), (c, d) in zip(change_a, change_b):
            if a != c or b != d:
                return False

    def rpc_request_batch(
        self, batch_requests: list[tuple[str, list[Any]]], extract_result: bool = True
    ) -> list[str]:
        """
        Sends batch requests to the substrate node using multiple threads and collects the results.

        Args:
            substrate: An instance of the substrate interface.
            batch_requests : A list of requests to be sent in batches.
            max_size: Maximum size of each batch in bytes.
            extract_result: Whether to extract the result from the response message.

        Returns:
            A list of results from the batch requests.

        Example:
            >>> rpc_request_batch(substrate_instance, [('method1', ['param1']), ('method2', ['param2'])])
            ['result1', 'result2', ...]
        """

        chunk_results: list[Any] = []
        # smaller_requests = self._make_request_smaller(batch_requests)
        request_id = 0
        with ThreadPoolExecutor() as executor:
            futures: list[Future[list[str]]] = []
            for chunk in [batch_requests]:
                request_ids: list[int] = []
                batch_payload: list[Any] = []
                for method, params in chunk:
                    request_id += 1
                    request_ids.append(request_id)
                    batch_payload.append(
                        {
                            "jsonrpc": "2.0",
                            "method": method,
                            "params": params,
                            "id": request_id,
                        }
                    )

                futures.append(
                    executor.submit(
                        self._send_batch,
                        batch_payload=batch_payload,
                        request_ids=request_ids,
                        extract_result=extract_result,
                    )
                )
            for future in futures:
                resul = future.result()
                chunk_results.append(resul)
        return chunk_results

    def rpc_request_batch_chunked(
        self, chunk_requests: list[Chunk], extract_result: bool = True
    ):
        """
        Sends batch requests to the substrate node using multiple threads and collects the results.

        Args:
            substrate: An instance of the substrate interface.
            batch_requests : A list of requests to be sent in batches.
            max_size: Maximum size of each batch in bytes.
            extract_result: Whether to extract the result from the response message.

        Returns:
            A list of results from the batch requests.

        Example:
            >>> rpc_request_batch(substrate_instance, [('method1', ['param1']), ('method2', ['param2'])])
            ['result1', 'result2', ...]
        """

        def split_chunks(chunk: Chunk, chunk_info: list[Chunk], chunk_info_idx: int):
            manhattam_chunks: list[tuple[Any, Any]] = []
            mutaded_chunk_info = deepcopy(chunk_info)
            max_n_keys = 35000
            for query in chunk.batch_requests:
                result_keys = query[1][0]
                keys_amount = len(result_keys)
                if keys_amount > max_n_keys:
                    mutaded_chunk_info.pop(chunk_info_idx)
                    for i in range(0, keys_amount, max_n_keys):
                        new_chunk = deepcopy(chunk)
                        splitted_keys = result_keys[i: i + max_n_keys]
                        splitted_query = deepcopy(query)
                        splitted_query[1][0] = splitted_keys
                        new_chunk.batch_requests = [splitted_query]
                        manhattam_chunks.append(splitted_query)
                        mutaded_chunk_info.insert(chunk_info_idx, new_chunk)
                else:
                    manhattam_chunks.append(query)
            return manhattam_chunks, mutaded_chunk_info

        assert len(chunk_requests) > 0
        mutated_chunk_info: list[Chunk] = []
        chunk_results: list[Any] = []
        # smaller_requests = self._make_request_smaller(batch_requests)
        request_id = 0

        with ThreadPoolExecutor() as executor:
            futures: list[Future[list[str]]] = []
            for idx, macro_chunk in enumerate(chunk_requests):
                _, mutated_chunk_info = split_chunks(macro_chunk, chunk_requests, idx)
            for chunk in mutated_chunk_info:
                request_ids: list[int] = []
                batch_payload: list[Any] = []
                for method, params in chunk.batch_requests:
                    # for method, params in micro_chunk:
                    request_id += 1
                    request_ids.append(request_id)
                    batch_payload.append(
                        {
                            "jsonrpc": "2.0",
                            "method": method,
                            "params": params,
                            "id": request_id,
                        }
                    )
                futures.append(
                    executor.submit(
                        self._send_batch,
                        batch_payload=batch_payload,
                        request_ids=request_ids,
                        extract_result=extract_result,
                    )
                )
            for future in futures:
                resul = future.result()
                chunk_results.append(resul)
        return chunk_results, mutated_chunk_info

    def _decode_response(
        self,
        response: list[str],
        function_parameters: list[tuple[Any, Any, Any, Any, str]],
        prefix_list: list[Any],
        block_hash: str,
    ) -> dict[str, dict[Any, Any]]:
        """
        Decodes a response from the substrate interface and organizes the data into a dictionary.

        Args:
            response: A list of encoded responses from a substrate query.
            function_parameters: A list of tuples containing the parameters for each storage function.
            last_keys: A list of the last keys used in the substrate query.
            prefix_list: A list of prefixes used in the substrate query.
            substrate: An instance of the SubstrateInterface class.
            block_hash: The hash of the block to be queried.

        Returns:
            A dictionary where each key is a storage function name and the value is another dictionary.
            This inner dictionary's key is the decoded key from the response and the value is the corresponding decoded value.

        Raises:
            ValueError: If an unsupported hash type is encountered in the `concat_hash_len` function.

        Example:
            >>> _decode_response(
                    response=[...],
                    function_parameters=[...],
                    last_keys=[...],
                    prefix_list=[...],
                    substrate=substrate_instance,
                    block_hash="0x123..."
                )
            {'storage_function_name': {decoded_key: decoded_value, ...}, ...}
        """

        def get_item_key_value(item_key: tuple[Any, ...]) -> tuple[Any, ...]:
            if isinstance(item_key, tuple):
                return tuple(k.value for k in item_key)
            return item_key.value

        def concat_hash_len(key_hasher: str) -> int:
            """
            Determines the length of the hash based on the given key hasher type.

            Args:
                key_hasher: The type of key hasher.

            Returns:
                The length of the hash corresponding to the given key hasher type.

            Raises:
                ValueError: If the key hasher type is not supported.

            Example:
                >>> concat_hash_len("Blake2_128Concat")
                16
            """

            if key_hasher == "Blake2_128Concat":
                return 16
            elif key_hasher == "Twox64Concat":
                return 8
            elif key_hasher == "Identity":
                return 0
            else:
                raise ValueError("Unsupported hash type")

        assert len(response) == len(function_parameters) == len(prefix_list)
        result_dict: dict[str, dict[Any, Any]] = {}
        for res, fun_params_tuple, prefix in zip(
            response, function_parameters, prefix_list
        ):
            if not res:
                continue
            res = res[0]
            changes = res["changes"]  # type: ignore
            value_type, param_types, key_hashers, params, storage_function = (
                fun_params_tuple
            )
            with self.get_conn(init=True) as substrate:
                for item in changes:
                    # Determine type string
                    key_type_string: list[Any] = []
                    for n in range(len(params), len(param_types)):
                        key_type_string.append(
                            f"[u8; {concat_hash_len(key_hashers[n])}]"
                        )
                        key_type_string.append(param_types[n])

                    item_key_obj = substrate.decode_scale(  # type: ignore
                        type_string=f"({', '.join(key_type_string)})",
                        scale_bytes="0x" + item[0][len(prefix):],
                        return_scale_obj=True,
                        block_hash=block_hash,
                    )
                    # strip key_hashers to use as item key
                    if len(param_types) - len(params) == 1:
                        item_key = item_key_obj.value_object[1]  # type: ignore
                    else:
                        item_key = tuple(  # type: ignore
                            item_key_obj.value_object[key + 1]  # type: ignore
                            for key in range(  # type: ignore
                                len(params), len(param_types) + 1, 2
                            )
                        )

                    item_value = substrate.decode_scale(  # type: ignore
                        type_string=value_type,
                        scale_bytes=item[1],
                        return_scale_obj=True,
                        block_hash=block_hash,
                    )
                    result_dict.setdefault(storage_function, {})
                    key = get_item_key_value(item_key)  # type: ignore
                    result_dict[storage_function][key] = item_value.value  # type: ignore

        return result_dict

    def query_batch(
        self, functions: dict[str, list[tuple[str, list[Any]]]],
        block_hash: str = None,
        verbose=False,
    ) -> dict[str, str]:
        """
        Executes batch queries on a substrate and returns results in a dictionary format.

        Args:
            substrate: An instance of SubstrateInterface to interact with the substrate.
            functions (dict[str, list[query_call]]): A dictionary mapping module names to lists of query calls (function name and parameters).

        Returns:
            A dictionary where keys are storage function names and values are the query results.

        Raises:
            Exception: If no result is found from the batch queries.

        Example:
            >>> query_batch(substrate_instance, {'module_name': [('function_name', ['param1', 'param2'])]})
            {'function_name': 'query_result', ...}
        """

        c.print(f'QueryBatch({functions})', verbose=verbose)
        result: dict[str, str] = {}
        if not functions:
            raise Exception("No result")
        with self.get_conn(init=True) as substrate:
            for module, queries in functions.items():
                storage_keys: list[Any] = []
                for fn, params in queries:
                    storage_function = substrate.create_storage_key(  # type: ignore
                        pallet=module, storage_function=fn, params=params
                    )
                    storage_keys.append(storage_function)

                block_hash = substrate.get_block_hash()
                responses: list[Any] = substrate.query_multi(  # type: ignore
                    storage_keys=storage_keys, block_hash=block_hash
                )

                for item in responses:
                    fun = item[0]
                    query = item[1]
                    storage_fun = fun.storage_function
                    result[storage_fun] = query.value

        return result

    def query_batch_map(
        self,
        functions: dict[str, list[tuple[str, list[Any]]]],
        block_hash: str = None,
        path = None,
        max_age=None,
        update=False,
        verbose = False,
    ) -> dict[str, dict[Any, Any]]:
        """
        Queries multiple storage functions using a map batch approach and returns the combined result.

        Args:
            substrate: An instance of SubstrateInterface for substrate interaction.
            functions (dict[str, list[query_call]]): A dictionary mapping module names to lists of query calls.

        Returns:
            The combined result of the map batch query.

        Example:
            >>> query_batch_map(substrate_instance, {'module_name': [('function_name', ['param1', 'param2'])]})
            # Returns the combined result of the map batch query
        """
        c.print(f'QueryBatchMap({functions})', verbose=verbose)
        if path != None:
            path = self.get_path(f'{self.network}/query_batch_map/{path}')
            return c.get(path, max_age=max_age, update=update)
        multi_result: dict[str, dict[Any, Any]] = {}

        def recursive_update(
            d: dict[str, dict[T1, T2]],
            u: Mapping[str, dict[Any, Any]],
        ) -> dict[str, dict[T1, T2]]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)  # type: ignore
                else:
                    d[k] = v  # type: ignore
            return d  # type: ignore

        def get_page():
            send, prefix_list = self.get_storage_keys(storage, queries, block_hash)
            with self.get_conn(init=True) as substrate:
                function_parameters = self.get_lists(storage, queries, substrate)
            responses = self.rpc_request_batch(send)
            # assumption because send is just the storage_function keys
            # so it should always be really small regardless of the amount of queries
            assert len(responses) == 1
            res = responses[0]
            built_payload: list[tuple[str, list[Any]]] = []
            for result_keys in res:
                built_payload.append(("state_queryStorageAt", [result_keys, block_hash]))
            _, chunks_info = self._make_request_smaller(built_payload, prefix_list, function_parameters)
            chunks_response, chunks_info = self.rpc_request_batch_chunked(chunks_info)
            return chunks_response, chunks_info
        
        block_hash = block_hash or self.block_hash() 
        for storage, queries in functions.items():
            chunks, chunks_info = get_page()
            # if this doesn't happen something is wrong on the code
            # and we won't be able to decode the data properly
            assert len(chunks) == len(chunks_info)
            for chunk_info, response in zip(chunks_info, chunks):
                try:
                    storage_result = self._decode_response(response, chunk_info.fun_params, chunk_info.prefix_list, block_hash)
                except Exception as e:
                    c.print(f'Error decoding response for {storage} with queries {queries}: {e}', color='red')
                    continue
                multi_result = recursive_update(multi_result, storage_result)

        results =  self.process_results(multi_result)
        if path != None:
            print('Saving results to -->', path)
            c.put(path, results)
        return results
            
    def block_hash(self, block: Optional[int] = None) -> str:
        with self.get_conn(init=True) as substrate:
            block_hash = substrate.get_block_hash(block)
        return block_hash
    
    def block(self, block: Optional[int] = None) -> int:
        with self.get_conn(init=True) as substrate:
            block_number = substrate.get_block_number(block)
        return block_number

    
    def runtime_spec_version(self):
        # Get the runtime version
        return self.query(name='SpecVersionRuntimeVersion', module='System')

    def query(
        self,
        name: str,
        params: list[Any] = [],
        module: str = "SubspaceModule",
    ) -> Any:
        """
        Queries a storage function on the network.

        Sends a query to the network and retrieves data from a
        specified storage function.

        Args:
            name: The name of the storage function to query.
            params: The parameters to pass to the storage function.
            module: The module where the storage function is located.

        Returns:
            The result of the query from the network.
        Raises:
            NetworkQueryError: If the query fails or is invalid.
        """
        if  'Weights' in name:
            module = 'SubnetEmissionModule'
        if '/' in name:
            module, name = name.split('/')
        result = self.query_batch({module: [(name, params)]})
        return result[name]

    def pallets(self):
        """
        Retrieves the list of pallets from the network.

        Returns:
            A list of pallets available on the network.
        """
        with self.get_conn(init=True) as substrate:
            pallets = substrate.get_metadata_pallets()
        return pallets

    def query_map(
        self,
        name: str='Emission',
        params: list[Any] = [],
        module: str = "SubspaceModule",
        extract_value: bool = True,
        max_age=None,
        update=False,
        block = None,
        block_hash: str = None,
    ) -> dict[Any, Any]:
        """
        Queries a storage map from a network node.

        Args:
            name: The name of the storage map to query.
            params: A list of parameters for the query.
            module: The module in which the storage map is located.

        Returns:
            A dictionary representing the key-value pairs
              retrieved from the storage map.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        if name == 'Weights':
            module = 'SubnetEmissionModule'
        path =  self.get_path(f'{self.network}/query_map/{module}/{name}_params={params}')
        result = c.get(path, None, max_age=max_age, update=update)
        if result == None:
            result = self.query_batch_map({module: [(name, params)]}, block_hash)
            if extract_value:
                if isinstance(result, dict):
                    result = result
                else:
                    result =  {k.value: v.value for k, v in result}  # type: ignore
            result = result.get(name, {})
            keys = result.keys()
            if any([isinstance(k, tuple) for k in keys]):
                new_result = {}
                for k,v in result.items():
                    self.dict_put(new_result, list(k), v )  
                result = new_result
            c.put(path, result)

        return self.process_results(result)

    def process_results(self, x:dict) -> dict:
        new_x = {}
        for k in list(x.keys()):
            if type(k) in  [tuple, list]:
                self.dict_put(new_x, list(k), x[k])
                new_x = self.process_results(new_x)
            elif isinstance(x[k], dict):
                new_x[k] = self.process_results(x[k])
            else:
                new_x[k] = x[k]
        return new_x
                
    def dict_put(self, input_dict: dict ,keys : list, value: Any ):
        """
        insert keys that are dot seperated (key1.key2.key3) recursively into a dictionary
        """
        if isinstance(keys, str):
            keys = keys.split('.')
        elif not type(keys) in [list, tuple]:
            keys = str(keys)
        key = keys[0]
        if len(keys) == 1:
            if  isinstance(input_dict,dict):
                input_dict[key] = value
            elif isinstance(input_dict, list):
                input_dict[int(key)] = value
            elif isinstance(input_dict, tuple):
                input_dict[int(key)] = value
        elif len(keys) > 1:
            if key not in input_dict:
                input_dict[key] = {}
            self.dict_put(input_dict=input_dict[key],
                                keys=keys[1:],
                                value=value)

    def to_nanos(self, amount):
        return amount * 10**9

    def is_float(self, x):
        """
        Check if the input is a float or can be converted to a float.
        """
        try:
            float(float(str(x).replace(',', '')))
            return True
        except ValueError:
            return False

    def get_balances(
        self, 
        addresses=None,
        extract_value: bool = False, 
        block_hash: str = None,
        threads = 8,
        timeout= 120

    ) -> dict[str, dict[str, int ]]:
        """
        Retrieves a mapping of account balances within the network.
        """

        addresses = addresses or list(c.key2address().values())
        with self.get_conn(init=True) as substrate:
            storage_keys = [substrate.create_storage_key(pallet='System', storage_function='Account', params=[ka]) for ka in addresses if not ka.startswith('0x')]
            balances =  substrate.query_multi(storage_keys, block_hash=block_hash)
        key2balance = {k:v[1].value['data']['free'] for k,v in zip(addresses, balances) }
        return key2balance


    def balance(
        self,
        addr: Ss58Address=None,
        fmt = 'j'
    ) -> int:
        """
        Retrieves the balance of a specific key.
        """

        addr = self.get_key_address(addr)
        result = self.query("Account", module="System", params=[addr])
        return self.format_amount(result["data"]["free"], fmt=fmt)
    bal = balance
    def block(self) -> dict[Any, Any]:
        """
        Retrieves information about a specific block in the network.
        """
        block_hash: str = None

        with self.get_conn() as substrate:
            block: dict[Any, Any] = substrate.get_block_number(  # type: ignore
                block_hash  # type: ignore
            )

        return block

    def existential_deposit(self, block_hash: str = None) -> int:
        """
        Retrieves the existential deposit value for the network.
        """
        return 1


    def module_exists(self, key, subnet=0):
        key_address = self.get_key(key).key_address
        return any([key_address == m['key'] for m in self.mods(subnet=subnet)])


    def voting_power_delegators(self) -> list[Ss58Address]:
        result = self.query("NotDelegatingVotingPower", [], module="GovernanceModule")
        return result

    def add_transfer_dao_treasury_proposal(
        self,
        key: Keypair,
        data: str,
        amount_nano: int,
        dest: Ss58Address,
    ):
        params = {"dest": dest, "value": amount_nano, "data": data}

        return self.call(
            module="GovernanceModule",
            fn="add_transfer_dao_treasury_proposal",
            params=params,
            key=key,
        )


    def to_nano(self, value):
        return value * (10 ** 9)
    

    def to_joules(self, value):
        return value / (10 ** 9)
    to_j = to_joules
        

    def get_key_name(self, key:str ) -> str:
        address2key = c.address2key()
        if hasattr(key, 'key_address'):
            key = key.key_address  
        assert key in address2key, f"Key {key} not found"
        return address2key.get(key, key)

    @classmethod
    def valid_h160_address(cls, address):
        # Check if it starts with '0x'
        if not address.startswith('0x'):
            return False
        
        # Remove '0x' prefix
        address = address[2:]
        
        # Check length
        if len(address) != 40:
            return False
        
        # Check if it contains only valid hex characters
        if not re.match('^[0-9a-fA-F]{40}$', address):
            return False
        
        return True

    def valid_ss58_address(self, address):
        from .substrate.utils.ss58 import is_valid_ss58_address
        return is_valid_ss58_address(address)


    def get_key_address(self, key:str ) -> str:
        if isinstance(key, str):
            if self.valid_h160_address(key) or self.valid_ss58_address(key):
                return key
            else:
                key = c.get_key(key)
                if key == None:
                    raise ValueError(f"Key {key} not found")
                return key.key_address
        elif hasattr(key, 'key_address'):
            return key.key_address
        else:
            raise ValueError(f"Key {key} not found")

    def get_key(self, key:str ):
        if isinstance(key, str):
            key = c.get_key( key )
        assert hasattr(key, 'key_address'), f"Key {key} not found"
        return key

    def get_path(self, path:str) -> str:
        if not path.startswith(self.folder):
            path = f'{self.folder}/{path}'
        return path

    def format_amount(self, x, fmt='nano') :
        if type(x) in [dict]:
            for k,v in x.items():
                x[k] = self.format_amount(v, fmt=fmt)
            return x
        if fmt in ['j', 'com', 'comai']:
            x = x / 10**9
        elif fmt in ['nano', 'n', 'nj', 'nanos', 'ncom']:
            x = x * 10**9
        else:
            raise NotImplementedError(fmt)
        return x

    @property
    def substrate(self):
        return self.get_conn()
    @property
    def block_number(self) -> int:
        return self.get_conn().block_number(block_hash=None)
    

    @staticmethod
    def vec82str(x):
        x = x or []
        return ''.join([chr(ch) for ch in x]).strip()
    
    def __str__(self):
        return f'Chain(network={self.network}, url={self.url})'
    
    def get_metadata_pallet(self, pallet):
        with self.get_conn() as substrate:
            metadata = substrate.get_metadata().get_metadata_pallet(pallet)
        return metadata
                    

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
        return self.call( module="Balances", fn="transfer_keep_alive", params=params, key=key, multisig=multisig, safety=safety)


    def sudo_multisig(self) -> List[str]:
        return self.get_multisig(sudo_multisig_data)

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
        if wait_for_finalization is None:
            wait_for_finalization = self.wait_for_finalization
        # getting the call ready
        with self.get_conn() as substrate:
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
        c.print(f"Call(network={self.network}\nmodule={info_call['module']} \nfn={info_call['fn']} \nkey={key.ss58_address} ({key_name}) \nparams={info_call['params']}) \n)", color='cyan')

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

    

    def events(self, block=None, back=None) -> list[dict[str, Any]]:
        """
        Get events from a specific block or the latest block
        """
        if back != None:
            block = self.block()
            since = self.block() - back
            assert since < block, f"Block {block} is not greater than since {since}"
            block2events  = {}
            future2block = {}
            for block in range(since, block + 1):
                print(f"Getting events for block {block}")
                f = c.submit(self.events, params=dict(block=block), timeout=60)
                future2block[f] = block
            for future in c.as_completed(future2block):
                block = future2block[future]
                events = future.result()
                block2events[block] = events
                print(f"Got {len(events)} events for block {block}")

            return block2events

        block = block or self.block()
        path = self.get_path(f'events/{block}')
        events = c.get(path, None)
        if events is not None:
            c.print(f"Events for block {block} already cached, returning cached events", color='green')
            return events
        block_hash = self.block_hash(block)
        with self.get_conn(init=True) as substrate:
            # Get events from the block
            events = substrate.get_events(block_hash)
        
        events = [e.value for e in events]
        # include the tx hash

        # c.put(path, events)
        return events
