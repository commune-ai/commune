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

class Chain:

    min_stake = 50000
    tempo = 60
    blocktime = block_time = 8
    blocks_per_day = 24*60*60/block_time

    
    urls = {
        "main":  {"lite": ["commune-api-node-0.communeai.net"],  "archive": ["commune-archive-node-0.communeai.net", "commune-archive-node-1.communeai.net"]},
        "test": {"lite": ["testnet.api.communeai.net"]}
    }
    networks = list(urls.keys())
    default_network : str = 'main' # network name [main, test]

    def __init__(
        self,
        network:str=None,
        url:str=None,
        mode = 'wss',
        num_connections: int = 1,
        wait_for_finalization: bool = False,
        test = False,
        ws_options = {},
        archive = False,
        timeout: int  = None,
        net = None,
        folder = os.path.expanduser(f'~/.commune/chain') 

    ):
        self.folder = folder
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

    def set_connections(self, num_connections: int = 1):
        return self.set_network(num_connections=num_connections)

    def set_network(self, 
                        network=None,
                        mode = 'wss',
                        url = None,
                        test = False,
                        archive = False,
                        num_connections: int = 1,
                        ws_options: dict[str, int] = {},
                        wait_for_finalization: bool = False,
                        timeout: int  = None ):
        t0 = c.time()

        if network == None:
            network = self.network
        if network in ['chain']:
            network = 'main'
        if test: 
            network = 'test'
        self.network = network
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
        url_options = self.urls[self.network].get(sub_key, [])
        if len(url_options) == 0 and self.archive:
            print(f'No archive nodes available for network {self.network}, switching to lite mode')
            self.archive = False
            return self.get_url(mode=mode, **kwargs)
        url = c.choice(url_options)
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

    def name(self, key: str, subnet=2):
        """
        Get the name of a module key
        """
        if isinstance(key, str):
            key = self.get_key(key)
        if isinstance(key, Keypair):
            key = key.ss58_address
        if subnet == None:
            subnet = self.get_subnet(subnet)
        return self.query(name='ModuleName', module='SubspaceModule', params=[key, subnet])

    def get_module_url(self, name: str,public = False):
        """
        Get the url of a module key
        """
        namespace = c.namespace()
        url = namespace.get(name, url)
        ip = (c.ip() if public else '0.0.0.0')
        port = url.split(':')[-1]
        url = ip +':'+ port
        return url

    def metadata(self, subnet=2) -> str:
        metadata = self.query_map('Metadata', [subnet])
        return metadata
    

    def curator_applications(self) -> dict[str, dict[str, str]]:
        applications = self.query_map(
            "CuratorApplications", module="GovernanceModule", params=[],
            extract_value=False
        )
        return applications

    def weights(self, subnet: int = 0, extract_value: bool = False ) -> dict[int, list[tuple[int, int]]]:
        subnet = self.get_subnet(subnet)
        weights_dict = self.query_map("Weights",[subnet],extract_value=extract_value, module='SubnetEmissionModule')
        return {int(k): v for k,v in weights_dict.items()}

    def root_weights(self, key: str=None, extract_value: bool = False):
        key_address = self.get_key_address(key)
        weights = self.module(key_address, subnet=0)['weights']
        netuid2subnet = self.netuid2subnet()
        weights = {netuid2subnet.get(k):v for k,v in weights}
        weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return weights

    def val2subnet(self) -> dict[int, list[tuple[int, int]]]:
        """
        Retrieves a mapping of validator weights to subnets.
        """
        weights = self.weights()
        netuid2subnet = self.netuid2subnet()
        val2subnet = {}
        for k, v in weights.items():
            for kk, vv in v:
                if not kk in val2subnet:
                    val2subnet[kk] = []
                val2subnet[kk].append((netuid2subnet.get(k), vv))
        return val2subnet

    def addresses( self, subnet: int = 0, extract_value: bool = False, max_age: int = 60, update: bool = False ) -> dict[int, str]:
        subnet = self.get_subnet(subnet)
        addresses = self.query_map("Address", [subnet], extract_value=extract_value, max_age=max_age, update=update)
        sorted_uids = list(sorted(list(addresses.keys())))
        return [addresses[uid] for uid in sorted_uids]


    def state(self, update=False, max_age=None):
        state = {}
        state['stake_from'] = self.stake_from(fmt='j', update=update, max_age=max_age)
        state['stake_to'] = self.stake_to(fmt='j')
        state['stake'] =  {k: sum(v.values()) for k,v in state['stake_from'].items()}
        return state

    def subnet(self,subnet=0, update=False, max_age=None):
        futures = []
        path = self.get_path(f'{self.network}/subnet_state/{subnet}')
        state = c.get(path, max_age=max_age, update=update)
        if state == None:
            c.print(f"subnet_state: {path} not found")
            futures = [c.submit(self.subnets, kwargs=dict(subnet=subnet, max_age=max_age, update=update)), 
                        c.submit(self.mods, kwargs=dict(subnet=subnet, max_age=max_age, update=update))]
            params, modules = c.wait(futures)
            state = {'params': params, 'modules': modules}
            c.put(path, state)
        return state
    sync = state

    def subnet_params(self, subnet=0, update=False, max_age=None):
        return self.subnet(subnet=subnet, update=update, max_age=max_age)['params']

    def stake_from(self, key=None, extract_value: bool = False, fmt='j', update=False, **kwargs
    ) -> dict[Ss58Address, list[tuple[Ss58Address, int]]]:
        """
        Retrieves a mapping of stakes from various sources for keys on the network.
        """
        params = [self.get_key_address(key)] if key else []
        result = self.query_map("StakeFrom", params, extract_value=extract_value, update=update, **kwargs)
        return self.format_amount(result, fmt=fmt)

    def get_stake(self, key=None, update=False): 
        """
        Retrieves the stake for a given key.
        """
        stake_from = self.stake_from(key=key, extract_value=True, update=update)
        return sum(stake_from.values())

    stakefrom = stake_from 

    def stake_to( self, key=None, extract_value: bool = False, fmt='j', update=False, **kwargs ) -> dict[Ss58Address, list[tuple[Ss58Address, int]]]:
        """
        Retrieves a mapping of stakes to destinations for keys on the network.
        """
        if key:
            result =  self.query_map("StakeTo", [self.get_key_address(key)], extract_value=False, update=update)
            return self.format_amount(result, fmt=fmt)
        stakefrom = self.stakefrom(extract_value=extract_value, fmt=fmt, update=update, **kwargs)
        staketo = {}
        for k,v in stakefrom.items():
            for kk,vv in v.items():
                if not kk in staketo:
                    staketo[kk] = {}
                staketo[kk][k] = vv
        return staketo
    staketo = stake_to

    def max_allowed_weights(
        self, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of maximum allowed weights for the network.
        """
        return self.query_map("MaxAllowedWeights", extract_value=extract_value)
    
    def legit_whitelist(
        self, extract_value: bool = False
    ) -> dict[Ss58Address, int]:
        """
        Retrieves a mapping of whitelisted addresses for the network.
        """
        return self.query_map( "LegitWhitelist", module="GovernanceModule", extract_value=extract_value)
    
    def subnet_names(self, extract_value: bool = False, max_age=None, update=False, block=None) -> dict[int, str]:
        """
        Retrieves a mapping of subnet names within the network.
        """
        subnet_names =  self.query_map("SubnetNames", extract_value=extract_value, max_age=max_age, update=update, block=block)

        return  [str(v.lower()) for k,v in subnet_names.items()]

    def subnet_map(self, max_age=None, update=False, **kwargs) -> dict[int, str]:
        """
        Retrieves a mapping of subnet names within the network.
        """
        return self.subnet2netuid(max_age=max_age, update=update, **kwargs)

    def get_subnet_name(self, subnet: str) -> int:
        subnet = self.get_subnet(subnet)
        subnet_map = self.subnet_map()
        netuid2name = {v:k for k,v in subnet_map.items()}
        if subnet in netuid2name:
            subnet = netuid2name[subnet]
        reverse_subnet_map = {v:k for k,v in subnet_map.items()}
        assert subnet in subnet_map, f"Subnet {subnet} not found, {subnet_map}"
        return subnet

    def get_subnet(self, subnet:Optional[str]=None) -> int:
        subnet = subnet or 0 
        if c.is_int(subnet):
            subnet = int(subnet)
        if isinstance(subnet, str):
            subnet2netuid = self.subnet2netuid()
            if subnet in subnet2netuid:
                subnet =  subnet2netuid[subnet]
            else:
                subnet2netuid = self.subnet2netuid(update=1)
                subnet = subnet.lower()
                subnet = subnet2netuid.get(subnet, subnet)
                # assert subnet in subnet2netuid, f"Subnet {subnet} not found"
                
        return subnet



    def get_balances(
        self, 
        addresses=None,
        extract_value: bool = False, 
        block_hash: str = None,
        threads = 24,
        timeout= 120,
        connections = 2,

    ) -> dict[str, dict[str, int ]]:
        """
        Retrieves a mapping of account balances within the network.
        """

            
        addresses = addresses or list(c.key2address().values())

        if threads > 1:
            self.set_connections(connections)
            futures = []
            progress = c.tqdm(total=len(addresses), desc='Getting Balances', unit='addr')
            chunk_size = max(1, len(addresses) // threads)
            
            future2addresses = {}
            for i in range(0, len(addresses), chunk_size):
                print(f'Getting balances for addresses {i} to {i + chunk_size}...')
                chunk = addresses[i:i + chunk_size]
                params = dict(addresses=chunk, extract_value=extract_value, block_hash=block_hash, threads=1)
                future2addresses[c.submit(self.get_balances, params)] = chunk

            results = {}
            for f in c.as_completed(future2addresses, timeout=timeout):
                addresses_chunk = future2addresses[f]
                balances_chunk = f.result()
                assert len(balances_chunk) == len(addresses_chunk), f"Expected {len(addresses_chunk)} balances, got {len(balances_chunk)}"
                progress.update(len(addresses_chunk))
                results.update(dict(zip(addresses_chunk, balances_chunk)))
            return results

        addresses = [a for a in addresses if not a.startswith('0x')]
        with self.get_conn(init=True) as substrate:
            storage_keys = [substrate.create_storage_key(pallet='System', storage_function='Account', params=[ka]) for ka in addresses]
            balances =  substrate.query_multi(storage_keys, block_hash=block_hash)
        key2balance = {k:v[1].value['data']['free'] for k,v in zip(addresses, balances) }
        return key2balance
    
    def my_balance(self,  max_age=None, update=False, **kwargs):
        path = self.get_path(f'{self.network}/my_balance')
        balances = c.get(path, None, update=update, max_age=max_age)
        if balances == None:
            key2address = c.key2address()
            addresses = list(key2address.values())
            balances = self.get_balances(addresses=addresses, **kwargs)
            address2key = c.address2key()
            balances = {address2key.get(k, k):v for k,v in balances.items()}
            c.put(path, balances)

        print(balances)    
        balances = {k: v for k, v in balances.items() if v > 0}
        balances = dict(sorted(balances.items(), key=lambda x: x[1], reverse=True))
        return self.format_amount(balances, fmt='j')
    
    def balances(self, *args, **kwargs):  
        return self.my_balance(*args, **kwargs)

    def proposal(self, proposal_id: int = 0):
        """
        Queries the network for a specific proposal.
        """

        return self.query(
            "Proposals",
            params=[proposal_id],
        )

    def proposals(
        self, extract_value: bool = False,update: bool = False
    ) -> dict[int, dict[str, Any]]:
        proposals =  self.query_map( "Proposals", extract_value=extract_value, module="GovernanceModule", update=update)
        return proposals

    props = proposals

    def dao_treasury_address(self) -> Ss58Address:
        return self.query("DaoTreasuryAddress", module="GovernanceModule")

    def namespace(self, subnet: int = 0, search=None, update=False, max_age=None) -> Dict[str, str]:
        subnet = self.get_subnet(subnet)
        path = self.get_path(f'{self.network}/namespace/{subnet}')
        namespace = c.get(path,None, max_age=max_age, update=update)
        if namespace == None:
            results =  self.query_batch_map({'SubspaceModule': [('Name', [subnet]), ('Address', [subnet])]})
            names = results['Name']
            addresses = results['Address']
            namespace = {}
            for uid, name in names.items():
                namespace[name] = addresses[uid]
        if search:
            namespace = {k:v for k,v in namespace.items() if search in k}
        return namespace

    def n(self, subnet: int = 0, max_age=None, update=False ) -> int:
        """
        Queries the network for the 'N' hyperparameter, which represents how
        many modules are on the network.
        """
        subnet = self.get_subnet(subnet)
        n =  self.query_map("N", params=[], max_age=max_age, update=update)
        if str(subnet) in n:
            subnet = str(subnet)
        return n[subnet]
    
    def total_stake(self, block_hash: str = None) -> int:
        """
        Retrieves a mapping of total stakes for keys on the network.
        """

        return self.query("TotalStake", block_hash=block_hash)

    def registrations_per_block(self):
        """
        Queries the network for the number of registrations per block.
        """

        return self.query(
            "RegistrationsPerBlock",
        )

    def unit_emission(self) -> int:
        """
        Queries the network for the unit emission setting.
        """

        return self.query("UnitEmission", module="SubnetEmissionModule")

    def wallets_with_min_balance(self, min_balance: int = 0) -> list[Ss58Address]:
        wallets = self.wallets(mode='list')
        new_wallets = []
        for w in wallets:
            if w['name'] != '--' and w['balance'] > min_balance:
                new_wallets.append(w)
        return [w['name'] for w in new_wallets]


    def wallet(self, key):
        balance = self.balance(key)
        staketo = self.staketo(key=key, extract_value=True)
        key = self.get_key_address(key)
        return {
            'key': key,
            'balance': balance,
            'staketo': staketo,
        }




    def tx_rate_limit(self) -> int:
        """
        Queries the network for the transaction rate limit.
        """

        return self.query(
            "TxRateLimit",
        )

    def subnet_burn(self) -> int:
        """
        Queries the network for the subnet burn value.
        """

        return self.to_joules(self.query(
            "SubnetBurn",
        ))
    
    def vote_mode_global(self) -> str:
        """
        Queries the network for the global vote mode setting.
        """

        return self.query(
            "VoteModeGlobal",
        )

    def max_proposals(self) -> int:
        """
        Queries the network for the maximum number of proposals allowed.
        """

        return self.query(
            "MaxProposals",
        )
        
    def get_stakefrom(
        self,
        key: Ss58Address,
        fmt = 'j',
    ) -> dict[str, int]:
        """
        Retrieves the stake amounts from all stakers to a specific staked address.
        """
        key = self.get_key_address(key)
        result = self.query_map("StakeFrom", [key], extract_value=False)
        return self.format_amount(result, fmt=fmt)
    get_stake_from = get_stakefrom


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

    def min_stake(self, update: int = 0) -> int:
        valis = self.valis(df=False, update=update)
        return min([v['stake'] for v in valis if v['stake'] > 0])

    def delegate_rootnet_control(self, key: Keypair, dest: Ss58Address):
        params = {"origin": key, "target": dest}

        return self.call(
            module="SubspaceModule",
            fn="delegate_rootnet_control",
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

    def subnet_params(self, 
                    subnet : Optional[str] = None,
                    block_hash: Optional[str] = None, 
                    max_age: Optional[int] =None, 
                    update=False) -> dict[int, SubnetParamsWithEmission]:
        """
        Gets all subnets info on the network
        """ 
        path = self.get_path(f'{self.network}/params_map')
        results = c.get(path,None, max_age=max_age, update=update)
        if results == None:
            params = []
            bulk_query = self.query_batch_map(
                {
                    "SubspaceModule": [
                        ("ImmunityPeriod", params),
                        ("MinAllowedWeights", params),
                        ("MaxAllowedWeights", params),
                        ("Tempo", params),
                        ("MaxAllowedUids", params),
                        ("Founder", params),
                        ("FounderShare", params),
                        ("IncentiveRatio", params),
                        ("SubnetNames", params),
                        ("MaxWeightAge", params),
                        ("BondsMovingAverage", params),
                        ("MaximumSetWeightCallsPerEpoch", params),
                        ("MinValidatorStake", params),
                        ("MaxAllowedValidators", params),
                        ("ModuleBurnConfig", params),
                        ("SubnetMetadata", params),
                    ],
                    "GovernanceModule": [
                        ("SubnetGovernanceConfig", params),
                    ],
                    "SubnetEmissionModule": [
                        ("SubnetEmission", params),
                    ],

                },
                block_hash,
            )

            # Extract the relevant data from the bulk query
            subnet_maps: SubnetParamsMaps = {
                "emission": bulk_query["SubnetEmission"],
                "tempo": bulk_query["Tempo"],
                "min_allowed_weights": bulk_query["MinAllowedWeights"],
                "max_allowed_weights": bulk_query["MaxAllowedWeights"],
                "max_allowed_uids": bulk_query["MaxAllowedUids"],
                "founder": bulk_query["Founder"],
                "founder_share": bulk_query["FounderShare"],
                "incentive_ratio": bulk_query["IncentiveRatio"],
                "name": bulk_query["SubnetNames"],
                "max_weight_age": bulk_query["MaxWeightAge"],
                "governance_configuration": bulk_query["SubnetGovernanceConfig"],
                "immunity_period": bulk_query["ImmunityPeriod"],
                "bonds_ma": bulk_query.get("BondsMovingAverage", {}),
                "maximum_set_weight_calls_per_epoch": bulk_query.get("MaximumSetWeightCallsPerEpoch", {}),
                "min_validator_stake": bulk_query.get("MinValidatorStake", {}),
                "max_allowed_validators": bulk_query.get("MaxAllowedValidators", {}),
                "module_burn_config": bulk_query.get("ModuleBurnConfig", {}),
                "metadata": bulk_query.get("SubnetMetadata", {}),
            }

            # Create a dictionary to store the results
            results: dict[int, SubnetParamsWithEmission] = {}

            default_subnet_map = {
                'min_validator_stake': self.to_nanos(50_000),
                'max_allowed_validators': 50,
                'maximum_set_weight_calls_per_epoch': 30,
            }
            subnet_map_keys = list(subnet_maps.keys())
            netuids = list(subnet_maps["name"].keys())
            for _netuid in netuids:
                subnet_result = {k:subnet_maps[k].get(_netuid, default_subnet_map.get(k, None)) for k in subnet_map_keys}
                subnet_result['module_burn_config'] = cast(BurnConfiguration, subnet_result["module_burn_config"])
                results[_netuid] = subnet_result
            c.put(path, results)
        
        results = {int(k):v for k,v in results.items()}
        if subnet != None: 
            subnet = self.get_subnet(subnet)
            print(f"UpdatingSubnet({subnet})")
            results =  results[subnet]

        return results


    def subnets(self, search=None, 
                        features=['name', 'emission', 'metadata', 'tempo', 'founder'] ,
                        max_age=None, update=False) -> pd.DataFrame:
        results =  self.subnet_params( update=update, max_age=max_age)
        if search:
            results = {k:v for k,v in results.items() if search in v['name']}
        results =  c.df(results.values())[features]
        results.sort_values('emission', inplace=True, ascending=False)
        results['emission'] = results['emission'].apply(lambda x: x/10**9 * self.blocks_per_day)
        return results

    def get_path(self, path:str) -> str:
        if not path.startswith(self.folder):
            path = f'{self.folder}/{path}'
        return path

    def global_params(self, max_age=None, update=False) -> NetworkParams:
        """
        Returns global parameters of the whole commune ecosystem
        """

        path = self.get_path(f'{self.network}/global_params')
        print(max_age   , 'max_age')
        result = c.get(path, None, max_age=max_age, update=update)
        if result == None:

            query_all = self.query_batch(
                {
                    "SubspaceModule": [
                        ("MaxNameLength", []),
                        ("MinNameLength", []),
                        ("MaxAllowedSubnets", []),
                        ("MaxAllowedModules", []),
                        ("MaxRegistrationsPerBlock", []),
                        ("MaxAllowedWeightsGlobal", []),
                        ("FloorFounderShare", []),
                        ("MinWeightStake", []),
                        ("Kappa", []),
                        ("Rho", []),
                        ("SubnetImmunityPeriod", []),
                        ("SubnetBurn", []),
                    ],
                    "GovernanceModule": [
                        ("GlobalGovernanceConfig", []),
                        ("GeneralSubnetApplicationCost", []),
                        ("Curator", []),
                    ],
                }
            )
            
            
            global_config = cast(GovernanceConfiguration,query_all["GlobalGovernanceConfig"])
            result: NetworkParams = {
                "max_allowed_subnets": int(query_all["MaxAllowedSubnets"]),
                "max_allowed_modules": int(query_all["MaxAllowedModules"]),
                "max_registrations_per_block": int(query_all["MaxRegistrationsPerBlock"]),
                "max_name_length": int(query_all["MaxNameLength"]),
                "min_weight_stake": int(query_all["MinWeightStake"]),
                "max_allowed_weights": int(query_all["MaxAllowedWeightsGlobal"]),
                "curator": Ss58Address(query_all["Curator"]),
                "min_name_length": int(query_all["MinNameLength"]),
                "floor_founder_share": int(query_all["FloorFounderShare"]),
                "general_subnet_application_cost": int(query_all["GeneralSubnetApplicationCost"]),
                "kappa": int(query_all["Kappa"]),
                "rho": int(query_all["Rho"]),
                "subnet_immunity_period": int(query_all["SubnetImmunityPeriod"]),
                "subnet_registration_cost": int(query_all["SubnetBurn"]),
                "governance_config": {
                    "proposal_cost": int(global_config["proposal_cost"]),
                    "proposal_expiration": int(global_config["proposal_expiration"]),
                    "vote_mode": global_config["vote_mode"],
                    "proposal_reward_treasury_allocation": int(global_config["proposal_reward_treasury_allocation"]),
                    "max_proposal_reward_treasury_allocation": int(global_config["max_proposal_reward_treasury_allocation"]),
                    "proposal_reward_interval": int(global_config["proposal_reward_interval"]),
                },
            }
            c.put(path, result)
        result['min_weight_stake'] = result['min_weight_stake']/10**9
        result['general_subnet_application_cost'] = result['general_subnet_application_cost']/10**9
        result['subnet_registration_cost'] = result['subnet_registration_cost']/10**9
        result['governance_config']['proposal_cost'] = result['governance_config']['proposal_cost']/10**9
        result['governance_config']['proposal_reward_treasury_allocation'] = result['governance_config']['proposal_reward_treasury_allocation']/10**9
        result['governance_config']['max_proposal_reward_treasury_allocation'] = result['governance_config']['max_proposal_reward_treasury_allocation']//10**9
        return result

    def founders(self):
        return self.query_map("Founder", module="SubspaceModule")
    
    
    def my_subnets(self, update=False):
        subnet2params = self.subnet_params(update=update)
        address2key = c.address2key()
        results = []
        for netuid,params in subnet2params.items():
            if params['founder'] in address2key:
                row =  {'name': params['name'], 
                        'netuid': netuid,  
                        'founder': address2key[params['founder']],
                        'tempo': params['tempo'],
                        'emission': params['emission'],
                        'metadata': params['metadata'],
                        }
                results += [row]
        # group by founder
        return c.df(results).sort_values('name')
        
    def mynets(self, update=False):
        return self.my_subnets(update=update)
    
    def my_modules(self, subnet=0, 
                   max_age=None, 
                   keys=None, 
                   features=['name', 'key', 'url',  'stake'],
                   df = False, 
                   update=False):
        if subnet == None:
            modules = []
            for sn, ks in self.key_map().items():
                sn_modules = self.my_mods(subnet=sn, keys=ks, df=False)
                for m in sn_modules:
                    m['subnet'] = sn
                    modules += [m]
            if df:
                modules =  c.df(modules)
                # modules = modules.groupb('key').agg(list).reset_index()
                # modules['stake'] = modules['stake'].apply(sum)
        else:
            subnet = self.get_subnet(subnet)
            path = self.get_path(f'my_modules/{self.network}/{subnet}')
            modules = c.get(path, None, max_age=max_age, update=update)
            if modules == None:
                keys = c.address2key().keys()
                modules = self.mods(keys=keys, subnet=subnet, features=features)
            
        return modules

    mymods = my_modules
    

    def valis(self, 
              subnet=0, 
              update=False,
              df=True,
              search=None,
              min_stake=50000,
              features=['name', 'key', 'stake_from', 'weights'],
              **kwargs):
         
        valis =  self.mods(subnet=subnet, features=features, update=update, **kwargs)
        
        if search != None:
            valis = [v for v in valis if search in v['name'] or search in v['key'] ]
        for i in range(len(valis)):
            v = valis[i]
            v['stake'] =   round(sum((v.get('stake_from', {}) or {}).values()) / 10**9, 2)
            valis[i] = v
            
        valis = [v for v in valis if v['stake'] >= min_stake]
        valis = sorted(valis, key=lambda x: x["stake"], reverse=True)
        if search != None:
            valis = [v for v in valis if search in v['name'] or search in v['key'] ]

        if  df:
            valis = c.df(valis)

        return valis

    name2storage_exceptions = {'key': 'Keys'}
    storage2name_exceptions = {v:k for k,v in name2storage_exceptions.items()}
    def storage2name(self, name):
        new_name = ''
        if name in self.storage2name_exceptions:
            return self.storage2name_exceptions[name]
        for i, ch in enumerate(name):
            if ch == ch.upper():
                ch = ch.lower() 
                if i > 0:
                    ch = '_' + ch
            new_name += ch
        return new_name
    
    def name2storage(self, name, name_map={'url': 'address'}):
        name = name_map.get(name, name)
        new_name = ''
        next_char_upper = False
        if name in self.name2storage_exceptions:
            return self.name2storage_exceptions[name]
        for i, ch in enumerate(name):
            if next_char_upper:
                ch = ch.upper()
                next_char_upper = False
            if ch == '_':
                next_char_upper = True
                ch = ''
            if i == 0:
                ch = ch.upper()
            new_name += ch
        return new_name
                
    def modules(self,
                    subnet=2,
                    max_age = tempo,
                    update=False,
                    timeout=30,
                    module = "SubspaceModule", 
                    features = ['key', 'url', 'name', 'weights'],
                    lite = True,
                    num_connections = 1,
                    search=None,
                    df = False,
                    keys = None,
                    **kwargs):
        subnet = self.get_subnet(subnet)
        og_features = features.copy()
        if 'stake' in og_features:
            features += ['stake_from']
            features.remove('stake')
        subnet_path = self.get_path(f'{self.network}/modules/{subnet}')
        feature2path = {f:subnet_path + '/' + f for f in features}
        future2feature = {}
        params = [subnet] if subnet != None else []
        results  = {}
        for feature in features:
            results[feature] = c.get(feature2path[feature], None, max_age=max_age, update=update)
            if results[feature] == None:
                storage_name = self.name2storage(feature)
                params = [] if feature in ['stake_from'] else ([subnet] if subnet != None else []) 
                fn_obj = self.query if bool(feature in ['incentive', 'dividends', 'emission']) else  self.query_map 
                future = c.submit(fn_obj, kwargs=dict(name=storage_name, params=params), timeout=timeout)
                future2feature[future] = feature
                
        progress = c.tqdm(total=len(future2feature))
        for future in c.as_completed(future2feature, timeout=timeout):
            feature = future2feature.pop(future)
            results[feature] = future.result()
            c.put(feature2path[feature], results[feature])
            progress.update(1)

        # process
        results = self.process_results(results)
        modules = []
        for uid in results['key'].keys():  
            m = {'key':  results['key'][uid]}       
            for f in features:
                if f == 'key':
                    continue
                m[f] = None
                if isinstance(results[f], dict):
                    if uid in results[f]:
                        m[f] = results[f][uid]
                    if m['key'] in results[f]:
                        m[f] = results[f][m['key']]
                elif isinstance(results[f], list):
                    m[f] = results[f][uid] 

                if f == 'weights':
                    if m[f] == None:
                        m[f] = []
            if 'stake' in og_features:
                m['stake'] = sum([v / 10**9 for k,v in (m['stake_from'] or {}).items() ])

            modules.append(m)  
        if search:
            modules = [m for m in modules if search in m['name']]
        if df:
            modules = c.df(modules)
        for i,m in enumerate(modules):
            modules[i] = {k:m[k] for k in features}
        if keys != None:
            modules = [m for m in modules if m['key'] in keys]
        return modules

    mods = modules

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
    
    def keys(self, subnet=0, update=False) -> List[str]:
        subnet = self.get_subnet(subnet)
        keys = self.query_map('Keys', params=[subnet], update=update)
        return [v for k,v in keys.items() if v]

    
    def key_map(self,update=False):
        key_map = self.query_map('Keys', params=[], update=update)
        return {int(k):list(v.values()) for k,v in key_map.items()}

    def key2uid(self, subnet=0) -> int:
        subnet = self.get_subnet(subnet)
        return {v:k for k,v in self.query_map('Keys', params=[subnet]).items()}

    def uid2key(self,subnet=0) -> int:
        key2uid = self.key2uid(subnet)
        return {v:k for k,v in key2uid.items()}  
    
    def is_registered(self, key='module', subnet=2,update=True) -> bool:
        """
        is the key registererd
        """
        key = self.get_key_address(key)
        key_map = self.key_map(update=update)
        if subnet== None:
            for keys in key_map.values():
                if key in keys:
                    return True
        else:

            subnet = self.get_subnet(subnet)
            keys = self.keys(subnet=subnet, update=update)
            if key in keys:
                return True
        return False
    reged = is_registered

    def is_any_registered(self, key=None, update=False) -> bool:
        """
        is any subnet regisrteerd
        """
        return self.is_registered(key=key, subnet=None, update=update)


    def registered_subnets(self, key=None):
        """return the registered subnets"""
        key = self.get_key_address(key)
        key_map = self.key_map(subnet=None, update=update)
        subnets = []
        for subnet, keys in key_map.items():
            if key in keys:
                subnets.append(subnet)
        return subnets




    def is_name_registered(self, name:str, subnet=0, update=False) -> bool: 
        name2key = self.name2key(subnet=subnet, update=update)
        return bool(name2key.get(name, None))

    def get_unique_name(self, 
                        name:str, 
                        subnet=0, 
                        suffix_key_length=4,
                        update=False) -> str:
        """
        Returns a unique name for the given name in the specified subnet.
        If the name is already registered, it appends a number to make it unique.
        """
        name = name.strip()
        suffix = ''
        key_address = self.get_key(name).key_address
        name2key = self.name2key(subnet=subnet, update=update)
        while name in name2key:
            if len(suffix) > 0 :
                name = '::'.join(name.split('::')[:-1])
            suffix = key_address[:suffix_key_length]
            name = f"{name}::{suffix}"
            suffix_key_length += 1
        return name

    def module(self, 
                   module, 
                   subnet=2,
                   fmt='j', 
                   mode = 'https', 
                   block = None, 
                   **kwargs ) -> 'ModuleInfo':
        url = self.get_url( mode=mode)
        subnet = self.get_subnet(subnet)
        module = self.get_key_address(module)
        module = requests.post(url, 
                               json={'id':1, 
                                     'jsonrpc':'2.0',  
                                     'method': 'subspace_getModuleInfo', 
                                     'params': [module, subnet]}
                               ).json()
        module = {**module['result']['stats'], **module['result']['params']}
        module['name'] = self.vec82str(module['name'])
        module['url'] = self.vec82str(module.pop('address'))
        module['dividends'] = module['dividends'] / U16_MAX
        module['incentive'] = module['incentive'] / U16_MAX
        module['stake_from'] = {k:self.format_amount(v, fmt=fmt) for k,v in module['stake_from'].items()}
        module['stake'] = sum([v for k,v in module['stake_from'].items() ])
        module['emission'] = self.format_amount(module['emission'], fmt=fmt)
        module['key'] = module.pop('controller', None)
        module['metadata'] = self.vec82str(module.pop('metadata', []))
        module['vote_staleness'] = (block or self.block()) - module['last_update']
        return module

    mod = module
    
    @staticmethod
    def vec82str(x):
        x = x or []
        return ''.join([chr(ch) for ch in x]).strip()

    def netuids(self,  update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet( update=update, block=block).keys())

    def emissions(self, **kwargs ) -> Dict[str, str]:
        subnets = self.subnets(**kwargs)[['name', 'emission']]
        return  dict(sorted(emissions.items(), key=lambda x: x[1], reverse=True))

    def subnet2netuid(self, **kwargs ) -> Dict[str, str]:
        return {v:int(k) for k,v in self.netuid2subnet(**kwargs).items()}

    name2netuid = subnet2netuid

    def netuid2subnet(self, update=False, block=None, max_age=None) -> Dict[int, str]:
        path = self.get_path(f'{self.network}/netuid2subnet')
        netuid2subnet = c.get(path, None, update=update, max_age=max_age)
        if netuid2subnet == None:
            netuid2subnet = self.query_map("SubnetNames", extract_value=False, block=block)
            c.put(path, netuid2subnet)
        
        netuid2subnet = {int(k):v for k,v in netuid2subnet.items()}

        return netuid2subnet
    
    def miners(self, subnet=0, max_age=None, update=False):
        return self.mods(subnet=subnet, max_age=max_age, update=update)
    
    def stats(self, subnet=0, max_age=None, update=False):
        modules =  c.df(self.mods(subnet=subnet, max_age=max_age, update=update))

        return modules
    
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

    def wallets(self,  update=False, max_age=None, mode='df'):
        """
        an overview of your wallets
        """
        my_stake = self.my_stake(update=update, max_age=max_age)
        my_balance = self.my_balance(update=update, max_age=max_age)
        key2address = c.key2address()
        wallets = []
        wallet_names = set(key2address.keys())
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
        key: str,
        url: str = 'NA',
        name : Optional[str] = None , 
        metadata: Optional[str] = None, 
        code : Optional[str] = None, # either code or metadata
        subnet: Optional[str] = 2,
        net = None,
        wait_for_finalization = False,
        public = False,
        stake = 0,
        safety = False,
        funder: Optional[Union[str, Keypair]] = None,
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

        network = self.get_subnet_name(net or subnet)
        name = name or key
        module_address =  c.get_key(key)
        name = self.get_unique_name(name, subnet=subnet)
        url = url or self.get_module_url(name, public=public)

        params = {
            "network_name": network,
            "address":  url,
            "name": name,
            "module_key": c.get_key(key or name).ss58_address,
            "metadata": metadata or code or 'NA',
        }

        funder = c.get_key(funder or self.funder())
        return  self.call("register", params=params, key=funder, wait_for_finalization=wait_for_finalization, safety=safety)


    def register_vali(self, key: str,funder=None, buffer=10):
        funder = funder or self.funder()
        if not self.is_registered(key=key, subnet=None):
            try:
                self.register(key=key,funder=funder)
            except ChainTransactionError as e:
                if 'already registered' in str(e):
                    print(f'Module {key} is already registered.')
                else:
                    raise e
        vali_stake = self.get_stake(key=key, update=True)
        min_stake = self.min_stake(update=True)
        key_address = self.get_key_address(key)
        if vali_stake < min_stake:
            funder_balance = self.balance(funder)
            remaining_stake = (min_stake - vali_stake) + buffer
            assert funder_balance >= remaining_stake, f'Funder {funder} has insufficient balance {funder_balance} < {remaining_stake}'
            self.stake(
                key=funder,
                amount=remaining_stake,
                dest=key_address,
            )
        return self.register(key=key, subnet=0, funder=funder)

    def deregister(self, key: Keypair, subnet: int=2) -> ExtrinsicReceipt:
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
        key = self.get_key(key)
        subnet = self.get_subnet(subnet)
        return self.call("deregister", params={"netuid": subnet}, key=key)
    
    def dereg(self, key: Keypair, subnet: int=0):
        return self.deregister(key=key, subnet=subnet)


    def funder_path(self, network=None) -> str:
        """
        Returns the path to the funder key for the specified network.
        """
        network = network or self.network
        return f'{self.folder}/{network}/funder'

    def funder(self):
        """
        Returns the funder key for the current network.
        """
        path = self.funder_path()
        funder = c.get(path, None)
        if funder is None:
            self.set_funder('module')            
        return c.get(path, None)
    def funder_key(self):
        """
        Returns the funder key for the current network.
        """           
        return c.get_key(self.funder())

    def set_funder(self, key: str) -> ExtrinsicReceipt:

        assert c.key_exists(key), f'Key {key} not found in key2address'
        return  c.put(self.funder_path(), key)

    def funder_balance(self, funder=None):
        funder = funder or self.funder()
        return self.balance(funder) 

    def fund(self, key_name='test', amount=10.0, subnet=0, funder=None, safety=False):
        balance = self.balance(key_name)
        funder = funder or self.funder()

        if balance < amount:
            amount = amount - balance
            tx = self.transfer(funder, amount, key_name, safety=safety)

        final_balance = self.balance(key_name)
        assert final_balance >= amount, f'Final balance {final_balance} is less than expected {amount}'
        return {'msg': f'Balance ensured for {key_name}: {final_balance}', 'success': True, 'balance': final_balance}



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

    def my_valis(self, subnet=0, min_stake=0, features=['name', 'key','weights', 'stake']):
        return c.df(self.my_mods(subnet, features=features ))

    def my_keys(self, subnet=0):
        return [m['key'] for m in self.my_mods(subnet)]

    def name2key(self, subnet=0, **kwargs) -> dict[str, str]:
        """
        Returns a mapping of names to keys for the specified subnet.
        """
        modules = self.mods(subnet=subnet, features=['name', 'key'], **kwargs)
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
            assert len(idx2key_options) > 0, f'No module key found with enough stake to unstake {amount}'
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
        safety: bool = False,
        existential_amount = 10

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
        if amount == 'all':
            amount = self.balance(key) - existential_amount      
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

    def events(self, block=None, back=None, from_block = None, to_block=None) -> list[dict[str, Any]]:
        """
        Get events from a specific block or the latest block
        """
        if back != None or from_block != None or to_block != None:
            if from_block is not None and to_block is not None:
                block = to_block
                since = from_block
            elif back is not None:
                block = self.block()
                since = block - back
            else:
                raise ValueError("Must specify either 'back' or 'from_block' and 'to_block'")
            assert since < block, f"Block {block} is not greater than since {since}"
            block2events  = {}
            future2block = {}
            for block in range(since, block + 1):
                path = self.get_path(f'events/{block}')
                events = c.get(path, None)
                if events is not None:
                    c.print(f"Events for block {block} already cached, returning cached events", color='green')
                    block2events[block] = events
                    continue
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

        c.put(path, events)
        return events


    def transfer_events(self, block=None, back=None):
        """
        Get transfer events from the blockchain
        
        Args:
            block: Specific block number to check (if None, uses latest)
            back: Number of blocks to look back from current block
        
        Returns:
            List of transfer events
        """
        # Get all events
        events = self.events(block=block, back=back)
        
        transfer_events = []
        
        # If back is specified, we get a dict of block->events
        if back is not None:
            for block_num, block_events in events.items():
                for event in block_events:
                    # Check if this is a transfer event from Balances module
                    if (event.get('module_id') == 'Balances' and 
                        event.get('event_id') == 'Transfer'):
                        event['block'] = block_num
                        transfer_events.append(event)
        else:
            # Single block events
            for event in events:
                if (event.get('module_id') == 'Balances' and 
                    event.get('event_id') == 'Transfer'):
                    event['block'] = block
                    transfer_events.append(event)
        
        return transfer_events