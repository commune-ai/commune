
"""
A client for interacting with Commune network nodes, querying storage,
submitting transactions, etc.

Attributes:
    wait_for_finalization: Whether to wait for transaction finalization.

Example:
```py
client = Subspace()
client.query(name='function_name', params=['param1', 'param2'])
```
"""
import requests
import json
import queue
import re
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Mapping, TypeVar, cast, List, Dict, Optional
from collections import defaultdict
from commune.network.substrate.storage import StorageKey
from commune.network.substrate import (ExtrinsicReceipt, 
                               Keypair, 
                               SubstrateInterface)# type: ignore
from commune.network.subspace.types import (ChainTransactionError,
                                    NetworkQueryError, 
                                    SubnetParamsMaps, 
                                    SubnetParamsWithEmission,
                                    BurnConfiguration, 
                                    GovernanceConfiguration,
                                    Ss58Address,  
                                    NetworkParams, 
                                    SubnetParams, 
                                    Chunk )
import commune as c
# TODO: InsufficientBalanceError, MismatchedLengthError etc

U16_MAX = 2**16 - 1
MAX_REQUEST_SIZE = 9_000_000
IPFS_REGEX = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

class Subspace(c.Module):


    url_map = {
        "main": [ "api.communeai.net"],
        "test": [ "testnet.api.communeai.net"]
    }

    wait_for_finalization: bool
    _num_connections: int
    _connection_queue: queue.Queue[SubstrateInterface]
    url: str

    def __init__(
        self,
        network='test',
        url: str = None,
        mode = 'wss',
        num_connections: int = 1,
        wait_for_finalization: bool = False,
        timeout: int | None = None,
    ):
        """
        Args:
            url: The URL of the network node to connect to.
            num_connections: The number of websocket connections to be opened.
        """
        self.set_network(network=network, mode=mode,url=url,  
                         num_connections=num_connections,  
                         wait_for_finalization=wait_for_finalization, 
                         timeout=timeout)


    def set_network(self, 
                        network,
                        url: str = None,
                        mode = 'wss',
                        num_connections: int = 1,
                        wait_for_finalization: bool = False,
                        timeout: int | None = None ):
        self._num_connections = num_connections
        self.wait_for_finalization = wait_for_finalization
        self._connection_queue = queue.Queue(num_connections)
        self.network = network
        self.url   = url or (mode + '://' + c.choice(self.url_map.get(network, [])))
        ws_options: dict[str, int] = {}
        if timeout != None:
            ws_options["timeout"] = timeout
        self.ws_options = ws_options
        for _ in range(num_connections):
            self._connection_queue.put(
                SubstrateInterface(self.url, ws_options=ws_options)
            )

    def get_url(self, 
                    mode='wss', 
                    **kwargs):
        return f'{mode}://'+c.choice(self.url_map[self.network])
    

    @property
    def connections(self) -> int:
        """
        Gets the maximum allowed number of simultaneous connections to the
        network node.
        """
        return self._num_connections

    @contextmanager
    def get_conn(self, timeout: float | None = None, init: bool = False):
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
        conn = self._connection_queue.get(timeout=timeout)
        if init:
            conn.init_runtime()  # type: ignore
        try:
            if conn.websocket and conn.websocket.connected:  # type: ignore
                yield conn
            else:
                conn = SubstrateInterface(self.url, ws_options=self.ws_options)
                yield conn
        finally:
            self._connection_queue.put(conn)

    def get_storage_keys(
        self,
        storage: str,
        queries: list[tuple[str, list[Any]]],
        block_hash: str | None,
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
        results: list[str | dict[Any, Any]] = []
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
            futures: list[Future[list[str | dict[Any, Any]]]] = []
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
            futures: list[Future[list[str | dict[Any, Any]]]] = []
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

        def get_item_key_value(item_key: tuple[Any, ...] | Any) -> tuple[Any, ...] | Any:
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
        self, functions: dict[str, list[tuple[str, list[Any]]]]
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
        block_hash: str | None = None,
        path = None,
        max_age=None,
        update=False,
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
        if path != None:
            return self.get(path, max_age=max_age, update=update)
        multi_result: dict[str, dict[Any, Any]] = {}

        def recursive_update(
            d: dict[str, dict[T1, T2] | dict[str, Any]],
            u: Mapping[str, dict[Any, Any] | str],
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
                storage_result = self._decode_response(response, chunk_info.fun_params, chunk_info.prefix_list, block_hash)
                multi_result = recursive_update(multi_result, storage_result)

        results =  self.process_results(multi_result)
        if path != None:
            print('Saving results to -->', path)
            self.put(path, results)
        return results
            

    def block_hash(self):
        with self.get_conn(init=True) as substrate:
            block_hash = substrate.get_block_hash()
        return block_hash
    
    def block(self):
        with self.get_conn(init=True) as substrate:
            block_number = substrate.get_block_number()
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
        result = self.query_batch({module: [(name, params)]})
        return result[name]

    def query_map(
        self,
        name: str,
        params: list[Any] = [],
        module: str = "SubspaceModule",
        extract_value: bool = True,
        max_age=0,
        update=False,
        block_hash: str | None = None,
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

        path =  f'{self.network}/query_map/{module}/{name}_params={params}'
        result = self.get(path, None, max_age=max_age, update=update)
        if result == None:
            
            print(f'Updating {path}')

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
            self.put(path, result)

        return self.process_results(result)


    def process_results(self, x:dict) -> dict:
        new_x = {}
        for k in list(x.keys()):
            if type(k) in  [tuple, list]:
                self.dict_put(new_x, k, x[k])
            if isinstance(x[k], dict):
                new_x[k] = self.process_results(x[k])
            if isinstance(k, str) and c.is_int(k):
                new_x[int(k)] = x[k]
                del x[k]
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


    def compose_call(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair | None,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool | None = True,
        sudo: bool = False,
        tip = 0,
        nonce=None,
        unsigned: bool = False,
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
        c.print(f'{module}::{fn} --> {params} network:{self.network} url:{self.url})')
        key = self.resolve_key(key)

        if key is None and not unsigned:
            raise ValueError("Key must be provided for signed extrinsics.")

        with self.get_conn() as substrate:
            if wait_for_finalization is None:
                wait_for_finalization = self.wait_for_finalization

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

            if not unsigned:
                extrinsic = substrate.create_signed_extrinsic(  # type: ignore
                    call=call, 
                    keypair=key, # type: ignore
                    nonce=nonce,
                    tip=tip
                )  # type: ignore
            else:
                extrinsic = substrate.create_unsigned_extrinsic(call=call)  # type: ignore

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
            
        if wait_for_finalization:
            response.process_events()
            if response.is_success:
                response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.network} with key {key.ss58_address}'}
            else:
                response =  {'success': False, 'error': response.error_message, 'msg': f'Failed to call {module}.{fn} on {self.network} with key {key.ss58_address}'}
        return response

    def compose_call_multisig(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair,
        signatories: list[Ss58Address],
        threshold: int,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool | None = None,
        sudo: bool = False,
        era: dict[str, int] | None = None,
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

            # modify the rpc methods at runtime, to allow for correct payment
            # fee calculation parity has a bug in this version,
            # where the method has to be removed
            rpc_methods = substrate.config.get("rpc_methods")  # type: ignore

            if "state_call" in rpc_methods:  # type: ignore
                rpc_methods.remove("state_call")  # type: ignore

            # create the multisig account
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

    def transfer(
        self,
        key: Keypair,
        dest: Ss58Address,
        amount: int,
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

        dest = self.resolve_key_address(dest)
        params = {"dest": dest, "value": amount}

        return self.compose_call(
            module="Balances", fn="transfer_keep_alive", params=params, key=key
        )

    def transfer_multiple(
        self,
        key: Keypair,
        destinations: list[Ss58Address],
        amounts: list[int],
        netuid: str | int = 0,
    ) -> ExtrinsicReceipt:
        """
        Transfers specified amounts of tokens from the signer's account to
        multiple target accounts.

        The `destinations` and `amounts` lists must be of the same length.

        Args:
            key: The keypair associated with the sender's account.
            destinations: A list of SS58 addresses of the recipients.
            amounts: Amount to transfer to each recipient, in nanotokens.
            netuid: The network identifier.

        Returns:
            A receipt of the transaction.

        Raises:
            InsufficientBalanceError: If the sender's account does not have
              enough balance for all transfers.
            ChainTransactionError: If the transaction fails.
        """

        assert len(destinations) == len(amounts)

        # extract existential deposit from amounts
        existential_deposit = self.existential_deposit()
        amounts = [a - existential_deposit for a in amounts]

        params = {
            "netuid": netuid,
            "destinations": destinations,
            "amounts": amounts,
        }

        return self.compose_call(
            module="SubspaceModule", fn="transfer_multiple", params=params, key=key
        )

    def stake(
        self,
        key: Keypair,
        amount: int,
        dest: Ss58Address,
    ) -> ExtrinsicReceipt:
        """
        Stakes the specified amount of tokens to a module key address.

        Args:
            key: The keypair associated with the staker's account.
            amount: The amount of tokens to stake, in nanotokens.
            dest: The SS58 address of the module key to stake to.
            netuid: The network identifier.

        Returns:
            A receipt of the staking transaction.

        Raises:
            InsufficientBalanceError: If the staker's account does not have
              enough balance.
            ChainTransactionError: If the transaction fails.
        """

        params = {"amount": amount, "module_key": dest}

        return self.compose_call(fn="add_stake", params=params, key=key)

    def unstake(
        self,
        key: Keypair,
        dest: Ss58Address,
        amount: int,
    ) -> ExtrinsicReceipt:
        """
        Unstakes the specified amount of tokens from a module key address.

        Args:
            key: The keypair associated with the unstaker's account.
            amount: The amount of tokens to unstake, in nanotokens.
            dest: The SS58 address of the module key to unstake from.
            netuid: The network identifier.

        Returns:
            A receipt of the unstaking transaction.

        Raises:
            InsufficientStakeError: If the staked key does not have enough
              staked tokens by the signer key.
            ChainTransactionError: If the transaction fails.
        """

        params = {"amount": amount, "module_key": dest}
        return self.compose_call(fn="remove_stake", params=params, key=key)

    def update_module(
        self,
        name: str,
        address: str,
        metadata: str | None = None,
        delegation_fee: int = 20,
        netuid: int = 0,
        key: Keypair = None,

    ) -> ExtrinsicReceipt:
        """
        Updates the parameters of a registered module.

        The delegation fee must be an integer between 0 and 100.

        Args:
            key: The keypair associated with the module's account.
            name: The new name for the module. If None, the name is not updated.
            address: The new address for the module.
                If None, the address is not updated.
            delegation_fee: The new delegation fee for the module,
                between 0 and 100.
            netuid: The network identifier.

        Returns:
            A receipt of the module update transaction.

        Raises:
            InvalidParameterError: If the provided parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """

        assert isinstance(delegation_fee, int)
        key = key or name
        
        params = {
            "netuid": netuid,
            "name": name,
            "address": address,
            "delegation_fee": delegation_fee,
            "metadata": metadata,
        }

        response = self.compose_call("update_module", params=params, key=key)

        return response

    def register(
        self,
        name: str,
        subnet: str = "Rootnet",
        address: str | None = None,
        module_key = None, 
        key: Keypair = None,
        metadata: str | None = 'NA',
        wait_for_finalization = True,
    ) -> ExtrinsicReceipt:
        """
        Registers a new module in the network.

        Args:
            key: The keypair used for registering the module.
            name: The name of the module.
            address: The address of the module. 
            key_address : The ss58_address of the module
            subnet: The network subnet to register the module in.
                If None, a default value is used.
        """
        key =  c.get_key(key)
        params = {
            "network_name": subnet,
            "address":  address or c.namespace().get(name, '0.0.0.0:8888' ),
            "name": name,
            "module_key":c.get_key(module_key or name, creaet_if_not_exists=True).ss58_address,
            "metadata": metadata,
        }
        response =  self.compose_call("register", params=params, key=key, wait_for_finalization=wait_for_finalization)
        return response

    def deregister(self, key: Keypair, netuid: int) -> ExtrinsicReceipt:
        """
        Deregisters a module from the network.

        Args:
            key: The keypair associated with the module's account.
            netuid: The network identifier.

        Returns:
            A receipt of the module deregistration transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        params = {"netuid": netuid}

        response = self.compose_call("deregister", params=params, key=key)

        return response

    def register_subnet(self, name: str, metadata: str | None = None,  key: Keypair=None) -> ExtrinsicReceipt:
        """
        Registers a new subnet in the network.

        Args:
            key (Keypair): The keypair used for registering the subnet.
            name (str): The name of the subnet to be registered.
            metadata (str | None, optional): Additional metadata for the subnet. Defaults to None.

        Returns:
            ExtrinsicReceipt: A receipt of the subnet registration transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        params = {
            "name": name,
            "metadata": metadata,
        }


        response = self.compose_call("register_subnet", params=params, key=key)
        return response

    def vote(
        self,
        key: Keypair,
        uids: list[int],
        weights: list[int],
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Casts votes on a list of module UIDs with corresponding weights.

        The length of the UIDs list and the weights list should be the same.
        Each weight corresponds to the UID at the same index.

        Args:
            key: The keypair used for signing the vote transaction.
            uids: A list of module UIDs to vote on.
            weights: A list of weights corresponding to each UID.
            netuid: The network identifier.

        Returns:
            A receipt of the voting transaction.

        Raises:
            InvalidParameterError: If the lengths of UIDs and weights lists
                do not match.
            ChainTransactionError: If the transaction fails.
        """

        assert len(uids) == len(weights)

        params = {
            "uids": uids,
            "weights": weights,
            "netuid": netuid,
        }
        response = self.compose_call("set_weights", params=params, key=key)
        return response

    def update_subnet(
        self,
        key: Keypair,
        params: SubnetParams = None,
        netuid: int = 0,
        **extra_params
    ) -> ExtrinsicReceipt:
        """
        Update a subnet's configuration.

        It requires the founder key for authorization.

        Args:
            key: The founder keypair of the subnet.
            params: The new parameters for the subnet.
            netuid: The network identifier.

        Returns:
            A receipt of the subnet update transaction.

        Raises:
            AuthorizationError: If the key is not authorized.
            ChainTransactionError: If the transaction fails.
        """
        default_params = self.subnet_params(netuid=netuid)
        governance_config = default_params.pop('governance_configuration', None)

        key = c.get_key(key)
        params = params or {}

        params.update(extra_params)
        params = {**default_params, **params}
        params["netuid"] = netuid
        params['vote_mode'] = governance_config['vote_mode']
        params['founder'] = key.ss58_address
        params["metadata"] = params.get("subnet_metadata", None)

        response = self.compose_call(
            fn="update_subnet",
            params=params,
            key=key,
        )

        return response

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
            netuid: The network identifier.

        Returns:
            A receipt of the stake transfer transaction.

        Raises:
            InsufficientStakeError: If the source module key does not have
            enough staked tokens. ChainTransactionError: If the transaction
            fails.
        """

        amount = amount - self.existential_deposit()

        params = {
            "amount": amount,
            "module_key": from_module_key,
            "new_module_key": dest_module_address,
        }

        response = self.compose_call("transfer_stake", key=key, params=params)

        return response

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
            netuid: The network identifier.

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

        response = self.compose_call("remove_stake_multiple", params=params, key=key)

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
            netuid: The network identifier.

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

        response = self.compose_call("add_stake_multiple", params=params, key=key)

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

        response = self.compose_call("add_profit_shares", params=params, key=key)

        return response

    def add_subnet_proposal(
        self, key: Keypair,
        params: dict[str, Any],
        ipfs: str,
        netuid: int = 0
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for creating or modifying a subnet within the
        network.

        The proposal includes various parameters like the name, founder, share
        allocations, and other subnet-specific settings.

        Args:
            key: The keypair used for signing the proposal transaction.
            params: The parameters for the subnet proposal.
            netuid: The network identifier.

        Returns:
            A receipt of the subnet proposal transaction.

        Raises:
            InvalidParameterError: If the provided subnet
                parameters are invalid.
            ChainTransactionError: If the transaction fails.
        """

        general_params = dict(params)
        general_params["netuid"] = netuid
        general_params["data"] = ipfs
        if "metadata" not in general_params:
            general_params["metadata"] = None

        # general_params["burn_config"] = json.dumps(general_params["burn_config"])
        response = self.compose_call(
            fn="add_subnet_params_proposal",
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

        response = self.compose_call(
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
        netuid: int = 0,
    ) -> ExtrinsicReceipt:
        """
        Submits a proposal for creating or modifying a custom subnet within the
        network.

        The proposal includes various parameters like the name, founder, share
        allocations, and other subnet-specific settings.

        Args:
            key: The keypair used for signing the proposal transaction.
            params: The parameters for the subnet proposal.
            netuid: The network identifier.

        Returns:
            A receipt of the subnet proposal transaction.
        """

        params = {
            "data": cid,
            "netuid": netuid,
        }

        response = self.compose_call(
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
        cid: str | None,
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

        response = self.compose_call(
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

        response = self.compose_call(
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

        response = self.compose_call(
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

        response = self.compose_call(
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

        response = self.compose_call(
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

        response = self.compose_call(
            "add_dao_application", module="GovernanceModule", key=key,
            params=params
        )

        return response

    def curator_applications(self) -> dict[str, dict[str, str]]:
        applications = self.query_map(
            "CuratorApplications", module="GovernanceModule", params=[],
            extract_value=False
        )
        return applications

    def proposals(
        self, extract_value: bool = False
    ) -> dict[int, dict[str, Any]]:
        """
        Retrieves a mappping of proposals from the network.

        Queries the network and returns a mapping of proposal IDs to
        their respective parameters.

        Returns:
            A dictionary mapping proposal IDs
            to dictionaries of their parameters.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map(
            "Proposals", extract_value=extract_value, module="GovernanceModule"
        )

    def weights(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, list[tuple[int, int]]] | None:
        """
        Retrieves a mapping of weights for keys on the network.

        Queries the network and returns a mapping of key UIDs to
        their respective weights.

        Args:
            netuid: The network UID from which to get the weights.

        Returns:
            A dictionary mapping key UIDs to lists of their weights.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        weights_dict = self.query_map(
            "Weights",
            [netuid],
            extract_value=extract_value
        )
        return weights_dict

    def key(
        self,
        netuid: int = 0,
        extract_value: bool = False,
    ) -> dict[int, Ss58Address]:
        """
        Retrieves a map of keys from the network.

        Fetches a mapping of key UIDs to their associated
        addresses on the network.
        The query can be targeted at a specific network UID if required.

        Args:
            netuid: The network UID from which to get the keys.

        Returns:
            A dictionary mapping key UIDs to their addresses.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """
        return self.query_map("Keys", [netuid], extract_value=extract_value)

    def address(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, str]:
        """
        Retrieves a map of key addresses from the network.

        Queries the network for a mapping of key UIDs to their addresses.

        Args:
            netuid: The network UID from which to get the addresses.

        Returns:
            A dictionary mapping key UIDs to their addresses.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query_map("Address", [netuid], extract_value=extract_value)


    def emission(self, extract_value: bool = False) -> dict[int, list[int]]:
        return self.query_map("Emission", extract_value=extract_value)

    def pending_emission(self, extract_value: bool = False) -> int:
        return self.query_map("PendingEmission", extract_value=extract_value, module="SubnetEmissionModule")


    def state(self, timeout=42):
        futures = []
        fns  = ['subnet_params', 'global_params', 'modules']
        futures = [c.submit(getattr(self,fn), kwargs=dict(update=1), timeout=timeout) for fn in fns]
        return dict(zip(fns, c.wait(futures, timeout=timeout)))
    sync = state

    def sync_loop(self, interval=30):
        while True:
            c.sleep(interval)
            self.sync()
        
    def subnet_emission(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a map of subnet emissions for the network.
        """
        return self.query_map("SubnetEmission", extract_value=extract_value, module="SubnetEmissionModule")

    def subnet_consensus(self, extract_value: bool = False) -> dict[int, str]:
        """
        Retrieves a map of subnet consensus types for the network.
        """

        return self.query_map("SubnetConsensusType", extract_value=extract_value, module="SubnetEmissionModule")

    def incentive(self, extract_value: bool = False) -> dict[int, list[int]]:
        """
        Retrieves a mapping of incentives for keys on the network.
        """

        return self.query_map("Incentive", extract_value=extract_value)

    def dividends(self, extract_value: bool = False) -> dict[int, list[int]]:
        """
        Retrieves a mapping of dividends for keys on the network.
        """

        return self.query_map("Dividends", extract_value=extract_value)

    def regblock(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of registration blocks for keys on the network.
        """

        return self.query_map(
            "RegistrationBlock", [netuid], extract_value=extract_value
        )

    def lastupdate(self, extract_value: bool = False) -> dict[int, list[int]]:
        """
        Retrieves a mapping of the last update times for keys on the network.
        """

        return self.query_map("LastUpdate", extract_value=extract_value)

    def stakefrom(
        self, extract_value: bool = False, fmt='j'
    ) -> dict[Ss58Address, list[tuple[Ss58Address, int]]]:
        """
        Retrieves a mapping of stakes from various sources for keys on the network.
        """

        result = self.query_map("StakeFrom", [], extract_value=extract_value)

        return self.format_amount(result, fmt=fmt)

    def staketo( self, extract_value: bool = False, fmt='j', **kwargs,  ) -> dict[Ss58Address, list[tuple[Ss58Address, int]]]:
        """
        Retrieves a mapping of stakes to destinations for keys on the network.
        """

        result = self.query_map("StakeTo", [], extract_value=extract_value, **kwargs)
        return self.format_amount(result, fmt=fmt)

    def delegationfee(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[str, int]:
        """
        Retrieves a mapping of delegation fees for keys on the network.
        """

        return self.query_map("DelegationFee", [netuid], extract_value=extract_value)

    def tempo(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of tempo settings for the network.
        """

        return self.query_map("Tempo", extract_value=extract_value)

    def immunity_period(self, extract_value: bool) -> dict[int, int]:
        """
        Retrieves a mapping of immunity periods for the network.
        """

        return self.query_map("ImmunityPeriod", extract_value=extract_value)

    def min_allowed_weights(
        self, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of minimum allowed weights for the network.
        """

        return self.query_map("MinAllowedWeights", extract_value=extract_value)

    def max_allowed_weights(
        self, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of maximum allowed weights for the network.
        """

        return self.query_map("MaxAllowedWeights", extract_value=extract_value)

    def max_allowed_uids(self, extract_value: bool = False) -> dict[int, int]:
        """
        Queries the network for the maximum number of allowed user IDs (UIDs)
        for each network subnet.
        """

        return self.query_map("MaxAllowedUids", extract_value=extract_value)

    def min_stake(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of minimum allowed stake on the network.
        """

        return self.query_map("MinStake", extract_value=extract_value)

    def max_stake(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of the maximum stake values for the network.
        """

        return self.query_map("MaxStake", extract_value=extract_value)

    def founder(self, extract_value: bool = False) -> dict[int, str]:
        """
        Retrieves a mapping of founders for the network.
        """

        return self.query_map("Founder", extract_value=extract_value)

    def founder_share(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of founder shares for the network.
        """

        return self.query_map("FounderShare", extract_value=extract_value)
    def incentive_ratio(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of incentive ratios for the network.
        """

        return self.query_map("IncentiveRatio", extract_value=extract_value)
    def trust_ratio(self, extract_value: bool = False) -> dict[int, int]:
        """
        Retrieves a mapping of trust ratios for the network.
        """

        return self.query_map("TrustRatio", extract_value=extract_value)

    def vote_mode_subnet(self, extract_value: bool = False) -> dict[int, str]:
        """
        Retrieves a mapping of vote modes for subnets within the network.
        """

        return self.query_map("VoteModeSubnet", extract_value=extract_value)

    def legit_whitelist(
        self, extract_value: bool = False
    ) -> dict[Ss58Address, int]:
        """
        Retrieves a mapping of whitelisted addresses for the network.
        """

        return self.query_map( "LegitWhitelist", module="GovernanceModule", extract_value=extract_value)
    endpoints = ['subnet_params']
    def subnet_names(self, extract_value: bool = False, max_age=10) -> dict[int, str]:
        """
        Retrieves a mapping of subnet names within the network.
        """

        return self.query_map("SubnetNames", extract_value=extract_value, max_age=max_age)
    def subnets(self):
        return self.subnet_names()

    def balances(
        self, extract_value: bool = False, block_hash: str | None = None
    ) -> dict[str, dict[str, int | dict[str, int | float]]]:
        """
        Retrieves a mapping of account balances within the network.
        """

        return self.query_map("Account", module="System", extract_value=extract_value, block_hash=block_hash)

    def registration_blocks(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, int]:
        """
        Retrieves a mapping of registration blocks for UIDs on the network.
        """

        return self.query_map(
            "RegistrationBlock", [netuid], extract_value=extract_value
        )

    def name(
        self, netuid: int = 0, extract_value: bool = False
    ) -> dict[int, str]:
        """
        Retrieves a mapping of names for keys on the network.
        """

        return self.query_map("Name", [netuid], extract_value=extract_value)

    #  == QUERY FUNCTIONS == #

    def immunity_period(self, netuid: int = 0) -> int:
        """
        Queries the network for the immunity period setting.
        """

        return self.query(
            "ImmunityPeriod",
            params=[netuid],
        )

    def max_set_weights_per_epoch(self):
        return self.query("MaximumSetWeightCallsPerEpoch")

    def min_allowed_weights(self, netuid: int = 0) -> int:
        """
        Queries the network for the minimum allowed weights setting.
        """

        return self.query(
            "MinAllowedWeights",
            params=[netuid],
        )

    def dao_treasury_address(self) -> Ss58Address:
        return self.query("DaoTreasuryAddress", module="GovernanceModule")

    def max_allowed_weights(self, netuid: int = 0) -> int:
        """
        Queries the network for the maximum allowed weights setting.
        """

        return self.query("MaxAllowedWeights", params=[netuid])

    def max_allowed_uids(self, netuid: int = 0) -> int:
        """
        Queries the network for the maximum allowed UIDs setting.
        """

        return self.query("MaxAllowedUids", params=[netuid])

    def name(self, netuid: int = 0) -> str:
        """
        Queries the network for the name of a specific subnet.
        """

        return self.query("Name", params=[netuid])
    

    def namespace(self, netuid: int = 0) -> Dict[str, str]:
        results =  self.query_batch_map({'SubspaceModule': [('Name', [netuid]), ('Address', [netuid])]})
        names = results['Name']
        addresses = results['Address']
        namespace = {}
        for uid, name in names.items():
            namespace[name] = addresses[uid]
        return namespace

    def subnet_name(self, netuid: int = 0) -> str:
        """
        Queries the network for the name of a specific subnet.
        """
        return self.query("SubnetNames", params=[netuid])

    def global_dao_treasury(self):
        return self.query("GlobalDaoTreasury", module="GovernanceModule")

    def n(self, netuid: int = 0) -> int:
        """
        Queries the network for the 'N' hyperparameter, which represents how
        many modules are on the network.
        """

        return self.query("N", params=[netuid])

    def tempo(self, netuid: int = 0) -> int:
        """
        Queries the network for the tempo setting, measured in blocks, for the
        specified subnet.
        """

        return self.query("Tempo", params=[netuid])

    def total_free_issuance(self, block_hash: str | None = None) -> int:
        """
        Queries the network for the total free issuance.
        """

        return self.query("TotalIssuance", module="Balances", block_hash=block_hash)

    def total_stake(self, block_hash: str | None = None) -> int:
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

    def max_registrations_per_block(self, netuid: int = 0):
        """
        Queries the network for the maximum number of registrations per block.
        """

        return self.query(
            "MaxRegistrationsPerBlock",
            params=[netuid],
        )

    def proposal(self, proposal_id: int = 0):
        """
        Queries the network for a specific proposal.
        """

        return self.query(
            "Proposals",
            params=[proposal_id],
        )

    def trust(self, netuid: int = 0):
        """
        Queries the network for the trust setting of a specific network subnet.
        """

        return self.query(
            "Trust",
            params=[netuid],
        )

    def uid(self, key: Ss58Address, netuid: int = 0) -> bool | None:
        """
        Queries the network for module UIDs associated with a specific key.
        """

        return self.query(
            "Uids",
            params=[netuid, key],
        )
    def uids(self,netuid: int = 0) -> bool | None:
        """
        Queries the network for module UIDs associated with a specific key.
        """

        return list(self.query_map(
            "Keys",
            params=[netuid],
        ).keys())

    def unit_emission(self) -> int:
        """
        Queries the network for the unit emission setting.
        """

        return self.query("UnitEmission", module="SubnetEmissionModule")

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

        return self.query(
            "SubnetBurn",
        )

    def burn_rate(self) -> int:
        """
        Queries the network for the burn rate setting.
        """

        return self.query(
            "BurnRate",
            params=[],
        )

    def burn(self, netuid: int = 0) -> int:
        """
        Queries the network for the burn setting.
        """

        return self.query("Burn", params=[netuid])

    def min_burn(self) -> int:
        """
        Queries the network for the minimum burn setting.
        """

        return self.query(
            "BurnConfig",
            params=[],
        )["min_burn"]

    def min_weight_stake(self) -> int:
        """
        Queries the network for the minimum weight stake setting.
        """

        return self.query("MinWeightStake", params=[])

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

    def max_registrations_per_block(self) -> int:
        """
        Queries the network for the maximum number of registrations per block.

        Retrieves the maximum number of registrations that can
        be processed in each block within the network.

        Returns:
            The maximum number of registrations per block on the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxRegistrationsPerBlock",
            params=[],
        )

    def max_name_length(self) -> int:
        """
        Queries the network for the maximum length allowed for names.

        Retrieves the maximum character length permitted for names
        within the network. Such as the module names

        Returns:
            The maximum length allowed for names on the network.

        Raises:
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query(
            "MaxNameLength",
            params=[],
        )

    def global_vote_threshold(self) -> int:
        """
        Queries the network for the global vote threshold.
        """

        return self.query(
            "GlobalVoteThreshold",
        )

    def max_allowed_subnets(self) -> int:
        """
        Queries the network for the maximum number of allowed subnets.
        """

        return self.query(
            "MaxAllowedSubnets",
            params=[],
        )

    def max_allowed_modules(self) -> int:
        """
        Queries the network for the maximum number of allowed modules.
        """

        return self.query(
            "MaxAllowedModules",
            params=[],
        )

    def min_stake(self, netuid: int = 0) -> int:
        """
        Queries the network for the minimum stake required to register a key.
            QueryError: If the query to the network fails or is invalid.
        """

        return self.query("MinStake", params=[netuid])

    def get_stakefrom(
        self,
        key: Ss58Address,
        fmt = 'j',
    ) -> dict[str, int]:
        """
        Retrieves the stake amounts from all stakers to a specific staked address.
        """
        key = self.resolve_key_address(key)
        result = self.query_map("StakeFrom", [key], extract_value=False)
        return self.format_amount(result, fmt=fmt)
    get_stake_from = get_stakefrom
    def get_staketo(
        self,
        key: Ss58Address,
        fmt = 'j'
    ) -> dict[str, int]:
        """
        Retrieves the stake amounts provided by a specific staker to all staked addresses.
        """
        key = self.resolve_key_address(key)
        result =  self.query_map("StakeTo", [key], extract_value=False)
        return self.format_amount(result, fmt=fmt)

    get_stake_to = get_staketo
    def balance(
        self,
        addr: Ss58Address,
        fmt = 'j'
    ) -> int:
        """
        Retrieves the balance of a specific key.
        """

        addr = self.resolve_key_address(addr)
        result = self.query("Account", module="System", params=[addr])
        return self.format_amount(result["data"]["free"], fmt=fmt)

    def block(self) -> dict[Any, Any] | None:
        """
        Retrieves information about a specific block in the network.
        """
        block_hash: str | None = None

        with self.get_conn() as substrate:
            block: dict[Any, Any] | None = substrate.get_block_number(  # type: ignore
                block_hash  # type: ignore
            )

        return block

    def existential_deposit(self, block_hash: str | None = None) -> int:
        """
        Retrieves the existential deposit value for the network.
        """

        with self.get_conn() as substrate:
            result: int = substrate.constant(  #  type: ignore
                "Balances", "ExistentialDeposit", block_hash
            ).value  #  type: ignore

        return result

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

        return self.compose_call(
            module="GovernanceModule",
            fn="add_transfer_dao_treasury_proposal",
            params=params,
            key=key,
        )

    def delegate_rootnet_control(self, key: Keypair, dest: Ss58Address):
        params = {"origin": key, "target": dest}

        return self.compose_call(
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
    
    def resolve_key_address(self, key:str ):
        if self.valid_ss58_address(key):
            return key
        else:
            key = c.get_key( key )
            return key.ss58_address

    def resolve_key(self, key:str ):
        key = c.get_key( key )
        return key


    def subnet_params(self, netuid=None, block_hash: str | None = None, max_age=None, update=False) -> dict[int, SubnetParamsWithEmission]:
        """
        Gets all subnets info on the network
        """
        path = f'{self.network}/subnet_params'
        results = self.get(path,None, max_age=max_age, update=update)
        results = self.process_results(results)

        if results == None or len(results) == 0:
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
                        ("TrustRatio", params),
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
        
            subnet_maps: SubnetParamsMaps = {
                "emission": bulk_query["SubnetEmission"],
                "tempo": bulk_query["Tempo"],
                "min_allowed_weights": bulk_query["MinAllowedWeights"],
                "max_allowed_weights": bulk_query["MaxAllowedWeights"],
                "max_allowed_uids": bulk_query["MaxAllowedUids"],
                "founder": bulk_query["Founder"],
                "founder_share": bulk_query["FounderShare"],
                "incentive_ratio": bulk_query["IncentiveRatio"],
                "trust_ratio": bulk_query["TrustRatio"],
                "name": bulk_query["SubnetNames"],
                "max_weight_age": bulk_query["MaxWeightAge"],
                "governance_configuration": bulk_query["SubnetGovernanceConfig"],
                "immunity_period": bulk_query["ImmunityPeriod"],
                "bonds_ma": bulk_query.get("BondsMovingAverage", {}),
                "maximum_set_weight_calls_per_epoch": bulk_query.get("MaximumSetWeightCallsPerEpoch", {}),
                "min_validator_stake": bulk_query.get("MinValidatorStake", {}),
                "max_allowed_validators": bulk_query.get("MaxAllowedValidators", {}),
                "module_burn_config": bulk_query.get("ModuleBurnConfig", {}),
                "subnet_metadata": bulk_query.get("SubnetMetadata", {}),
            }

            results: dict[int, SubnetParamsWithEmission] = {}

            default_subnet_map = {
                'min_validator_stake': self.to_nano(50_000),
                'max_allowed_validators': 50,
                'maximum_set_weight_calls_per_epoch': 30
            }
            subnet_map_keys = list(subnet_maps.keys())
            netuids = list(subnet_maps["name"].keys())
            for _netuid in netuids:
                subnet = {k:subnet_maps[k].get(_netuid, default_subnet_map.get(k, None)) for k in subnet_map_keys}
                subnet['module_burn_config'] = cast(BurnConfiguration, subnet["module_burn_config"])
                results[_netuid] = subnet
            self.put(path, results)
        if netuid != None: 
            return results[netuid]
        return results

    def global_params(self, max_age=None, update=False) -> NetworkParams:
        """
        Returns global parameters of the whole commune ecosystem
        """
        path = f'{self.network}/global_params'
        result = self.get(path, None, max_age=max_age, update=update)
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
                        ("FloorDelegationFee", []),
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
            
            
            global_config = cast(
                GovernanceConfiguration,
                query_all["GlobalGovernanceConfig"]
            )
            result: NetworkParams = {
                "max_allowed_subnets": int(query_all["MaxAllowedSubnets"]),
                "max_allowed_modules": int(query_all["MaxAllowedModules"]),
                "max_registrations_per_block": int(query_all["MaxRegistrationsPerBlock"]),
                "max_name_length": int(query_all["MaxNameLength"]),
                "min_weight_stake": int(query_all["MinWeightStake"]),
                "floor_delegation_fee": int(query_all["FloorDelegationFee"]),
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
            self.put(path, result)
        return result

    def clean_feature_name(self, x):
        new_x = ''
        for i, ch in enumerate(x):
            if ch == ch.upper():
                ch = ch.lower() 
                if i > 0:
                    ch = '_' + ch
            new_x += ch
        return new_x

    def modules(self,
                    netuid=None,
                    timeout=30,
                    max_age = 100,
                    update=False,
                    module = "SubspaceModule", 
                    features = ['Name', 
                            'Address', 
                            'Keys',
                            'Weights',
                            'Incentive',
                            'Dividends', 
                            'Emission', 
                            'DelegationFee', 
                            'LastUpdate',
                            'Metadata'
                            ],
                    default_module = {
                        'Weights': [], 
                        'Incentive': 0,
                        'Emissions': 0, 
                        'Dividends': 0,
                        'DelegationFee': 30,
                        'LastUpdate': -1,
                    },
                    **kwargs):

        path = f'{self.network}/modules'
        results = self.get(path, None, max_age=max_age, update=update)
        if results == None:
            results = self.query_batch_map({module:[(f, []) for f in features]},self.block_hash())
            results = self.process_results(results)
            self.put(path, results)
        netuids = list(results['Keys'].keys())
        modules = {}
        og_netuid = netuid
        for netuid in netuids:
            modules[netuid] = []
            for uid in results['Keys'][netuid].keys():
                module = {'uid': uid}
                for f in features:
                    module[f] = results[f].get(netuid, {})
                    if isinstance(module[f], dict):
                        module[f] = module[f].get(uid, default_module.get(f, None)) 
                    elif isinstance(module[f], list):
                        module[f] = module[f][uid]
                modules[netuid].append({self.clean_feature_name(k): v for k,v in module.items()})
        if og_netuid != None:
            return modules[og_netuid]
        return modules

    def get_rate_limit(self, address):
        return self.resolve_key_address(address)

    def format_amount(self, x, fmt='nano') :
        if type(x) in [dict]:
            for k,v in x.items():
                x[k] = self.format_amount(v, fmt=fmt)
            return x

        if fmt in ['j', 'com', 'comai']:
            x = x / 10**9
        elif fmt in ['nanos', 'n', 'nj']:
            x = x 
        else:
            raise NotImplementedError(fmt)

        return x

    @property
    def block_number(self) -> int:
        return self.substrate.block_number(block_hash=None)

    @property
    def substrate(self): 
        return SubstrateInterface(self.url, ws_options=self.ws_options)
    @staticmethod
    def vec82str(l:list):
        return ''.join([chr(x) for x in l]).strip()

    def keys(self, netuid = 0 ) -> List[str]:
        return list(self.query_map('Keys', params=[netuid]).values())
    
    def get_module(self, 
                    module,
                    netuid=0,
                    fmt='j',
                    mode = 'https',
                    block = None,
                    **kwargs ) -> 'ModuleInfo':
        url = self.get_url( mode=mode)
        json={'id':1, 'jsonrpc':'2.0',  'method': 'subspace_getModuleInfo', 'params': [module, netuid]}
        module = requests.post(url,  json=json).json()
        module = {**module['result']['stats'], **module['result']['params']}
        module['name'] = self.vec82str(module['name'])
        module['address'] = self.vec82str(module['address'])
        module['dividends'] = module['dividends'] / (U16_MAX)
        module['incentive'] = module['incentive'] / (U16_MAX)
        module['stake_from'] = {k:self.format_amount(v, fmt=fmt) for k,v in module['stake_from'].items()}
        module['stake'] = sum([v for k,v in module['stake_from'].items() ])
        module['emission'] = self.format_amount(module['emission'], fmt=fmt)
        module['key'] = module.pop('controller', None)
        module['metadata'] = module.pop('metadata', {})
        module['vote_staleness'] = (block or self.block()) - module['last_update']
        return module


    def netuids(self,  update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet( update=update, block=block).keys())

    def netuid2subnet(self ) -> Dict[str, str]:
        subnet_names = self.query_map('SubnetNames', [])
        subnet_names = dict(sorted(subnet_names.items(), key=lambda x: x[0]))
        return subnet_names
    
    def subnet2netuid(self ) -> Dict[str, str]:
        subnet2netuid =  {v:k for k,v in self.netuid2subnet().items()}
        return subnet2netuid
    name2netuid = subnet2netuid

    def transform_stake_dmap(self, stake_storage: dict[tuple[Ss58Address, Ss58Address], int]) -> dict[Ss58Address, list[tuple[Ss58Address, int]]]:
        """
        Transforms either the StakeTo or StakeFrom storage into the stake legacy data type.
        """
        transformed: dict[Ss58Address, list[tuple[Ss58Address, int]]] = defaultdict(list)
        [transformed[k1].append((k2, v)) for (k1, k2), v in stake_storage.items()]

        return dict(transformed)