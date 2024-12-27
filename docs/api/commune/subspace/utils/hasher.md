# hasher

Helper functions used to calculate keys for Substrate storage items

Source: `commune/subspace/utils/hasher.py`

## Classes

### blake2b

Return a new BLAKE2b hash object.

## Functions

### `blake2_128(data)`

Helper function to calculate a 16 bytes Blake2b hash for provided data, used as key for Substrate storage items

Parameters
----------
data

Returns
-------

### `blake2_128_concat(data)`

Helper function to calculate a 16 bytes Blake2b hash for provided data, concatenated with data, used as key
for Substrate storage items

Parameters
----------
data

Returns
-------

### `blake2_256(data)`

Helper function to calculate a 32 bytes Blake2b hash for provided data, used as key for Substrate storage items

Parameters
----------
data

Returns
-------

### `identity(data)`



### `two_x64_concat(data)`

Helper function to calculate a xxh64 hash with concatenated data for provided data,
used as key for several Substrate

Parameters
----------
data

Returns
-------

### `xxh128(data)`

Helper function to calculate a 2 concatenated xxh64 hash for provided data, used as key for several Substrate

Parameters
----------
data

Returns
-------

### `xxh64(data)`



