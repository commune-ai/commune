# extensions



Source: `commune/subspace/extensions.py`

## Classes

### Extension

Base class of all extensions

#### Methods

##### `close(self)`

Cleanup process of the extension. This function is being called by the ExtensionRegistry.

Returns
-------

##### `debug_message(self, message: str)`

Submits a debug message in the logger

Parameters
----------
message: str

Returns
-------

Type annotations:
```python
message: <class 'str'>
```

##### `init(self, substrate: 'SubstrateInterface')`

Initialization process of the extension. This function is being called by the ExtensionRegistry.

Parameters
----------
substrate: SubstrateInterface

Returns
-------

Type annotations:
```python
substrate: SubstrateInterface
```

### SearchExtension

Type of `Extension` that implements functionality to improve and enhance search capability

#### Methods

##### `close(self)`

Cleanup process of the extension. This function is being called by the ExtensionRegistry.

Returns
-------

##### `debug_message(self, message: str)`

Submits a debug message in the logger

Parameters
----------
message: str

Returns
-------

Type annotations:
```python
message: <class 'str'>
```

##### `filter_events(self, **kwargs) -> list`

Filters events to match provided search criteria e.g. block range, pallet name, accountID in attributes

Parameters
----------
kwargs

Returns
-------
list

Type annotations:
```python
return: <class 'list'>
```

##### `filter_extrinsics(self, **kwargs) -> list`

Filters extrinsics to match provided search criteria e.g. block range, pallet name, signed by accountID

Parameters
----------
kwargs

Returns
-------

Type annotations:
```python
return: <class 'list'>
```

##### `get_block_timestamp(self, block_number: int) -> int`

Return a UNIX timestamp for given `block_number`.

Parameters
----------
block_number: int The block_number to retrieve the timestamp for

Returns
-------
int

Type annotations:
```python
block_number: <class 'int'>
return: <class 'int'>
```

##### `init(self, substrate: 'SubstrateInterface')`

Initialization process of the extension. This function is being called by the ExtensionRegistry.

Parameters
----------
substrate: SubstrateInterface

Returns
-------

Type annotations:
```python
substrate: SubstrateInterface
```

##### `search_block_number(self, block_datetime: datetime.datetime, block_time: int = 6, **kwargs) -> int`

Search corresponding block number for provided `block_datetime`. the prediction tolerance is provided with
`block_time`

Parameters
----------
block_datetime: datetime
block_time: int
kwargs

Returns
-------
int

Type annotations:
```python
block_datetime: <class 'datetime.datetime'>
block_time: <class 'int'>
return: <class 'int'>
```

### SubstrateNodeExtension

Implementation of `SearchExtension` using only Substrate RPC methods. Could be significant inefficient.

#### Methods

##### `close(self)`

Cleanup process of the extension. This function is being called by the ExtensionRegistry.

Returns
-------

##### `debug_message(self, message: str)`

Submits a debug message in the logger

Parameters
----------
message: str

Returns
-------

Type annotations:
```python
message: <class 'str'>
```

##### `filter_events(self, block_start: int = None, block_end: int = None, pallet_name: str = None, event_name: str = None, account_id: str = None) -> list`

Filters events to match provided search criteria e.g. block range, pallet name, accountID in attributes

Parameters
----------
kwargs

Returns
-------
list

Type annotations:
```python
block_start: <class 'int'>
block_end: <class 'int'>
pallet_name: <class 'str'>
event_name: <class 'str'>
account_id: <class 'str'>
return: <class 'list'>
```

##### `filter_extrinsics(self, block_start: int = None, block_end: int = None, ss58_address: str = None, pallet_name: str = None, call_name: str = None) -> list`

Filters extrinsics to match provided search criteria e.g. block range, pallet name, signed by accountID

Parameters
----------
kwargs

Returns
-------

Type annotations:
```python
block_start: <class 'int'>
block_end: <class 'int'>
ss58_address: <class 'str'>
pallet_name: <class 'str'>
call_name: <class 'str'>
return: <class 'list'>
```

##### `get_block_timestamp(self, block_number: int) -> int`

Return a UNIX timestamp for given `block_number`.

Parameters
----------
block_number: int The block_number to retrieve the timestamp for

Returns
-------
int

Type annotations:
```python
block_number: <class 'int'>
return: <class 'int'>
```

##### `init(self, substrate: 'SubstrateInterface')`

Initialization process of the extension. This function is being called by the ExtensionRegistry.

Parameters
----------
substrate: SubstrateInterface

Returns
-------

Type annotations:
```python
substrate: SubstrateInterface
```

##### `search_block_number(self, block_datetime: datetime.datetime, block_time: int = 6, **kwargs) -> int`

Search corresponding block number for provided `block_datetime`. the prediction tolerance is provided with
`block_time`

Parameters
----------
block_datetime: datetime
block_time: int
kwargs

Returns
-------
int

Type annotations:
```python
block_datetime: <class 'datetime.datetime'>
block_time: <class 'int'>
return: <class 'int'>
```

### SubstrateNodeSearchExtension

Implementation of `SearchExtension` using only Substrate RPC methods. Could be significant inefficient.

#### Methods

##### `close(self)`

Cleanup process of the extension. This function is being called by the ExtensionRegistry.

Returns
-------

##### `debug_message(self, message: str)`

Submits a debug message in the logger

Parameters
----------
message: str

Returns
-------

Type annotations:
```python
message: <class 'str'>
```

##### `filter_events(self, block_start: int = None, block_end: int = None, pallet_name: str = None, event_name: str = None, account_id: str = None) -> list`

Filters events to match provided search criteria e.g. block range, pallet name, accountID in attributes

Parameters
----------
kwargs

Returns
-------
list

Type annotations:
```python
block_start: <class 'int'>
block_end: <class 'int'>
pallet_name: <class 'str'>
event_name: <class 'str'>
account_id: <class 'str'>
return: <class 'list'>
```

##### `filter_extrinsics(self, block_start: int = None, block_end: int = None, ss58_address: str = None, pallet_name: str = None, call_name: str = None) -> list`

Filters extrinsics to match provided search criteria e.g. block range, pallet name, signed by accountID

Parameters
----------
kwargs

Returns
-------

Type annotations:
```python
block_start: <class 'int'>
block_end: <class 'int'>
ss58_address: <class 'str'>
pallet_name: <class 'str'>
call_name: <class 'str'>
return: <class 'list'>
```

##### `get_block_timestamp(self, block_number: int) -> int`

Return a UNIX timestamp for given `block_number`.

Parameters
----------
block_number: int The block_number to retrieve the timestamp for

Returns
-------
int

Type annotations:
```python
block_number: <class 'int'>
return: <class 'int'>
```

##### `init(self, substrate: 'SubstrateInterface')`

Initialization process of the extension. This function is being called by the ExtensionRegistry.

Parameters
----------
substrate: SubstrateInterface

Returns
-------

Type annotations:
```python
substrate: SubstrateInterface
```

##### `search_block_number(self, block_datetime: datetime.datetime, block_time: int = 6, **kwargs) -> int`

Search corresponding block number for provided `block_datetime`. the prediction tolerance is provided with
`block_time`

Parameters
----------
block_datetime: datetime
block_time: int
kwargs

Returns
-------
int

Type annotations:
```python
block_datetime: <class 'datetime.datetime'>
block_time: <class 'int'>
return: <class 'int'>
```

### datetime

datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

The year, month and day arguments are required. tzinfo may be None, or an
instance of a tzinfo subclass. The remaining arguments may be ints.

### timedelta

Difference between two datetime values.

timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

All arguments are optional and default to 0.
Arguments may be integers or floats, and may be positive or negative.

