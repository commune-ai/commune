```markdown
# Key Management

In this tutorial, we'll explore the usage of the `commune` Python package for managing keys, balances, stakes, and key statistics.

## Listing Keys

To start, let's list all the available keys using the `keys()` function:


```bash
c keys
```
or
```python
c.keys()
```

```
[
   'model.openrouter::replica.1',
    'model.openrouter::replica.2',
    'model.openrouter::replica.3',
    'model.openrouter::replica.4',
    'model.openrouter::replica.5',
    'model.openrouter::replica.6',
    'model.openrouter::replica.7',
    'model.openrouter::replica.8',
    'model.openrouter::replica.9'
]
```

## Adding and Removing Keys

You can add and remove keys with the following steps:

### Adding a New Key

To add a new key, use the `add_key()` function:

```python
c.add_key('fam')
```
or 
    
```bash
c add_key fam
```

## Getting Key Info

You can also retrieve key info using the `key_info()` function:

```python
c.key_info('fam')  # Replace 'fam' with the key name

```
{
    'crypto_type': 1,
    'seed_hex': '6a363df4c2b7eaeb0b13efedbd37308d2bda3be8bc8aa758ecc00eb3089f7b97',
    'derive_path': None,
    'path': 'fam',
    'ss58_format': 42,
    'public_key': '38199493328ca2224364c77204ee61008a9cab5a8246906201357ef056b82142',
    'ss58_address': '5DLG8wM2beoHcveKEXxuh2NRgh55vRRx8b1PE4Ch3ZE8fndL',
    'private_key': 
'd8e1c3d46f813eafac0d44481737e87b06241ba9cb5d6f760f8d62df48be450d2a84dcdfe506f218bc6646fe8
9daa1c1d1fd7af3a64ea0f3e8a73cc766743aa1',
    'mnemonic': 'typical piece chair oven lift trap case current tomorrow wrap motor 
light'
}
```



### Removing a Key

To remove a key, use the `rm_key()` function:

```python
c.rm_key('demo')  # Replace 'demo' with the key you want to remove
```

## Saving and Loading Keys

You can save and load keys for future use:

### Saving Keys

To save the keys, use the `save_keys()` function:

```python
c.save_keys()
```

### Loading Keys

To load the saved keys, use the `load_keys()` function:

```python
c.load_keys()
```

## Retrieving Balances and Stakes

You can retrieve balance and stake information for a specific key:

### Balance

To get the balance for a key, use the `get_balance()` function:

```python
c.get_balance('fam')  # Replace 'fam' with the key name
```
or 
```bash
c get_balance fam
```

### Stake

To get the stake for a key, use the `get_stake()` function:

```bash
c get_stake fam # Replace 'fam' with the key name or the address
```

```python
c.get_stake('fam')  # Replace 'fam' with the key name
```

### Get Registered Keys



