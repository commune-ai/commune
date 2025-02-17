
# KEY

The key is a sr25519 key that is used to sign, encrypt, decrypt and verify any string or messege. 
We can also replicate the key to other chains through using the same seed to generate the other keys. This means you can have one key instead of multiple keys for each chain, which is more convient and secure.

c.add_key("alice")

or 

c add_key alice

key already exists at alice
{
    'crypto_type': 1,
    'seed_hex': '518fad1043efc934a759334215ef54d48e1f8836355ed864bbb797f90ecb32b7',
    'derive_path': None,
    'path': 'alice',
    'ss58_format': 42,
    'public_key': '7cd0e327f4f6649719158892dafe766a5efd0185cb5fe17548d294f00f12661b',
    'private_key': 
'943fb89150a67192919a43004f87685faba470e754fe4ff0af6a93e7fc54dc0a6cceb6fbc29d610d5486ba78969f609ea83753fb9e32d58df0c67f13
dfcbbd68',
    'mnemonic': 'quantum belt rival casual benefit obscure sight wool pupil jaguar guide mango',
    'key': '5EtMr6n6APFay8FFdhhP9sMPwvv1Nfcm5yxiRTxviHH4WVZg'
}
Now this generates a random key and if you want to save it to a file you can do so like this.

c.add_key("alice")

or 

c add_key alice


{
    'crypto_type': 1,
    'seed_hex': 
'518fad1043efc934a759334215ef54d48e1f8836355ed864bbb797f90ecb32b7',
    'derive_path': None,
    'path': 'alice',
    'ss58_format': 42,
    'public_key': 
'7cd0e327f4f6649719158892dafe766a5efd0185cb5fe17548d294f00f12661b',
    'private_key': 
'943fb89150a67192919a43004f87685faba470e754fe4ff0af6a93e7fc54dc0a6cceb6fb
c29d610d5486ba78969f609ea83753fb9e32d58df0c67f13dfcbbd68',
    'mnemonic': 'quantum belt rival casual benefit obscure sight wool 
pupil jaguar guide mango',
    'key': '5EtMr6n6APFay8FFdhhP9sMPwvv1Nfcm5yxiRTxviHH4WVZg'
}


# Refreshing existing key

c add_key alice refresh=True



To list all the keys you can do so like this.

c.keys("alice")

or

c keys alice

[
    'alice',
]

To search for your keys you can do so like this. The search term finds all of the keys that contain the search term.

c keys ali 
[
    'alice',
    'alice2',
    'alice3',
]


# Save Keys

To save the keys to a file you can do so like this.

c save_keys

This saves the keys to a specific path in the config file. You can also specify the path like this.

To sign a message you can do so like this.

key = c.get_key("alice")



Original (Substrate) signature output :

key.sign("hello")

hexadecimal (bytes):

b'\xd6RV\xf4)\x88\x9aC\x99$\xe5E\xa5N=\xcf\xf4\x7f\xc7\\\xfe\xa1V\xdd\xc0
\xfc\x1bz:\x17\xa1$[\x84Al\xb0\xee\x0b\xedg\xc2\xe7\x93\x00\xf1~}\xd2r;\x
f2\xb4.\x90\xf2k\xd1\x10\xd9\xd5\x8f\x9d\x85'

dictionary

{"data":"hello",
"signature":"0x7e7","public_key":"0x7cd0e327f4f6649719158892dafe766a5efd0185cb5fe17548d294f00f12661b"}


String Output 

This is a string that cotainers the data and signature. The seperator is used to mainly distinguish the data from the signature.

{DATA}{SEPERATOR}{SIGNATURE}



Signature Tickets for Temporary Tokens

In the ticket the timestamp is taken, and the seperator is "::ticket::".

such that the format is 
timestamp::ticket::signature

by calling 

c.ticket("alice")

the alice key signs the current timestamp and returns the ticket.

1713500654.659339::ticket::e0559b535129037a62947c65af35f17c50d29b4a5c31df86b069d8ada5bcbb230f4c1e996393e6721f78d88f9b512b
6493b5ca743d027091585366875c6bea8e

now to verify the ticket you can do so like this.

c.verify_ticket("1713500654.659339::ticket::e0559b535129037a62947c65af35f17c50d29b4a5c31df86b069d8ada5bcbb230f4c1e996393e6721f78d88f9b512b6493b5ca743d027091585366875c6bea8e")

to get the signer

c.ticket2signer("1713500654.659339::ticket::e0559b535129037a62947c65af35f17c50d29b4a5c31df86b069d8ada5bcbb230f4c1e996393e6721f78d88f9b512b6493b5ca743d027091585366875c6bea8e")

To create a temperary token you can do so like this.

Temporary Tokens using Time Stampted Signaturs: Verification Without Your Keys

This allows for anyone to sign a timestamp, and vendors can verify the signature. This does not require the seed to be exposed, and can be used to identify key likley to be the same person. The only issue is if the staleness of the timestamp is too old. This can be adjusted by the vendor.



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
    'key': '5DLG8wM2beoHcveKEXxuh2NRgh55vRRx8b1PE4Ch3ZE8fndL',
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
c.save_keys(path='./keys.json') # save the key mnemonics to this file
```

### Loading Keys

To load the saved keys, use the `load_keys()` function:

```python
c.load_keys('./keys.json')
```


# SUBSPACE #


## Retrieving Balances and Stakes

You can retrieve balance and stake information for a specific key:

### Balance

To get the balance for a key, use the `get_balance()` function:

```python
c.get_balance('fam')  # Replace 'fam' with the key name
```
or 
```bash
c balance fam
```

### Get stake of the Key

To get the stake for a key, use the `get_stake()` function:

```bash
c get_stake fam # Replace 'fam' with the key name or the address
```

```python
c.get_stake('fam', netuid='text')  # Replace 'fam' with the key name
```
