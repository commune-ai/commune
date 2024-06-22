The key is a sr25519 key that is used to sign, encrypt, decrypt and verify any string or messege. 
We can also replicate the key to other chains through using the same seed to generate the other keys. This means you can have one key instead of multiple keys for each chain, which is more convient and secure.

![Alt text](image_key.png)

c.add_key("alice")

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


# Save Keys

To save the keys to a file you can do so like this.

c save_keys



To sign a message you can do so like this.

key = c.get_key("alice")
key.sign("hello world")


Bytes Output:

b'\xd6RV\xf4)\x88\x9aC\x99$\xe5E\xa5N=\xcf\xf4\x7f\xc7\\\xfe\xa1V\xdd\xc0
\xfc\x1bz:\x17\xa1$[\x84Al\xb0\xee\x0b\xedg\xc2\xe7\x93\x00\xf1~}\xd2r;\x
f2\xb4.\x90\xf2k\xd1\x10\xd9\xd5\x8f\x9d\x85'

Dictionary Output:

{"data":"hello world","signature":"0x7e7","public_key":"0x7cd0e327f4f6649719158892dafe766a5efd0185cb5fe17548d294f00f12661b"}

String Output:

This cotains the data and signature joined by a seperator.

{DATA}{SEPERATOR}{SIGNATURE}


key2balance map


c key2balance


key2stake map

c key2stake


