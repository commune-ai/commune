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
    'ss58_address': '5EtMr6n6APFay8FFdhhP9sMPwvv1Nfcm5yxiRTxviHH4WVZg'
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
    'ss58_address': '5EtMr6n6APFay8FFdhhP9sMPwvv1Nfcm5yxiRTxviHH4WVZg'
}



To list all the keys you can do so like this.

c.keys("alice")

or

c keys alice

[
    'alice',
]


To sign a message you can do so like this.

key = c.get_key("alice")
key.sign("hello world")

b'\xd6RV\xf4)\x88\x9aC\x99$\xe5E\xa5N=\xcf\xf4\x7f\xc7\\\xfe\xa1V\xdd\xc0
\xfc\x1bz:\x17\xa1$[\x84Al\xb0\xee\x0b\xedg\xc2\xe7\x93\x00\xf1~}\xd2r;\x
f2\xb4.\x90\xf2k\xd1\x10\xd9\xd5\x8f\x9d\x85'


Other signature outputs:


dictionary

{"data":"hello world","signature":"0x7e7","public_key":"0x7cd0e327f4f6649719158892dafe766a5efd0185cb5fe17548d294f00f12661b"}


string 
This is a string that cotainers the data and signature. The seperator is used to mainly distinguish the data from the signature.

{DATA}{SEPERATOR}{SIGNATURE}

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


