# raydium_py

Python library to trade on AMM v4 and CPMM Raydium pools. 

***NOTE: CLMM is not supported. ***

```
pip install solana==0.35.0 solders==0.21.0
```

Updated: 3/30/2025

# Instructions

Clone the repo, and add your Private Key (Base58 string) and RPC to the config.py.

**When swapping, you must use the pool id, also known as the pair address. Do not use the mint aka token address.** 

We cannot pass the mint directly because there can be several pools for a single mint.

It is up to the user to fetch the pool ids via the Raydium API or via RPC methods I've included. 


**If you can - please support my work and donate to: 3pPK76GL5ChVFBHND54UfBMtg36Bsh1mzbQPTbcK89PD**

If you use my code, please give me credit in your project! 


# Contact

Telegram: https://t.me/AL_THE_BOT_FATHER

Group Telegram: https://t.me/Bot_Mafia_Support


# FAQS

**What format should my private key be in?** 

The private key should be in the base58 string format, not bytes. 

**Why are my transactions being dropped?** 

You get what you pay for. Don't use the main-net RPC, just spend the money for Helius or Quick Node.

**How do I change the fee?** 

Modify the UNIT_BUDGET and UNIT_PRICE in the config.py. 

**Why is this failing for USDC pairs?** 

This code only works for SOL pairs. 

**Why are there "no pool keys found"?** 

Free tier RPCs do not permit GET_PROGRAM_ACCOUNTS()! You must use a paid RPC.

**Does this code work on devnet?**

No. 
