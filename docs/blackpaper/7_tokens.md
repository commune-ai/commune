
Tokens:


We want anyone to make a token where the community can provide liquidity to support new ideas that benefit open source projects. These tokens can be minted if they are connected the a subnet or a module. The native token has a null address of 


LastSeller Token Parameters

The tokens are issued one to one when they are provided with emissions. This attempts to peg the price of each token one to one with commune. If there is indeed price manipulation through whales, we have two mechanisms to prevent this. 

LastSeller Tax: The first is the last seller tax which is the maximum percentage of the wallet that made the final trade that caused the pool to be adjusted.

min_price: The minimum price ratio before the pool is liquidated at that price




```python
c.add_token(key='5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w', #the key of the token
            min_price=0.9, # the minimum price ratio before the pool is adjusted
            seed_amount=1000, # the initial amount of tokens
            last_seller_tax=0.5, # 50 percent of the wallet that made the final trade that caused the pool to
            last_seller_period=1000, # the period of time that the last_seller tax is taken
            )


```python
{
    'key': '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w', # the key of the token
    'last_seller_tax': 0.1, # 10 percent of the wallet that made the final trade that caused the pool to be 
    'last_seller_period': 1000, # the period of time that the last_seller tax is taken
    'min_price': 0.9, # the minimum price ratio before the pool is adjusted
    'max_price': 1.1, # the maximum price ratio before the pool is adjusted
    'seed_amount' : 1000, # the initial amount of native tokens
} 
```

The token is now registered inside the chain's Token Storage Mapping the token's address to the balances holders.


Token State
```python

{
    'params': {
        'key': '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w', # the key of the token
        'last_seller_tax': 0.1, # 10 percent of the wallet that made the final trade that caused the pool to be 
        'min_price': 0.9, # the minimum price ratio before the pool is adjusted
        'max_price': 1.1, # the maximum price ratio before the pool is adjusted
        'seed_amount' : 1000, # the initial amount of native tokens
    } 
    'balances': {
        '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w': 1000 # the balance of the token holder who deposited the seed amount
    },
    'total_balance': 1000, # the total balance of the token pool
    'native_balance': 1000 # the balance of the pool
}



```

Token Balances

The balances of the token holders are stored in the balances dictionary. The key is the address of the token holder and the value is the balance of the token holder.

Native Pool Balance

The native pool balance is the balance of the native tokens in the pool. This is the balance of the native tokens in the pool.

Minumum Price Ratio

The minimum price ratio is the minimum price ratio before the pool is adjusted. This is the minimum price ratio before the pool is adjusted. If the price hits the minimum price ratio, the token liquidates at that price, and the token is removed from storage.

LastSellerPenalty:

The last seller penalty is the maximum percentage of the stake is distributed to the other holders of the token. For example, if billy converts his token into the native token with a last seller penalty of 10, then 10 percent of the stake is distributed to the other holders of the token.


Maximum Price Ratio

The maximum price ratio is the maximum price ratio before the pool is adjusted. This is the maximum price ratio before the pool is adjusted. If the price hits the maximum price ratio, the token pool is adjusted such that the price is brought back to the maximum price ratio.



Native Pool Balance

The native pool balance is the balance of the native tokens in the pool. This is the balance of the native tokens in the pool. 

 
```python

The following 
c.swap_tokens(from_token='5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w', 
              to_token='5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w
              amount=1000)
{
    'from_token_amount': 50000,
    'to_token_amount': 1000,
}
```

Token Minting from Emissions

If a module or subnet points towards a token address. The mint ratio will direct the tokens towards the emission provider. A module or subnet can connect with one or many tokens which can combine as meta token pools, which are adjusted bundles of token pools. However this part of the project will focus on single tokens per module and subnet.

Connecting a token to a module or subnet

```python
c.add_module_token(module='module',  # the module address
                  token='5F4bEvY7UBoM47qqedn8tv55YFAuqFwq1AgNHN9MXV5Dpteg',  # the token address
                  emission_ratio=0.1 # the ratio of emissions that are minted to the token
                  )
```
This means 10 percent of the liquidity (0.1) is being directed into the pool while getting the tokens minted in addition to the tokens from the pool. This forces less volitility as it incentivizes volume.

Connecting all of the Assets in the World : A decentralized blackrock 

The following will connect all of the assets in the world to the chain. We do this by allowing for the option for minting tokens using multisignature wallets. These signers of the ultisignature can approve the minting and burning of tokens upon collateralization of the assets. This allows for any token to be minted and burned upon the collateralization of the assets. 
