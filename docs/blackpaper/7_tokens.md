
Tokens:

We want anyone to make a token where the community can provide liquidity to support new ideas that benefit open source projects. These tokens can be minted if they are connected the a subnet or a module. The native token has a null address of 

```python
c.add_token(key='5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w', #the key of the token
            supply=1000000, # initial supplys
            native_tokens=1000 # the initial native tokens 
            mint_ratio=1, # the number of tokens that are minted when emission enters from a module or a subnet
            k=1 # the curve ratio on the uniswap curve
            )

Token Params
```python
{
    'key': '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w',
    'token_pool': 51000,
    'native_pool': 1000,
    'mint_ratio': 10
    'k': 1,
} 
```

The token is now registered inside the chain's Token Storage Mapping the token's address to the balances holders.


Token State
```python

{
    'params': {
        'key': '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w', # the key of the token
        'token_pool': 51000, # the balance of the token pool
        'max_price_ratio': 1.1  # the maximum price ratio before the pool is adjusted
        'min_price_ratio': 0.9 # the minimum price ratio before the pool is adjustedssss
        'k': 1, # the curve ratio on the uniswap curve
    } 
    'balances': {
        '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w': 51000
    },
    'native_pool_balance': 1000 # the balance of the pool
}



```

Token Parameters

supply : The total supply of tokens. 
token_pool: The ratio of emissions of native to staked tokens minted
min_price_ratio: The minimum price ratio before the pool is adjusted
max_price_ratio: The maximum price ratio before the pool is adjusted
k: The curve ratio on the uniswap curve

Token Balances

The balances of the token holders are stored in the balances dictionary. The key is the address of the token holder and the value is the balance of the token holder.

Native Pool Balance

The native pool balance is the balance of the native tokens in the pool. This is the balance of the native tokens in the pool.

Minumum Price Ratio

The minimum price ratio is the minimum price ratio before the pool is adjusted. This is the minimum price ratio before the pool is adjusted. If the price hits the minimum price ratio, the token pool is adjusted such that the price is brought back to the minimum price ratio.

Maximum Price Ratio

The maximum price ratio is the maximum price ratio before the pool is adjusted. This is the maximum price ratio before the pool is adjusted. If the price hits the maximum price ratio, the token pool is minted to  adjusted such that the price is brought back to the minimum price ratio.


Token Minting from Emissions

If a module or subnet points towards a token address. The mint ratio will direct the tokens towards the emission provider. The stakers will also get the tokens minted in addition to the tokens from the pool. 


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
