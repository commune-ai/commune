
Tokens

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

{ '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w'
    {
        'params': {
            'key': '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w',
            'token_pool': 51000,
            'k': 1,
        } 
        'balances': {
            '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w': 51000
        }
    }
}

```


The token creator can update the token by adding more tokens to the pool through emissions. 

Token Parameters

supply : The total supply of tokens. 
mint_ratio: The ratio of emissions of native to staked tokens minted
min_price

 
```python

The following 
c.swap_tokens(from_token='5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w', 
              to_token='5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w
              amount=51000)
{
    'token_key': '5Fe8eMg6YGDhZUwnnmiarTyNNGACGHeoDTVXez94yGA9mz9w',
    'token_pool': 51000,
    'native_pool': 1000,
    'k': 1
}
```

Token Minting from Emissions

If a module or subnet points towards a token address. The mint ratio will direct the tokens towards the emission provider. A module or subnet can connect with one or many tokens which can combine as meta token pools, which are adjusted bundles of token pools. However this part of the project will focus on single tokens per module and subnet.

Connecting a token to 

This will register the token onto the chain


c.update_module('module', token_ratio=0.1)

This means 10 percent of the liquidity (0.1) is being directed into the pool while getting the tokens minted in addition to the tokens from the pool. This forces less volitility as it incentivizes volume.