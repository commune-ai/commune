
Commune: Incentivizing Applications in a Multichain World

Introduction

Proof of Stake Systems have come a far way since the introduction of the first PoS system in 2012. The first PoS system was Peercoin, which was a hybrid PoW/PoS system. The first pure PoS system was NXT, which was released in 2013. Since then, PoS systems have become more complex and have introduced new features such as slashing, validator rotation, and validator bonding. However, PoS systems are still limited in their ability to incentivize applications to run on their network. This is because PoS systems are designed to incentivize validators to secure the network, not to incentivize applications to run on the network. This is a problem because applications are the lifeblood of a blockchain network. Without applications, there is no demand for the network's native token, which means there is no demand for the network's security. This is why we need a new consensus system that is designed to incentivize applications to run on the network. This is where Commune comes in. Commune is a modular consensus system that is designed to incentivize applications to run on the network. Commune achieves this by introducing a new token model that is designed to incentivize applications to run on the network. This token model is based on the concept of bonding curves, which are a mathematical formula that determines the price of an asset based on its supply. Commune uses bonding curves to incentivize applications to run on the network by rewarding them with tokens that are created by the bonding curve. This creates a virtuous cycle where applications are incentivized to run on the network, which creates demand for the network's native token, which increases the security of the network, which attracts more applications to run on the network. This is the power of Commune. Commune is a modular consensus system that is designed to incentivize applications to run on the network. Commune achieves this by introducing a new token model that is based on the concept of bonding curves, which are a mathematical formula that determines the price of an asset based on its supply. Commune uses bonding curves to incentivize applications to run on the network by rewarding them with tokens that are created by the bonding curve. This creates a virtuous cycle where applications are incentivized to run on the network, which creates demand for the network's native token, which increases the security of the network, which attracts more applications to run on the network. This is the power of Commune.


Basics


Module/Object:

A module is a set of functions, and you can think of it as a class object in python. The motivation is to have a smart contract language that mimics python but is meant for the offchain world.  This is defined as a class or a set of stateless functions. A module can either be a single file (class inside a file) or a folder
with multiple files (stateless functions). Modules are turing complete, assuming the underlying language is turing complete, this allows modules to become anything. 

Key: 

A key is a module that can sign, encrypt and decrypt messeges. It is derived from 
a private key and has a public key. The key_address is derived from the public key 
via a hash function. The key's chain compatibility depends on its key_type. For instance, the keytype for a polkadot key (SR25519) is different from an ethereum key (ECDSA). Our keys are multichain as they can be used to sign transactions on different networks. We use AES for encryption and decryption and derive the symmetric key by hashing the private key of the SR25519. We also make sure the same private key can be used to sign transactions on different networks. This requires breaking convention 
from the original substrate key implementation. 

Server

a server is the module exposing its functions as http/ws/{protocol} endpoints. The server is responsible for handling requests and responses to and from the module.

Network:

A network module is meant to track 

you have a S vector of size (n) where n is the number of modules. Each module has a state that is updated by the network


Each token is defined by the bridging network and the liquidity pool the network is connected to. The liquidity pool is used to swap tokens between different networks

Token Incentives

The bonding curve is a magical thing that allows early contributors to have higher rewards than late contributors. Like it or not, this is how every asset is priced, which is the amount of one asset over the amount of another asset. We will use the following bonding curve

```
x * y = k 
# where x is the amount of the first asset
# y is the amount of the second asset
# k is the constant
```

Tokenized Staking 


Tokenized staking involves rewarding early stakers by calculating their emissions share with the amount of tokens they recieve as a result of the bonding curve. 
Each user recieves validator tokens in exhange for their staked tokens. The emissions are added to the validator's pool on the native side and the validator pool emits 1 token for every emitted token. 


Tracking Stop Loss for Each Staker

Each staker enters the the bonding curve with a stop loss. The stop loss is the minimum price the staker is willing to sell their tokens for. If the price of the token falls below the token stop loss, the pool is deregistered and the stakers get the pool value in the native token. 

Price Max and Min

If a buyer buys a token at a price that is higher than the max price, stake tokens are minted and the staker recieves the native tokens at the max price. If the price of the token falls below the min price, the pool is deregistered and the stakers get the pool value in the native token. This allows for a stop loss and stop gain mechanism to reduce the volatility of the token.

What is a token without this stop loss mechanism? It is a token that can be pumped and dumped by anyone and can have a minimum of 0 value and a max of infinity. This is why we need a stop loss mechanism to prevent the token from being dumped to 0.


AntiRug Mechanism

The issue with using staking bonding curves is potential volitility.

A subnet is a set of modules in a network. A subnet is created by the founder of the subnet. The founder locks any amount of liquidity into the subnet. This liquidity cannot be removed and is the final liquidation when the subnet is deregistered. This prevents any pump and dump schemes by the founder. Subnets are prioritized based on the amount of liquidity locked in the subnet. The subnet with the most liquidity is deregistered last. This linear relationship adds elasticity to the pricing mechanism, and does not have artificial limits on the amount of liquidity that can be locked in the subnet.

Min And Max Pricing of the Token

The subnet founder can set the min and max price of the token. The minumum price acts as a stop loss where the subnet is deregistered if the price of the token falls below a certain ratio in the past epoch. In this scenario the liquidator that is responsible for the slippage of the price that crosses the mininum price recieves a penalty of X percent slashed from the their liquidation. 

Token Emission vs Native Emission

The emissions can print any ratio of tokens between the min and max ratio.  is set by the founder of the subnet native emission mints a 1 to 1 ratio with the token emission, this allows for the price to stabilize around the native token, where it would lower the price of the token if the token is > 1 and increase the price of the token if the token is < 1. 

Subnet Creation

Subnets are created by the founder of the subnet. The founder locks any amount of liquidity into the subnet. This liquidity cannot be removed and is the final liquidation when the subnet is deregistered. This prevents any pump and dump schemes by the founder.


The subnet founder can set the min and max price of the token. The minumum price acts as a stop loss where the subnet is deregistered if the price of the token falls below a certain ratio in the past epoch. In this scenario the liquidator that is responsible for the slippage of the price that crosses the mininum price recieves a penalty of X percent based on the setting final_liquidation penalty. 



Subnet Deregistration from 

When a host is deregistered, the subnets are prioritized based on the amount of liquidity locked in the subnet. The subnet with the most liquidity is deregistered last. This creates a linear relationship between the amount of liquidity locked in the subnet and the time it takes to deregister the subnet. 


Subnet Deregistration from Stop Loss 

The price of the token is determined by the bonding curve. The bonding curve is a mathematical formula that determines the price of an asset based on its supply. When the price of` the token falls below a certain threshold, the subnet is deregistered. This prevents any pump and dump schemes by the founder or other players. 

Stake Loaning 

User to User Token Lending:

Users can lend their tokens to other users. This allows users to increase their stake without having to buy more tokens. The user that is lending their tokens receives a percentage of the rewards from the user that is borrowing their tokens. This allows users to increase their stake without having to buy more tokens.

Validator to Validator Token Lending:

Validators can lend their tokens to other validators. This allows validators to increase their stake without having to buy more tokens. The validator that is lending their tokens receives a percentage of the rewards from the validator that is borrowing their tokens. This allows validators to increase their stake without having to buy more tokens.

Locking Other Tokens in the Subnet from Other Networks

In order to lock a token you need to register a token state channel bridge between that token. Using IBC, we can create derivatives of pools pegged with commune tokens using an oracle that shows the price of the token as well as the native token. The network can add limits to liquidity in the case of low liquidity tokens to prevent the token from being dumped.


Why A Module is Turing Complete and Can Be Anything

A module is a set of functions, and you can think of it as a class object in python. The motivation is to have a smart contract language that mimics object oriented languages that already exist, starting with python, but is meant for the offchain world.  This is defined as a class or a set of