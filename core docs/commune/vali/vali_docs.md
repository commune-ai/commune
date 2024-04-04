# README

## Validator Deployment Tutorial

Welcome to the Validator Deployment Tutorial! Here you will find a step-by-step guide on how to deploy a validator on a blockchain network using the Commune module in Python. This tutorial will cover validator registration, staking, and the method to acquire a validator's status.

## Table of contents:
- Registering a Validator
- Staking a Validator
  - Staking to Multiple Validators
  - Staking Spread
- Viewing Validator Statistics

### Registering a Validator

To kick-start your network validation journey, the first task is to register your validator. Use the following CLI command/python code:

CLI Command:
```bash
c vali register tag=whadup
```
Python:
 ```python 
c.module('vali').register(tag='whadup')
```
This command creates a key tagged "vali::whadup". Continue deploying it by initiating the service and registering it:
   
```bash
c serve vali::whadup
c register vali::whadup
```

### Staking Your Validator

After registration, the next crucial step is staking. This process allows your validator to validate transactions and blocks, participate in consensus protocols, and earn rewards. The tutorial covers two staking scenarios: 

- **Staking a Single Validator** :

   Stake your validator with another key using this CLI command:

   ```bash
   c stake vali::whadup {amount}
   ```
   The default staked amount is your entire balance. If you have an insufficient balance, proceed to unstake. Use the following command:

   ```bash
   c unstake {amount}
   ```

- **Staking to Multiple Validators**:
   
    Distribute your stakes among multiple validators with the following command. The example demonstrates how to spread 100 units between two validators, "vali::whadup" & "vali::whadup2":

   ```bash
   c stake_many amount=100 modules=["vali::whadup","vali::whadup2"]
   ```

- **Stake Spread**:

    The stake spread mechanism allows distributing stakes among defined validators evenly. See how to allocate 100 units to the validators under 'whadup' category:

   ```bash
   c stake_spread whadup amount=100
   ```   

### Viewing Your Validator's Status

To check the performance metrics and other pertinent details of your validator, run the below commands:

CLI Command:
```bash
c vstats
```
Python:
```python
c.vstats()
```
Following is an example of the output you should expect:

![Validator Statistics Output](image.png)

This README is geared up to provide a basic understanding of validator deployment using simple CLI commands & Python scripting. Now that you know how to deploy a validator and stake your assets, you can contribute validations to your blockchain network while securing incentive rewards!