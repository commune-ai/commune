Subnet Tutorial

In this tutorial, you will learn how to deploy a subnet on the network and perform various tasks related to staking, registration, and validation.

What is a subnet, you ask? A subnet is a group of validators that validate miners on the network. 

The subnet we are going to do is called "subnet", but you can name it whatever you want, by renaming the subnet/subnet.py file to {subnet_name}{subnet_name}.py.

Each validator has a score function

Your score function will be used to determine the weight of your vote in the network.
To define the score function, you need to define a function called "score" in your vali.py file.


## Important Validator Functions

score_module

The validator runs a thread executor over the namespace of the subnet, where miners are chosen to be validated. The validator will then run the score function on each miner in the namespace, and the miner with the highest score will be chosen to be validated.

filter_module

If you want to filter the namespace of the subnet, you can define a function called "filter_module" in your vali.py file. This filter function will be used to filter the namespace of the subnet, and only the miners that pass the filter will be validated.


Important Miner Functions

forward 

The forward function is used to forward the miner to the subnet. The miner will be sent to the subnet, and the subnet will validate the miner. This can be a different funciton if the validator calls a different function to validate the miner. It is important to understand the function and inputs/outputs of the function being called on the validator.


def filter_module(module_name:str):
    if module_name.startswith("miner"):
        return True
    return False

Start a Subnet

Serve the validator on the local network

c.serve("subnet.vali::test",  network='local')

Serve a the miners on the local network (default, so you don't need to specify the network)

c.serve("subnet.miner::test_1")
c.serve("subnet.miner::test_2")


Leaderboard

To check the leaderboard of the subnet, use the following command:

```python

c.call("subnet.vali/leaderboard")
```
                   name    w  staleness   latency
0  subnet.miner::test_0  1.0  10.957290  0.015491
1  subnet.miner::test_1  1.0  11.076705  0.026521
2  subnet.miner::test_2  1.0  11.035383  0.012495


