# README - Subspace Module

The Subspace module is a robust implementation designed to handle a diverse set of functionalities including network statistics tracking, module registration/updation, network syncing, and token management among others.

## Getting Started

### Set-Up and Dependencies

Before using Subspace, ensure you have Docker installed. Subspace runs on Docker and provides containerized execution of processes. 

## Key Terminologies

**Root Key:** The root key is your main key for managing commune modules.

```bash 
c root_key
```

## Features

### Register a Module

To register a module:

```bash
c model.openai register tag=sup api_key=sk-...
```
or
```bash
c register model.openai tag=sup api_key=sk-...
```
*Note:* Ensure you specify a unique tag as it will not work if the name has already been taken in the subnet.

### Updating a Module

You can easily update the module's name and address:

```bash
c update_module model.openai name=model.openai::fam1 address=124.545.545:8080
```
*Note:* Renaming the module requires a server restart. 

### Syncing with the Network

Run the Subspace sync loop, updating every 60 seconds:

```bash
c s loop interval=60
```

### Check Statistics

Check the stats of a module. To sync these stats with the network, you cab use the same command:

```bash
c stats
```

### Token Management

Transferring tokens to a module:

```bash
c transfer 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S 100 
```
Staking/Unstaking tokens on/from a module:

```bash
c stake module=5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S amount=100
c unstake  module=5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S amount=10
```

### Starting a Local Node/Chain

To start a local node:

```bash
c start_local_node
```
To start a chain:

```bash
c start_chain dev valis=4 nonvalis=4
```

*Note:* You can specify the chain to be 'main' or 'dev'.

## Contact Support

If you encounter any issues or have suggestions to improve the module, leave your comments on the GitHub repository @[Github Repo Link].

## Conclusion

Subspace module is a one-stop solution for various activities such as staking/unstaking tokens, registering and updating modules, and network synchronization. Visit [Subspace Documentation] (Url) for more information about the various functionalities provided.