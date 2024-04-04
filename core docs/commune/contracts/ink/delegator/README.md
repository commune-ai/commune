# Delegator Smart Contract Collection

The Delegator smart contract series is designed to demonstrate how to execute other smart contracts on-chain. This package contains four contracts:

- Delegator (root): This contract delegates calls to either the Adder or Subber contracts.
- Adder: Increases a value in the Accumulator contract.
- Subber: Decreases a value in the Accumulator contract.
- Accumulator: Contains a simple `i32` value that can be incremented or decremented.

To test this suite of contracts:

1. Compile all contracts using the `./build-all.sh` script to create the respective `.contract` packages in the `target/ink/` directory:
   - `target/ink/delegator.contract`
   - `target/ink/adder/adder.contract`
   - `target/ink/subber/subber.contract`
   - `target/ink/accumulator/accumulator.contract`
   
2. Upload the `.contract` package of Accumulator, Adder, and Subber to your chosen blockchain.  

3. Note the individual hashes of the uploaded contracts, found on the uploaded contracts page of the [Contracts UI](https://contracts-ui.substrate.io/).

4. Using those hashes and a starting value of your choosing, instantiate the Delegator smart contract. Make sure the endowment is ample - 1,000,000 if you're using `substrate-contracts-node` - so that the Delegator contract can instantiate the other three contracts on your behalf.

5. Once instantiated, use the Delegator contract's operations to `delegate` calls to either the Adder or Subber contract, thereby increasing or decreasing the value stored in the Accumulator contract. You can also use `switch` to change the contract to which the Delegator contract is currently delegating. By default, the Delegator contract will delegate to the Adder contract.
