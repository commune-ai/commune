# Checkpoints - Solidity Library

Checkpoints is a Solidity library used for maintaining checkpoints of certain data at different points in time. It is useful where past values need to be traced back and viewed via block numbers.

## Features

- The library defines a `History` struct that stores the `Checkpoint` data that include a block number and a value.

- By providing a block number, the library can retrieve its corresponding past value using the `getAtBlock` and `getAtProbablyRecentBlock` functions. If there is no available checkpoint at that block, it will return the nearest checkpoint before it.

- New data can be recorded into checkpoints via the `push` function. This updated data is then recorded as the checkpoint for the current block.

- The library also provides a `latest` function which returns the latest checkpoint's value, and a `latestCheckpoint` function which confirms if a checkpoint is present in the data structure and if so returns the values from the latest checkpoint.

- The number of existing checkpoints can be returned using the `length` function.

Please note that all function calls require blocks to be mined. This library gives a way to trace back and monitor any data at different points in time using Ethereum's block history.

## Prerequisites

To use this library, first import it into your Solidity file. Then, define a variable of type `Checkpoints.History` in your smart contract.

## Example

```javascript
pragma solidity ^0.8.0;
// After importing the Checkpoints library
Checkpoints.History private _checkpoints;

function addData(uint256 value) public {
    Checkpoints.push(_checkpoints, value);
}

function getOldData(uint256 blockNumber) public view returns (uint256) {
    return Checkpoints.getAtBlock(_checkpoints, blockNumber);
}
```

In the above example, when the `addData` function is called with some `value`, it adds the `value` as a checkpoint for the current transaction block. Then, by calling the `getOldData` function with a specific `blockNumber`, it returns the value that was added during the corresponding block.

The library was last updated in OpenZeppelin Contracts v4.5.0.