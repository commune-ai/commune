# Governor Smart Contract

The Governor contract is a core part of the governance system which can be extended through various modules. It's designed to facilitate a decentralized voting system on the Ethereum blockchain.

This contract requires several functions to be implemented in various modules:

- A counting module must implement {quorum}, {_quorumReached}, {_voteSucceeded} and {_countVote}
- A voting module must implement {_getVotes}
- Additionally, the {votingPeriod} must also be implemented

## Features

- **Propose a change:** A user with a voting weight of at least the proposal threshold can propose a change to the system. The proposal includes a description of the changes, targets, values and calldata.
- **Voting:** Voting on proposals. Users can vote in favor or against a proposal. The total weight of votes in favor must be greater than the threshold for a proposal to be accepted.
- **Execution:** If a proposal is accepted, it can be executed. Execution can only take place after the voting period has ended. The execute function performs the actions specified in the proposal.
- **Cancellation:** Proposals can be cancelled. Cancelled proposals cannot be executed.
- **Relay:** This function allows the governor to interact with other contracts. This is useful when the governance executor is a contract other than the governor itself (such as a timelock contract).

## Interfaces and Inheritance

The contract extends from `Context`, `ERC165`, `EIP712`, `IERC721Receiver`, `IERC1155Receiver` and implements `IGovernor`.

- **EIP165:** Standard for the detection of the interfaces that a smart contract does support.
- **EIP712:** Standard for typed data signing.
- **ERC721Receiver and ERC1155Receiver:** Standards for non-fungible token and multiple token management respectively.
- **IGovernor:** Interface that defines the required functions for a governor contract.
