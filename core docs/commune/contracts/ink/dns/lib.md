# Domain Name Service

This is a domain name service contract written in ink! It provides an implementation of a secure and decentralized domain name resolution service on Polkadot. The main function of this contract is domain name resolution, which refers to the retrieval of numeric values corresponding to readable and easily memorable names such as "polka.dot". These names can be used to facilitate transfers, voting, and DApp-related operations instead of resorting to long IP addresses that are hard to remember.

This ink! contract serves as a mapping between user-friendly names to blockchain addresses (AccountId). It includes event logging support for registration, address changes, and ownership transfers.

## Usage

The contract includes the following methods:

### Register

This function allows a user to register a new name with the caller as the owner. The function will fail with an Error::NameAlreadyExists if a name is already registered within the system.

### SetAddress

The setAddress function provides a way for a user to set an address for a specific name as long as the calling user is the owner of the name. If the name already has a registered owner and the caller is not that owner, the function will fail with Error::CallerIsNotOwner.

### Transfer

The Transfer function allows for the ownership of a domain name to be transferred from the current owner to a new owner. Similar to the setAddress function, the function will fail if the caller is not the current owner.

### GetAddress and GetOwner

These functions are used to retrieve the address or owner associated with a specific name.

## Testing

The code also includes a suite of tests for checking the functionality of register, setAddress and transfer functions. The tests check expected functionality, as well as expected failures.

## Errors

This contract includes a custom Error enum to communicate error states. Errors include:
- NameAlreadyExists: returned if the name already exists upon registration
- CallerIsNotOwner: returned if the caller is not the owner when an operation requiring ownership is attempted.