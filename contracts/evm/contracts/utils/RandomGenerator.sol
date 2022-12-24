
// Solidity program to
// demonstrate on how
// to generate a random number
pragma solidity >=0.6.6;

// Creating a contract
contract RandomGenerator
{

// Initializing the state variable
uint randNonce = 0;
uint _modulus = 100;

// Defining a function to generate
// a random number
function rand() public  returns(uint)
{
// increase nonce
randNonce++;
return uint(keccak256(abi.encodePacked(block.timestamp ,msg.sender,randNonce))) % _modulus;
}
}
