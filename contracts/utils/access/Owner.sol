pragma solidity ^0.8.7;

contract Owner {

   address owner;

   constructor() public {
       owner = msg.sender;
   }

   modifier onlyOwner {
      require(msg.sender == owner);
      _;
   }
   modifier costs(uint price) {
      if (msg.value >= price) {
         _;
      }
   }
}