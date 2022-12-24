// // SPDX-License-Identifier: MIT

// pragma solidity >=0.8;

// import "./Contract.sol";
// import "interfaces/oracle/chainlink/AggregatorV3.sol";

// contract ContractDeployer {

//     address public owner;

//     constructor() {
//         owner = msg.sender;
//     }

//     mapping(address => address) public getContractAddress;
//     mapping(address => address) public getContractOwner;

//     Contract[] public contracts;

//     function deployContract() external payable {
//         uint256 minimumUsd = 50 * 10 ** 18;
//         require(getConversionRate(msg.value) >= minimumUsd, "You need to spend at least $50 to deploy your contract");
//         Contract _contract = new Contract();
//         contracts.push(_contract);            //vlt address(msg.sender) ????
//         contracts.push(msg.sender);
//         getContractAddress[msg.sender] = _contract.address;
//         getContractOwner[_contract.address] = msg.sender;
//     }

//     /**
//      * @dev Neccessary for deployContract()
//      */
//     function getEthPrice() public view returns (uint256) { 
//         AggregatorV3Interface priceFeed = AggregatorV3Interface(0x8A753747A1Fa494EC906cE90E9f37563A8AF630e);
//         (,int256 answer,,,) = priceFeed.latestRoundData();
//         return uint256(answer) * 10000000000;
//     }

//     /**
//      * @dev Neccessary for deployContract()
//      */
//     function getConversionRate(uint256 ethAmount) public view returns (uint256) {
//         uint256 ethPriceInWei = getEthPrice();
//         uint256 ethAmountInUsd = (ethPriceInWei * ethAmount) / 1000000000000000000;
//         return ethAmountInUsd;
//     }

//     function withdraw() external payable {
//         require(msg.sender == owner, "You are not the owner of this contract");
//         payable(owner).transfer(address(this).balance);
//     }
// }