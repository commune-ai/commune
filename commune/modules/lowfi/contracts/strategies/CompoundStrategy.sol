// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface IComet {
    function supply(address asset, uint amount) external;
    function withdraw(address asset, uint amount) external;
    function balanceOf(address account) external view returns (uint256);
}

contract CompoundStrategy is Ownable {
    IERC20 public stablecoin;
    IComet public comet;
    address public aggregator;
    uint256 public depositedAmount;
    
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator");
        _;
    }
    
    constructor(address _stablecoin, address _comet, address _aggregator) {
        stablecoin = IERC20(_stablecoin);
        comet = IComet(_comet);
        aggregator = _aggregator;
    }
    
    function deposit(uint256 amount) external onlyAggregator returns (uint256) {
        stablecoin.transferFrom(msg.sender, address(this), amount);
        stablecoin.approve(address(comet), amount);
        comet.supply(address(stablecoin), amount);
        depositedAmount += amount;
        return amount;
    }
    
    function withdraw(uint256 amount) external onlyAggregator returns (uint256) {
        comet.withdraw(address(stablecoin), amount);
        stablecoin.transfer(aggregator, amount);
        depositedAmount -= amount;
        return amount;
    }
    
    function getBalance() external view returns (uint256) {
        return comet.balanceOf(address(this));
    }
    
    function harvest() external onlyAggregator returns (uint256) {
        uint256 currentBalance = comet.balanceOf(address(this));
        uint256 yield = currentBalance > depositedAmount ? currentBalance - depositedAmount : 0;
        
        if (yield > 0) {
            comet.withdraw(address(stablecoin), yield);
            stablecoin.transfer(aggregator, yield);
        }
        
        return yield;
    }
    
    function updateAggregator(address _newAggregator) external onlyOwner {
        aggregator = _newAggregator;
    }
}