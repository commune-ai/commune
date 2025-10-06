// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface IAaveLendingPool {
    function deposit(address asset, uint256 amount, address onBehalfOf, uint16 referralCode) external;
    function withdraw(address asset, uint256 amount, address to) external returns (uint256);
}

interface IAToken is IERC20 {
    function balanceOf(address user) external view returns (uint256);
}

contract AaveStrategy is Ownable {
    IERC20 public stablecoin;
    IAaveLendingPool public aaveLendingPool;
    IAToken public aToken;
    address public aggregator;
    
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator");
        _;
    }
    
    constructor(address _stablecoin, address _aaveLendingPool, address _aToken, address _aggregator) {
        stablecoin = IERC20(_stablecoin);
        aaveLendingPool = IAaveLendingPool(_aaveLendingPool);
        aToken = IAToken(_aToken);
        aggregator = _aggregator;
    }
    
    function deposit(uint256 amount) external onlyAggregator returns (uint256) {
        stablecoin.transferFrom(msg.sender, address(this), amount);
        stablecoin.approve(address(aaveLendingPool), amount);
        aaveLendingPool.deposit(address(stablecoin), amount, address(this), 0);
        return amount;
    }
    
    function withdraw(uint256 amount) external onlyAggregator returns (uint256) {
        uint256 withdrawn = aaveLendingPool.withdraw(address(stablecoin), amount, aggregator);
        return withdrawn;
    }
    
    function getBalance() external view returns (uint256) {
        return aToken.balanceOf(address(this));
    }
    
    function harvest() external onlyAggregator returns (uint256) {
        uint256 currentBalance = aToken.balanceOf(address(this));
        uint256 yield = currentBalance > 0 ? currentBalance - stablecoin.balanceOf(address(this)) : 0;
        if (yield > 0) {
            aaveLendingPool.withdraw(address(stablecoin), yield, aggregator);
        }
        return yield;
    }
}