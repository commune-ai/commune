// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface ICurvePool {
    function add_liquidity(uint256[2] calldata amounts, uint256 min_mint_amount) external returns (uint256);
    function remove_liquidity_one_coin(uint256 token_amount, int128 i, uint256 min_amount) external returns (uint256);
    function balanceOf(address account) external view returns (uint256);
}

interface ICurveGauge {
    function deposit(uint256 value) external;
    function withdraw(uint256 value) external;
    function balanceOf(address account) external view returns (uint256);
    function claim_rewards() external;
}

contract CurveStrategy is Ownable {
    IERC20 public stablecoin;
    ICurvePool public curvePool;
    ICurveGauge public gauge;
    address public aggregator;
    uint256 public depositedAmount;
    
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator");
        _;
    }
    
    constructor(address _stablecoin, address _curvePool, address _gauge, address _aggregator) {
        stablecoin = IERC20(_stablecoin);
        curvePool = ICurvePool(_curvePool);
        gauge = ICurveGauge(_gauge);
        aggregator = _aggregator;
    }
    
    function deposit(uint256 amount) external onlyAggregator returns (uint256) {
        stablecoin.transferFrom(msg.sender, address(this), amount);
        stablecoin.approve(address(curvePool), amount);
        
        uint256[2] memory amounts = [amount, 0];
        uint256 lpTokens = curvePool.add_liquidity(amounts, 0);
        
        IERC20(address(curvePool)).approve(address(gauge), lpTokens);
        gauge.deposit(lpTokens);
        
        depositedAmount += amount;
        return amount;
    }
    
    function withdraw(uint256 amount) external onlyAggregator returns (uint256) {
        uint256 lpBalance = gauge.balanceOf(address(this));
        uint256 lpToWithdraw = (amount * lpBalance) / depositedAmount;
        
        gauge.withdraw(lpToWithdraw);
        uint256 withdrawn = curvePool.remove_liquidity_one_coin(lpToWithdraw, 0, 0);
        
        stablecoin.transfer(aggregator, withdrawn);
        depositedAmount -= amount;
        return withdrawn;
    }
    
    function getBalance() external view returns (uint256) {
        return gauge.balanceOf(address(this));
    }
    
    function harvest() external onlyAggregator returns (uint256) {
        gauge.claim_rewards();
        
        uint256 currentBalance = gauge.balanceOf(address(this));
        uint256 estimatedValue = depositedAmount;
        uint256 yield = currentBalance > estimatedValue ? (currentBalance - estimatedValue) / 10 : 0;
        
        if (yield > 0) {
            gauge.withdraw(yield);
            uint256 withdrawn = curvePool.remove_liquidity_one_coin(yield, 0, 0);
            stablecoin.transfer(aggregator, withdrawn);
            return withdrawn;
        }
        
        return 0;
    }
    
    function updateAggregator(address _newAggregator) external onlyOwner {
        aggregator = _newAggregator;
    }
}