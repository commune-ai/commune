// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

interface IYieldStrategy {
    function deposit(uint256 amount) external returns (uint256);
    function withdraw(uint256 amount) external returns (uint256);
    function getBalance() external view returns (uint256);
    function harvest() external returns (uint256);
}

contract StablecoinYieldAggregator is ERC20, ReentrancyGuard, Ownable {
    IERC20 public stablecoin;
    uint256 public constant TOTAL_SHARES = 21_000_000 * 1e18;
    uint256 public constant POOL_ALLOCATION_BPS = 1000; // 10% to pool
    uint256 public totalPoolShares;
    
    mapping(address => IYieldStrategy) public strategies;
    address[] public strategyList;
    mapping(address => uint256) public strategyAllocations;
    mapping(address => uint256) public userPoolShares;
    
    uint256 public totalDeposited;
    uint256 public totalYieldGenerated;
    
    event Deposited(address indexed user, uint256 amount, uint256 shares);
    event Withdrawn(address indexed user, uint256 amount, uint256 shares);
    event YieldHarvested(uint256 amount, uint256 poolAmount);
    event StrategyAdded(address indexed strategy, uint256 allocation);
    event PoolSharesAllocated(address indexed user, uint256 shares);
    
    constructor(address _stablecoin) ERC20("Yield Aggregator Token", "YAT") {
        stablecoin = IERC20(_stablecoin);
    }
    
    function addStrategy(address _strategy, uint256 _allocationBps) external onlyOwner {
        require(_strategy != address(0), "Invalid strategy");
        require(_allocationBps <= 10000, "Invalid allocation");
        
        strategies[_strategy] = IYieldStrategy(_strategy);
        strategyList.push(_strategy);
        strategyAllocations[_strategy] = _allocationBps;
        
        emit StrategyAdded(_strategy, _allocationBps);
    }
    
    function deposit(uint256 _amount) external nonReentrant {
        require(_amount > 0, "Amount must be > 0");
        require(stablecoin.transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        
        uint256 shares = totalSupply() == 0 ? _amount : (_amount * totalSupply()) / totalDeposited;
        _mint(msg.sender, shares);
        
        totalDeposited += _amount;
        _distributeToStrategies(_amount);
        
        emit Deposited(msg.sender, _amount, shares);
    }
    
    function withdraw(uint256 _shares) external nonReentrant {
        require(_shares > 0 && _shares <= balanceOf(msg.sender), "Invalid shares");
        
        uint256 amount = (_shares * totalDeposited) / totalSupply();
        _burn(msg.sender, _shares);
        
        totalDeposited -= amount;
        _withdrawFromStrategies(amount);
        
        require(stablecoin.transfer(msg.sender, amount), "Transfer failed");
        emit Withdrawn(msg.sender, amount, _shares);
    }
    
    function harvestYield() external nonReentrant {
        uint256 totalYield = 0;
        
        for (uint256 i = 0; i < strategyList.length; i++) {
            address strategyAddr = strategyList[i];
            IYieldStrategy strategy = strategies[strategyAddr];
            uint256 yield = strategy.harvest();
            totalYield += yield;
        }
        
        uint256 poolAmount = (totalYield * POOL_ALLOCATION_BPS) / 10000;
        uint256 userAmount = totalYield - poolAmount;
        
        totalDeposited += userAmount;
        totalYieldGenerated += totalYield;
        
        _allocatePoolShares(poolAmount);
        
        emit YieldHarvested(totalYield, poolAmount);
    }
    
    function _distributeToStrategies(uint256 _amount) internal {
        for (uint256 i = 0; i < strategyList.length; i++) {
            address strategyAddr = strategyList[i];
            uint256 allocation = strategyAllocations[strategyAddr];
            uint256 strategyAmount = (_amount * allocation) / 10000;
            
            if (strategyAmount > 0) {
                stablecoin.approve(strategyAddr, strategyAmount);
                strategies[strategyAddr].deposit(strategyAmount);
            }
        }
    }
    
    function _withdrawFromStrategies(uint256 _amount) internal {
        uint256 remaining = _amount;
        
        for (uint256 i = 0; i < strategyList.length && remaining > 0; i++) {
            address strategyAddr = strategyList[i];
            uint256 strategyBalance = strategies[strategyAddr].getBalance();
            uint256 toWithdraw = remaining > strategyBalance ? strategyBalance : remaining;
            
            if (toWithdraw > 0) {
                strategies[strategyAddr].withdraw(toWithdraw);
                remaining -= toWithdraw;
            }
        }
    }
    
    function _allocatePoolShares(uint256 _amount) internal {
        if (totalPoolShares >= TOTAL_SHARES) return;
        
        uint256 sharesToMint = (_amount * 1e18) / 1e6; // Simplified conversion
        if (totalPoolShares + sharesToMint > TOTAL_SHARES) {
            sharesToMint = TOTAL_SHARES - totalPoolShares;
        }
        
        totalPoolShares += sharesToMint;
    }
    
    function claimPoolShares() external nonReentrant {
        uint256 userShares = (userPoolShares[msg.sender] * balanceOf(msg.sender)) / totalSupply();
        require(userShares > 0, "No shares to claim");
        
        userPoolShares[msg.sender] = 0;
        emit PoolSharesAllocated(msg.sender, userShares);
    }
    
    function getPoolSharesAvailable() external view returns (uint256) {
        return TOTAL_SHARES - totalPoolShares;
    }
    
    function getTotalValue() external view returns (uint256) {
        uint256 total = stablecoin.balanceOf(address(this));
        for (uint256 i = 0; i < strategyList.length; i++) {
            total += strategies[strategyList[i]].getBalance();
        }
        return total;
    }
}