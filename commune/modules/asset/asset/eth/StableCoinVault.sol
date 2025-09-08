// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract StableCoinVault {
    mapping(address => mapping(address => uint256)) public userBalances;
    mapping(address => bool) public acceptedTokens;
    address[] public tokenList;
    address public owner;
    
    event Deposit(address indexed user, address indexed token, uint256 amount);
    event Withdrawal(address indexed user, address indexed token, uint256 amount);
    event TokenAdded(address indexed token);
    event TokenRemoved(address indexed token);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        // Add default stablecoins
        acceptedTokens[0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48] = true; // USDC
        acceptedTokens[0xdAC17F958D2ee523a2206206994597C13D831ec7] = true; // USDT
        acceptedTokens[0x6B175474E89094C44Da98b954EedeAC495271d0F] = true; // DAI
        tokenList.push(0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48);
        tokenList.push(0xdAC17F958D2ee523a2206206994597C13D831ec7);
        tokenList.push(0x6B175474E89094C44Da98b954EedeAC495271d0F);
    }
    
    function addToken(address token) external onlyOwner {
        require(!acceptedTokens[token], "Token already accepted");
        acceptedTokens[token] = true;
        tokenList.push(token);
        emit TokenAdded(token);
    }
    
    function removeToken(address token) external onlyOwner {
        require(acceptedTokens[token], "Token not accepted");
        acceptedTokens[token] = false;
        emit TokenRemoved(token);
    }
    
    function deposit(address token, uint256 amount) external {
        require(acceptedTokens[token], "Token not accepted");
        require(amount > 0, "Amount must be greater than 0");
        
        IERC20(token).transferFrom(msg.sender, address(this), amount);
        userBalances[msg.sender][token] += amount;
        
        emit Deposit(msg.sender, token, amount);
    }
    
    function withdraw(address token, uint256 amount) external {
        require(userBalances[msg.sender][token] >= amount, "Insufficient balance");
        
        userBalances[msg.sender][token] -= amount;
        IERC20(token).transfer(msg.sender, amount);
        
        emit Withdrawal(msg.sender, token, amount);
    }
    
    function getUserBalance(address user, address token) external view returns (uint256) {
        return userBalances[user][token];
    }
    
    function getAcceptedTokens() external view returns (address[] memory) {
        return tokenList;
    }
}