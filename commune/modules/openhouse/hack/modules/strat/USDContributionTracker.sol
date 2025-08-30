// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title USDContributionTracker
 * @dev Contract to track USD value of contributions to a liquidity pool
 */
contract USDContributionTracker is Ownable {
    // Mapping from token address to price feed address
    mapping(address => address) public priceFeeds;
    
    // Mapping from user address to token address to contribution amount
    mapping(address => mapping(address => uint256)) public contributions;
    
    // Mapping from user address to total USD contribution (scaled by 1e18)
    mapping(address => uint256) public totalUsdContributions;
    
    // Total USD value contributed to the pool (scaled by 1e18)
    uint256 public totalPoolUsdValue;
    
    // Array of all tokens supported by the pool
    address[] public supportedTokens;
    
    // Events
    event ContributionAdded(address indexed user, address indexed token, uint256 amount, uint256 usdValue);
    event PriceFeedUpdated(address indexed token, address indexed priceFeed);
    event TokenAdded(address indexed token);
    
    /**
     * @dev Constructor
     */
    constructor() {}
    
    /**
     * @dev Add a new supported token with its price feed
     * @param token Address of the token
     * @param priceFeed Address of the Chainlink price feed for the token
     */
    function addSupportedToken(address token, address priceFeed) external onlyOwner {
        require(token != address(0), "Invalid token address");
        require(priceFeed != address(0), "Invalid price feed address");
        
        // Check if token is already supported
        bool isSupported = false;
        for (uint i = 0; i < supportedTokens.length; i++) {
            if (supportedTokens[i] == token) {
                isSupported = true;
                break;
            }
        }
        
        if (!isSupported) {
            supportedTokens.push(token);
            emit TokenAdded(token);
        }
        
        priceFeeds[token] = priceFeed;
        emit PriceFeedUpdated(token, priceFeed);
    }
    
    /**
     * @dev Update the price feed for a token
     * @param token Address of the token
     * @param priceFeed Address of the new price feed
     */
    function updatePriceFeed(address token, address priceFeed) external onlyOwner {
        require(token != address(0), "Invalid token address");
        require(priceFeed != address(0), "Invalid price feed address");
        
        // Ensure token is supported
        bool isSupported = false;
        for (uint i = 0; i < supportedTokens.length; i++) {
            if (supportedTokens[i] == token) {
                isSupported = true;
                break;
            }
        }
        require(isSupported, "Token not supported");
        
        priceFeeds[token] = priceFeed;
        emit PriceFeedUpdated(token, priceFeed);
    }
    
    /**
     * @dev Record a contribution to the pool
     * @param user Address of the user making the contribution
     * @param token Address of the token being contributed
     * @param amount Amount of tokens contributed
     */
    function recordContribution(address user, address token, uint256 amount) external onlyOwner {
        require(user != address(0), "Invalid user address");
        require(token != address(0), "Invalid token address");
        require(amount > 0, "Amount must be greater than 0");
        require(priceFeeds[token] != address(0), "Token not supported");
        
        // Update contribution record
        contributions[user][token] += amount;
        
        // Calculate USD value
        uint256 usdValue = getUsdValue(token, amount);
        
        // Update total USD contributions
        totalUsdContributions[user] += usdValue;
        totalPoolUsdValue += usdValue;
        
        emit ContributionAdded(user, token, amount, usdValue);
    }
    
    /**
     * @dev Get the latest price of a token in USD (scaled by 1e8)
     * @param token Address of the token
     * @return Latest price of the token in USD
     */
    function getLatestPrice(address token) public view returns (int256) {
        address priceFeedAddress = priceFeeds[token];
        require(priceFeedAddress != address(0), "Price feed not found");
        
        AggregatorV3Interface priceFeed = AggregatorV3Interface(priceFeedAddress);
        (, int256 price, , , ) = priceFeed.latestRoundData();
        return price;
    }
    
    /**
     * @dev Calculate the USD value of a token amount
     * @param token Address of the token
     * @param amount Amount of tokens
     * @return USD value of the tokens (scaled by 1e18)
     */
    function getUsdValue(address token, uint256 amount) public view returns (uint256) {
        int256 price = getLatestPrice(token);
        require(price > 0, "Invalid price");
        
        // Get token decimals
        uint8 decimals = IERC20(token).decimals();
        
        // Convert to USD value with 18 decimals precision
        // Price from Chainlink has 8 decimals
        uint256 usdValue = (amount * uint256(price) * 10**10) / (10**decimals);
        return usdValue;
    }
    
    /**
     * @dev Get the total USD value contributed by a user
     * @param user Address of the user
     * @return Total USD value contributed by the user
     */
    function getUserUsdContribution(address user) external view returns (uint256) {
        return totalUsdContributions[user];
    }
    
    /**
     * @dev Get the total USD value of the pool
     * @return Total USD value of the pool
     */
    function getPoolUsdValue() external view returns (uint256) {
        return totalPoolUsdValue;
    }
    
    /**
     * @dev Get the contribution of a user for a specific token
     * @param user Address of the user
     * @param token Address of the token
     * @return Amount of tokens contributed by the user
     */
    function getUserTokenContribution(address user, address token) external view returns (uint256) {
        return contributions[user][token];
    }
    
    /**
     * @dev Get the list of all supported tokens
     * @return Array of token addresses
     */
    function getSupportedTokens() external view returns (address[] memory) {
        return supportedTokens;
    }
    
    /**
     * @dev Get the contribution percentage of a user relative to the total pool
     * @param user Address of the user
     * @return Percentage of the user's contribution (scaled by 1e18, so 1e16 = 1%)
     */
    function getUserContributionPercentage(address user) external view returns (uint256) {
        if (totalPoolUsdValue == 0) return 0;
        return (totalUsdContributions[user] * 1e18) / totalPoolUsdValue;
    }
}
