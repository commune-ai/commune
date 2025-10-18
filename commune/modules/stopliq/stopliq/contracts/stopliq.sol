// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.19;

import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
// Also import Uniswap's TickMath, LiquidityAmounts, FullMath libs

contract StopLossVault {
    INonfungiblePositionManager public immutable positionManager;
    address public governance;
    uint256 public positionCounter;

    struct PositionInfo {
        address owner;
        uint256 tokenId;
        address token0;
        address token1;
        uint24 poolFee;
        uint128 liquidity;
        uint256 initialValueUSD; // scaled to 1e18
        uint16 stopLossBps; // e.g., 2000 = 20.00%
        uint256 bounty; // bounty in native token or USDC - protocol policy
        bool active;
    }

    mapping(uint256 => PositionInfo) public positions;

    // Chainlink feeds mapping token => aggregator
    mapping(address => AggregatorV3Interface) public priceFeeds;

    event PositionCreated(uint256 indexed posId, address indexed owner, uint256 tokenId, uint256 initialValueUSD);
    event StopLossTriggered(uint256 indexed posId, uint256 valueUSD, address by, uint256 bountyPaid);

    modifier onlyOwner(uint256 posId) {
        require(msg.sender == positions[posId].owner, "not owner");
        _;
    }

    constructor(address _positionManager) {
        positionManager = INonfungiblePositionManager(_positionManager);
        governance = msg.sender;
    }

    // Admin: register chainlink feed for token
    function setPriceFeed(address token, address feed) external {
        require(msg.sender == governance, "not gov");
        priceFeeds[token] = AggregatorV3Interface(feed);
    }

    // Create position: user must approve tokens to this contract beforehand
    function createPosition(
        address token0,
        address token1,
        uint24 fee,
        int24 tickLower,
        int24 tickUpper,
        uint256 amount0Desired,
        uint256 amount1Desired,
        uint16 stopLossBps,
        uint256 bounty
    ) external payable returns (uint256 posId) {
        // Transfer tokens in, mint Uniswap position to this contract
        // Omitted: safeTransferFrom token0/token1

        INonfungiblePositionManager.MintParams memory params = INonfungiblePositionManager.MintParams({
            token0: token0,
            token1: token1,
            fee: fee,
            tickLower: tickLower,
            tickUpper: tickUpper,
            amount0Desired: amount0Desired,
            amount1Desired: amount1Desired,
            amount0Min: 0,
            amount1Min: 0,
            recipient: address(this),
            deadline: block.timestamp
        });

        (uint256 tokenId, uint128 liquidity, uint256 amount0, uint256 amount1) = positionManager.mint(params);

        // compute initialValueUSD via feeds (call helper)
        uint256 initialValueUSD = _computeValueUSD(token0, amount0) + _computeValueUSD(token1, amount1);

        posId = ++positionCounter;
        positions[posId] = PositionInfo({
            owner: msg.sender,
            tokenId: tokenId,
            token0: token0,
            token1: token1,
            poolFee: fee,
            liquidity: liquidity,
            initialValueUSD: initialValueUSD,
            stopLossBps: stopLossBps,
            bounty: bounty,
            active: true
        });

        emit PositionCreated(posId, msg.sender, tokenId, initialValueUSD);
    }

    // Anyone can call execute when condition met
    function executeStopLoss(uint256 posId) external {
        PositionInfo storage p = positions[posId];
        require(p.active, "inactive");

        uint256 valueUSD = positionValueUSD(posId);
        uint256 threshold = (p.initialValueUSD * (10000 - p.stopLossBps)) / 10000;
        require(valueUSD <= threshold, "not triggered");

        // call Uniswap to decrease full liquidity
        ( , , , , , , , uint128 liquidity, , , , ) = positionManager.positions(p.tokenId);

        if (liquidity > 0) {
            // decreaseLiquidity params - remove 100% (liquidity)
            INonfungiblePositionManager.DecreaseLiquidityParams memory dec = INonfungiblePositionManager.DecreaseLiquidityParams({
                tokenId: p.tokenId,
                liquidity: liquidity,
                amount0Min: 0,
                amount1Min: 0,
                deadline: block.timestamp
            });
            positionManager.decreaseLiquidity(dec);
        }

        // collect all
        INonfungiblePositionManager.CollectParams memory collectParams = INonfungiblePositionManager.CollectParams({
            tokenId: p.tokenId,
            recipient: address(this),
            amount0Max: type(uint128).max,
            amount1Max: type(uint128).max
        });
        positionManager.collect(collectParams);

        // After collecting, transfer assets back to p.owner.
        // Omitted: compute amounts of token0/token1 and transfer, handle bounty payout (in native token or chosen asset)

        p.active = false;
        emit StopLossTriggered(posId, valueUSD, msg.sender, p.bounty);
    }

    function positionValueUSD(uint256 posId) public view returns (uint256) {
        PositionInfo storage p = positions[posId];
        // Query positionManager.positions to get liquidity and ticks
        ( , , , , int24 tickLower, int24 tickUpper, , uint128 liquidity, , , , ) = positionManager.positions(p.tokenId);

        // Query pool sqrtPrice and compute amounts for liquidity using TickMath & LiquidityAmounts libraries
        // get sqrtPriceX96 from pool's slot0
        // convert liquidity to amount0, amount1
        // query price feeds and compute USD value
        // For brevity in this skeleton, we return initialValueUSD (replace with real impl)
        return p.initialValueUSD; // PLACEHOLDER - real impl required
    }

    // helper that reads chainlink feed and multiplies
    function _computeValueUSD(address token, uint256 amount) internal view returns (uint256) {
        AggregatorV3Interface feed = priceFeeds[token];
        require(address(feed) != address(0), "no feed");
        (, int256 price, , ,) = feed.latestRoundData(); // e.g., 8 decimals
        uint8 decimals = feed.decimals();
        // normalize: amount * price / (10 ** tokenDecimals) and scale to 1e18
        // Omitted token decimals lookup & math for brevity
        return uint256(price) * amount; // PLACEHOLDER - implement properly
    }
}
