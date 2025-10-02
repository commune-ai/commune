// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IOracleAdapter } from "../IOracleAdapter.sol";
import { Decimals } from "../utils/Decimals.sol";

interface AggregatorV3Interface {
    function decimals() external view returns (uint8);
    function latestRoundData() external view returns (
        uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound
    );
    function getRoundData(uint80 _roundId) external view returns (
        uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound
    );
}

/// oracleConfig = abi.encode(address feed, bool scanForwardFromHint, uint80 roundHint)
contract ChainlinkAdapter is IOracleAdapter {
    using Decimals for int256;
    function resolve(bytes calldata oracleConfig, uint64 endTime, uint256 maxDelayAfterEnd, bytes calldata)
        external payable override returns (int256 price1e8, uint256 publishTime, bytes32 oracleId)
    {
        (address feedAddr, bool scan, uint80 hint) = abi.decode(oracleConfig,(address,bool,uint80));
        AggregatorV3Interface feed = AggregatorV3Interface(feedAddr);
        uint8 dec = feed.decimals();
        uint80 rid; int256 ans; uint256 ts; uint80 aIR;
        if (scan) {
            if (hint == 0) { (rid,, , ,) = feed.latestRoundData(); } else { rid = hint; }
            while (true) { (rid, ans, , ts, aIR) = feed.getRoundData(rid); if (ts >= endTime) break; unchecked { rid++; } }
        } else { (rid, ans, , ts, aIR) = feed.latestRoundData(); }
        require(ts >= endTime && ts <= endTime + maxDelayAfterEnd, "time");
        require(ans > 0 && aIR >= rid, "round");
        price1e8 = ans.to1e8(dec); publishTime = ts; oracleId = bytes32(uint256(uint160(feedAddr)));
    }
    function earliestFinalizeTime(bytes calldata, uint64 endTime) external pure returns (uint256) { return endTime; }
}
