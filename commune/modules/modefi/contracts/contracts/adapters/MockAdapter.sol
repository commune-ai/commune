// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IOracleAdapter } from "../IOracleAdapter.sol";

/// LOCAL DEV ONLY: settable price & publish time.
contract MockAdapter is IOracleAdapter {
    struct Feed { int256 price1e8; uint256 publishTime; }
    mapping(bytes32 => Feed) public feeds; // oracleId -> feed

    function resolve(bytes calldata oracleConfig, uint64 endTime, uint256 maxDelayAfterEnd, bytes calldata)
        external payable returns (int256, uint256, bytes32)
    {
        bytes32 id = abi.decode(oracleConfig,(bytes32)); Feed memory f = feeds[id];
        require(f.price1e8 > 0, "unset"); require(f.publishTime >= endTime && f.publishTime <= endTime + maxDelayAfterEnd, "time");
        return (f.price1e8, f.publishTime, id);
    }
    function earliestFinalizeTime(bytes calldata, uint64 endTime) external pure returns (uint256) { return endTime; }
    function devSet(bytes32 id, int256 price1e8, uint256 publishTime) external { feeds[id] = Feed(price1e8, publishTime); }
}
