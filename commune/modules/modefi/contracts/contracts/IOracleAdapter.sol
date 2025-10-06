// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice Adapters return prices scaled to 1e8 plus publishTime.
interface IOracleAdapter {
    function resolve(
        bytes calldata oracleConfig,
        uint64 endTime,
        uint256 maxDelayAfterEnd,
        bytes calldata auxData
    ) external payable returns (int256 price1e8, uint256 publishTime, bytes32 oracleId);

    function earliestFinalizeTime(bytes calldata oracleConfig, uint64 endTime) external view returns (uint256);
}
