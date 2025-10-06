// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IOracleAdapter } from "../IOracleAdapter.sol";

interface IPyth {
    function getUpdateFee(bytes[] calldata priceUpdateData) external view returns (uint256);
    function updatePriceFeeds(bytes[] calldata priceUpdateData) external payable;
    function getPriceUnsafe(bytes32 id) external view returns (int64 price, uint64 conf, int32 expo, uint publishTime);
}

/// oracleConfig = abi.encode(address pyth, bytes32 priceId)
/// auxData       = abi.encode(bytes[] priceUpdatePayloads)
contract PythAdapter is IOracleAdapter {
    function resolve(bytes calldata oracleConfig, uint64 endTime, uint256 maxDelayAfterEnd, bytes calldata auxData)
        external payable override returns (int256 price1e8, uint256 publishTime, bytes32 oracleId)
    {
        (address pythAddr, bytes32 priceId) = abi.decode(oracleConfig,(address,bytes32));
        IPyth pyth = IPyth(pythAddr);
        bytes[] memory updates = abi.decode(auxData,(bytes[]));
        if (updates.length > 0) { uint256 fee = pyth.getUpdateFee(updates); require(msg.value >= fee,"pyth fee"); pyth.updatePriceFeeds{value:fee}(updates); }
        (int64 px,, int32 expo, uint pt) = pyth.getPriceUnsafe(priceId);
        int256 scaled; if (expo <= 0) { scaled = int256(px) * int256(10 ** uint32(uint32(8)+uint32(-expo))); } else { scaled = int256(px) / int256(10 ** uint32(uint32(expo)-uint32(8))); }
        require(pt >= endTime && pt <= endTime + maxDelayAfterEnd, "time");
        price1e8 = scaled; publishTime = pt; oracleId = priceId;
    }
    function earliestFinalizeTime(bytes calldata, uint64 endTime) external pure returns (uint256) { return endTime; }
}
