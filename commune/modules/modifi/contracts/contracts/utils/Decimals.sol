// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

library Decimals {
    function to1e8(int256 price, uint8 fromDec) internal pure returns (int256) {
        if (fromDec == 8) return price;
        if (fromDec > 8) return price / int256(10 ** (fromDec - 8));
        return price * int256(10 ** (8 - fromDec));
    }
}
