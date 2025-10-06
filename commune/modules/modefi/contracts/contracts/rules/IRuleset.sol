// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IRuleset {
    /// @dev Return per-player payout amounts and a short description string.
    function computeWinners(
        address[] calldata players,
        int256[] calldata guesses,
        uint256[] calldata stakes,
        int256 resolvedAnswer
    ) external view returns (uint256[] memory payouts, string memory description);
}
