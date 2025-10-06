// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import { IRuleset } from "./IRuleset.sol";

/// @notice Winners are guesses within ±toleranceBps of resolved price; ties split.
contract RangeBettingRuleset is IRuleset {
    int256 public immutable toleranceBps; // e.g., 500 = 5%
    constructor(int256 _tol) { toleranceBps = _tol; }
    function computeWinners(address[] calldata players, int256[] calldata guesses, uint256[] calldata stakes, int256 resolved)
        external view returns (uint256[] memory payouts, string memory description)
    {
        payouts = new uint256[](players.length); uint256 total; for (uint i=0;i<stakes.length;i++) total+=stakes[i];
        int256 low = resolved - (resolved * toleranceBps / 10000); int256 high = resolved + (resolved * toleranceBps / 10000);
        uint count; for (uint i=0;i<guesses.length;i++) if (guesses[i]>=low && guesses[i]<=high) count++;
        if (count==0) return (payouts, "No winner in range");
        uint256 share = total / count; for (uint i=0;i<guesses.length;i++) if (guesses[i]>=low && guesses[i]<=high) payouts[i]=share;
        description = "Within ±range wins; ties split";
    }
}
