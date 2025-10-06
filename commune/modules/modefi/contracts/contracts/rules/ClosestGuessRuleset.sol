// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import { IRuleset } from "./IRuleset.sol";

contract ClosestGuessRuleset is IRuleset {
    function computeWinners(address[] calldata players, int256[] calldata guesses, uint256[] calldata stakes, int256 resolved)
        external pure returns (uint256[] memory payouts, string memory description)
    {
        payouts = new uint256[](players.length);
        uint256 best = type(uint256).max; uint256 winners; uint256 total;
        for (uint i=0;i<players.length;i++){ total += stakes[i]; uint256 d = _abs(resolved, guesses[i]); if (d < best){ best=d; winners=1; } else if (d==best){ winners++; } }
        if (winners==0) return (payouts, "No winner");
        uint256 share = total / winners;
        for (uint i=0;i<players.length;i++) if (_abs(resolved,guesses[i])==best) payouts[i]=share;
        description = "Closest guess wins; ties split";
    }
    function _abs(int256 a,int256 b) internal pure returns(uint256){ return a>=b?uint256(a-b):uint256(b-a); }
}
