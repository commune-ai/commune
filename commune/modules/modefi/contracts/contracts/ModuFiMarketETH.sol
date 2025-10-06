// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { IOracleAdapter } from "./IOracleAdapter.sol";
import { IRuleset } from "./rules/IRuleset.sol";

contract ReentrancyGuard { uint256 private _g; modifier nonReentrant(){ require(_g==0,"REENT"); _g=1; _; _g=0; } }

/// @title ModuFiMarketETH — ETH-stake market with modular oracle + ruleset
contract ModuFiMarketETH is ReentrancyGuard {
    // ---- Immutable config ----
    address public immutable creator; address public immutable protocolTreasury; uint16 public immutable protocolFeeBps; uint16 public immutable creatorFeeBps;
    address public immutable adapter; bytes public immutable oracleConfig; IRuleset public immutable ruleset; uint64 public immutable endTime; uint256 public immutable maxDelayAfterEnd; uint256 public immutable minPlayers;

    // ---- State ----
    struct Bet { int256 guess; uint256 amount; bool claimed; }
    mapping(address=>Bet) public bets; address[] public players; uint256 public totalStaked; bool public finalized; int256 public resolvedAnswer; uint256 public resolvedTime; bytes32 public resolvedOracleId;

    event BetPlaced(address indexed p, int256 guess, uint256 amount); event Finalized(int256 answer1e8, uint256 publishTime, bytes32 oracleId); event Claimed(address indexed p, uint256 amount);

    constructor(address _creator,address _treasury,uint16 _feeP,uint16 _feeC,address _adapter,bytes memory _cfg,address _ruleset,uint64 _end,uint256 _maxDelay,uint256 _minPlayers){
        require(_creator!=address(0)&&_treasury!=address(0)&&_adapter!=address(0)&&_ruleset!=address(0),"zero");
        require(_feeP+_feeC<=2000,"fee cap"); require(_minPlayers>=2,"min players");
        creator=_creator; protocolTreasury=_treasury; protocolFeeBps=_feeP; creatorFeeBps=_feeC; adapter=_adapter; oracleConfig=_cfg; ruleset=IRuleset(_ruleset); endTime=_end; maxDelayAfterEnd=_maxDelay; minPlayers=_minPlayers;
    }

    function placeBetETH(int256 guess) external payable nonReentrant { require(block.timestamp<endTime,"closed"); require(bets[msg.sender].amount==0,"one bet"); require(msg.value>0,"stake>0"); bets[msg.sender]=Bet({guess:guess,amount:msg.value,claimed:false}); players.push(msg.sender); totalStaked+=msg.value; emit BetPlaced(msg.sender,guess,msg.value); }

    function finalize(bytes calldata auxData) external payable nonReentrant {
        require(!finalized,"finalized"); require(block.timestamp>=endTime,"early"); require(players.length>=minPlayers,"few players");
        (int256 price, uint256 ts, bytes32 oid) = IOracleAdapter(adapter).resolve{value: msg.value}(oracleConfig,endTime,maxDelayAfterEnd,auxData);
        require(price>0,"bad price"); finalized=true; resolvedAnswer=price; resolvedTime=ts; resolvedOracleId=oid; emit Finalized(price,ts,oid);
    }

    function claim() external nonReentrant {
        Bet storage b = bets[msg.sender]; require(b.amount>0,"no bet"); require(!b.claimed,"claimed"); require(finalized,"not fin");
        // Build arrays for ruleset
        uint n = players.length; address[] memory addrs = new address[](n); int256[] memory gs = new int256[](n); uint256[] memory sts = new uint256[](n);
        for (uint i=0;i<n;i++){ address p = players[i]; Bet storage bb = bets[p]; addrs[i]=p; gs[i]=bb.guess; sts[i]=bb.amount; }
        (uint256[] memory payouts, ) = ruleset.computeWinners(addrs, gs, sts, resolvedAnswer);
        // First claiming winner pays fees once
        (uint256 feeP, uint256 feeC, uint256 prize) = _feesAndPrize();
        if (feeP+feeC>0 && address(this).balance>=totalStaked) { (bool ok1,) = protocolTreasury.call{value:feeP}(""); require(ok1,"feeP"); if (creatorFeeBps>0){ (bool ok2,) = creator.call{value:feeC}(""); require(ok2,"feeC"); } }
        uint idx = _indexOf(msg.sender); uint256 owed = (payouts.length>idx) ? payouts[idx] : 0; b.claimed=true; if (owed>0){ (bool ok,) = msg.sender.call{value:owed}(""); require(ok,"xfer"); emit Claimed(msg.sender,owed); } else { emit Claimed(msg.sender,0); }
    }

    function _indexOf(address who) internal view returns (uint idx){ for (uint i=0;i<players.length;i++) if (players[i]==who) return i; return type(uint).max; }
    function _feesAndPrize() internal view returns (uint256 feeP,uint256 feeC,uint256 prize){ uint256 totalFee=(totalStaked*(protocolFeeBps+creatorFeeBps))/10_000; feeP=(totalStaked*protocolFeeBps)/10_000; feeC=(totalStaked*creatorFeeBps)/10_000; prize=totalStaked-totalFee; }
}
