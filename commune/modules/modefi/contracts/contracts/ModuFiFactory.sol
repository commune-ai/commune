// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import { Registry } from "./Registry.sol";
import { ModuFiMarketETH } from "./ModuFiMarketETH.sol";

contract ModuFiFactory {
    event MarketCreated(address indexed market, address indexed creator, address adapter, address ruleset);

    address public protocolTreasury; uint16 public protocolFeeBps; Registry public immutable registry;

    constructor(address _treasury, uint16 _feeBps, address _registry){ require(_treasury!=address(0)&&_registry!=address(0),"zero"); require(_feeBps<=2000,"fee>20"); protocolTreasury=_treasury; protocolFeeBps=_feeBps; registry=Registry(_registry); }

    function setProtocolFee(uint16 bps) external { require(msg.sender==protocolTreasury,"only"); require(bps<=2000,"fee>20"); protocolFeeBps=bps; }
    function setTreasury(address t) external { require(msg.sender==protocolTreasury,"only"); require(t!=address(0),"0"); protocolTreasury=t; }

    function createETHMarketModular(address adapter, bytes calldata oracleConfig, address ruleset, uint64 endTime, uint16 creatorFeeBps, uint256 maxDelayAfterEnd, uint256 minPlayers)
        external returns (address mkt)
    {
        require(adapter!=address(0)&&ruleset!=address(0),"zero"); require(endTime>block.timestamp+5 minutes,"soon"); require(creatorFeeBps+protocolFeeBps<=2000,"fee cap"); require(minPlayers>=2,"minPlayers");
        mkt = address(new ModuFiMarketETH(msg.sender, protocolTreasury, protocolFeeBps, creatorFeeBps, adapter, oracleConfig, ruleset, endTime, maxDelayAfterEnd, minPlayers));
        registry.add(mkt, msg.sender); emit MarketCreated(mkt,msg.sender,adapter,ruleset);
    }
}
