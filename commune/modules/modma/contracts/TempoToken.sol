// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address) external view returns (uint256);
    function allowance(address,address) external view returns (uint256);
    function approve(address,uint256) external returns (bool);
    function transfer(address,uint256) external returns (bool);
    function transferFrom(address,address,uint256) external returns (bool);
}

library SafeERC20 {
    function safeTransferFrom(IERC20 t, address from, address to, uint256 v) internal {
        bool ok = t.transferFrom(from, to, v);
        require(ok, "TRANSFER_FROM_FAIL");
    }
    function safeTransfer(IERC20 t, address to, uint256 v) internal {
        bool ok = t.transfer(to, v);
        require(ok, "TRANSFER_FAIL");
    }
}

contract TempoToken {
    using SafeERC20 for IERC20;

    error TransfersDisabled();
    error EpochNotActive();
    error EpochNotEnded();
    error AlreadyPinged();
    error ZeroAmount();
    error NothingToRedeem();
    error TooEarly();
    error BadBounds();

    event EpochStarted(uint256 indexed epochId, uint40 startsAt, uint40 endsAt);
    event EpochSettled(uint256 indexed epochId, uint256 pot, uint256 totalTempo);
    event LiquidityProvided(address indexed provider, uint256 indexed epochId, uint256 amount);
    event Ping(address indexed miner, uint256 indexed epochId, uint256 minted);
    event Redeem(address indexed miner, uint256 indexed epochId, uint256 burned, uint256 payout);

    string public constant name = "Tempo";
    string public constant symbol = "TEMPO";
    uint8  public constant decimals = 18;

    IERC20 public immutable baseAsset;
    uint40 public immutable minEpoch;
    uint40 public immutable maxEpoch;
    uint40 public immutable tempoTarget;
    uint16 public immutable tempoJitterBps;
    uint128 public immutable baseEmission;
    uint128 public immutable timeWeight;

    uint256 private _totalSupply;
    mapping(address => uint256) private _balanceOf;

    uint256 public epochId;
    uint40  public epochStart;
    uint40  public epochEnd;
    uint256 public currentPot;
    mapping(uint256 => uint256) public epochPot;
    mapping(uint256 => uint256) public epochTotalSupply;

    mapping(uint256 => mapping(address => bool)) public hasPingedEpoch;
    mapping(uint256 => mapping(address => uint256)) public epochBalanceAtSettle;
    mapping(uint256 => mapping(address => bool)) public redeemed;

    constructor(
        address baseAsset_,
        uint40 minEpoch_,
        uint40 maxEpoch_,
        uint40 tempoTarget_,
        uint16 tempoJitterBps_,
        uint128 baseEmission_,
        uint128 timeWeight_
    ) {
        require(baseAsset_ != address(0), "BASE_ZERO");
        if (minEpoch_ == 0 || maxEpoch_ < minEpoch_) revert BadBounds();
        baseAsset = IERC20(baseAsset_);
        minEpoch = minEpoch_;
        maxEpoch = maxEpoch_;
        tempoTarget = tempoTarget_;
        tempoJitterBps = tempoJitterBps_;
        baseEmission = baseEmission_;
        timeWeight = timeWeight_;

        epochId = 1;
        epochStart = uint40(block.timestamp);
        epochEnd = uint40(block.timestamp + _sampleDuration());
        emit EpochStarted(epochId, epochStart, epochEnd);
    }

    function totalSupply() external view returns (uint256) { return _totalSupply; }
    function balanceOf(address a) external view returns (uint256) { return _balanceOf[a]; }
    function allowance(address, address) external pure returns (uint256) { return 0; }
    function approve(address, uint256) external pure returns (bool) { revert TransfersDisabled(); }
    function transfer(address, uint256) external pure returns (bool) { revert TransfersDisabled(); }
    function transferFrom(address, address, uint256) external pure returns (bool) { revert TransfersDisabled(); }

    function provideLiquidity(uint256 amount) external {
        if (amount == 0) revert ZeroAmount();
        baseAsset.safeTransferFrom(msg.sender, address(this), amount);
        currentPot += amount;
        emit LiquidityProvided(msg.sender, epochId, amount);
    }

    function ping() external {
        uint40 nowTs = uint40(block.timestamp);
        if (nowTs >= epochEnd) revert EpochNotActive();
        if (hasPingedEpoch[epochId][msg.sender]) revert AlreadyPinged();
        hasPingedEpoch[epochId][msg.sender] = true;

        uint256 epochLen = uint256(epochEnd) - uint256(epochStart);
        uint256 timeLeft = uint256(epochEnd) - uint256(nowTs);
        uint256 bonus = (uint256(timeWeight) * timeLeft) / epochLen;
        uint256 minted = uint256(baseEmission) + bonus;

        _mint(msg.sender, minted);
        emit Ping(msg.sender, epochId, minted);
    }

    function settleEpoch() public {
        uint40 nowTs = uint40(block.timestamp);
        if (nowTs < epochEnd) revert EpochNotEnded();
        uint256 eid = epochId;
        uint256 pot = currentPot;
        uint256 supply = _totalSupply;

        epochPot[eid] = pot;
        epochTotalSupply[eid] = supply;

        emit EpochSettled(eid, pot, supply);

        epochId = eid + 1;
        epochStart = nowTs;
        epochEnd = uint40(nowTs + _sampleDuration());
        currentPot = 0;
        _totalSupply = 0;

        emit EpochStarted(epochId, epochStart, epochEnd);
    }

    function redeem(uint256 eid) external {
        if (epochTotalSupply[eid] == 0) revert NothingToRedeem();
        if (block.timestamp <= epochEnd && eid == epochId) revert TooEarly();
        if (!hasPingedEpoch[eid][msg.sender]) revert NothingToRedeem();
        if (redeemed[eid][msg.sender]) revert NothingToRedeem();

        uint256 userBal = _epochBalanceLazySnapshot(eid, msg.sender);
        if (userBal == 0) revert NothingToRedeem();

        redeemed[eid][msg.sender] = true;
        uint256 supply = epochTotalSupply[eid];
        uint256 pot = epochPot[eid];
        uint256 payout = (pot * userBal) / supply;

        baseAsset.safeTransfer(msg.sender, payout);
        emit Redeem(msg.sender, eid, userBal, payout);
    }

    function _mint(address to, uint256 amt) internal {
        _totalSupply += amt;
        unchecked { _balanceOf[to] += amt; }
    }

    function _epochBalanceLazySnapshot(uint256 eid, address user) internal returns (uint256 snapBal) {
        snapBal = epochBalanceAtSettle[eid][user];
        if (snapBal != 0) return snapBal;
        if (eid + 1 == epochId) {
            uint256 bal = _balanceOf[user];
            if (bal != 0) {
                epochBalanceAtSettle[eid][user] = bal;
                snapBal = bal;
                _balanceOf[user] = 0;
            }
        }
    }

    function _sampleDuration() internal view returns (uint256) {
        uint256 u = uint256(block.prevrandao);
        uint256 bps = u % 10000;
        int256 jitter = int256(int32(int256(int(bps)) - 5000));
        int256 span = (int256(uint256(tempoTarget)) * int256(uint256(tempoJitterBps))) / 10000;
        int256 dur = int256(uint256(tempoTarget)) + (span * jitter) / 5000;
        if (dur < int256(uint256(minEpoch))) dur = int256(uint256(minEpoch));
        if (dur > int256(uint256(maxEpoch))) dur = int256(uint256(maxEpoch));
        return uint256(dur);
    }

    function currentEpoch() external view returns (uint256) { return epochId; }
    function epochEndsAt() external view returns (uint40) { return epochEnd; }
    function hasPinged(uint256 eid, address who) external view returns (bool) {
        return hasPingedEpoch[eid][who];
    }
}
