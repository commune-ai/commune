

pragma solidity ^0.8.7;
import "contracts/utils/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

abstract contract StakerAccessControl is AccessControl {


    /* ############ STAKING PORTION  ###############*/

    // track the staked balances for each account
    mapping(address=>uint256) public _stakeBalance; 
    uint256 public totalStakeBalance;

    // minimum amount for people to stake (over 0)
    uint256 public minStakeAmount = 0;

    // for staking the model nft 
    // user calleds contract.StakeModel() with sent ether
    function stake() external payable {
        require(msg.value > minStakeAmount, 
                    "stake value does not meet expectations");
        _grantRole("staker", msg.sender);
        _stakeBalance[msg.sender] += msg.value;
        totalStakeBalance += msg.value;
    }


    // internal function for handling priveledges ()
    function _unstake(uint256 amount) internal {

        if (amount > _stakeBalance[msg.sender] ) {
            amount = _stakeBalance[msg.sender];
        }

        _stakeBalance[msg.sender] -= amount;
        totalStakeBalance -= amount; 
        if (_stakeBalance[msg.sender] == 0) {
            _revokeRole("staker", msg.sender);
        }    
    }


    // unstake up to the staker's balance
    function unstake(uint256 amount) external payable {
        _unstake(amount);
        payable(msg.sender).transfer(amount);
    }

    // unstake everything from msg.sender staked on model
    function unstakeAll() external payable {
        _unstake(_stakeBalance[msg.sender]);
        payable(msg.sender).transfer(_stakeBalance[msg.sender]);
    } 
    // get the staked value from msg.sender
    function getStakeValue() external view returns(uint256) {
        return _stakeBalance[msg.sender];
    }


    function getStakeRatio() external view returns(uint256) {
        uint256 stakeRatio = 0;
        if (totalStakeBalance>0) {
            stakeRatio =  (_stakeBalance[msg.sender] * 1000)/ totalStakeBalance;
        }
        return stakeRatio;
    }


}