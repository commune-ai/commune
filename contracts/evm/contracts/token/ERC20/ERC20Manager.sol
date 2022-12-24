// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.7.0) (token/ERC20/ERC20.sol)

pragma solidity ^0.8.0;


import "contracts/openzeppelin/token/ERC20/IERC20.sol";
import "contracts/openzeppelin/token/ERC20/extensions/IERC20Metadata.sol";
import "contracts/openzeppelin/utils/Context.sol";

/**
 * @dev Implementation of the {IERC20} interface.
 *
 * This implementation is agnostic to the way tokens are created. This means
 * that a supply mechanism has to be added in a derived contract using {_mint}.
 * For a generic mechanism see {ERC20PresetMinterPauser}.
 *
 * TIP: For a detailed writeup see our guide
 * https://forum.openzeppelin.com/t/how-to-implement-erc20-supply-mechanisms/226[How
 * to implement supply mechanisms].
 *
 * We have followed general OpenZeppelin Contracts guidelines: functions revert
 * instead returning `false` on failure. This behavior is nonetheless
 * conventional and does not conflict with the expectations of ERC20
 * applications.
 *
 * Additionally, an {Approval} event is emitted on calls to {transferFrom}.
 * This allows applications to reconstruct the allowance for all accounts just
 * by listening to said events. Other implementations of the EIP may not emit
 * these events, as it isn't required by the specification.
 *
 * Finally, the non-standard {decreaseAllowance} and {increaseAllowance}
 * functions have been added to mitigate the well-known issues around setting
 * allowances. See {IERC20-approve}.
 */
contract ERC20Manager {


    function totalSupply(address token) public view virtual returns (uint256) {
        return IERC20(token).totalSupply();
    }
    
    /**
     * @dev See {IERC20-balanceOf}.
     */
    function balanceOf(address token, address account) public view virtual returns (uint256) {
        return IERC20(token).balanceOf(account);
    }

    /**
     * @dev See {IERC20-allowance}.
     */

    function allowance(address token, address owner, address spender) public view virtual returns (uint256) {
        return IERC20(token).allowance(owner, spender);
    }

}
