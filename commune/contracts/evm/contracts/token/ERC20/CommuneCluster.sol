// // SPDX-License-Identifier: MIT
// // OpenZeppelin Contracts (last updated v4.7.0) (token/ERC20/ERC20.sol)

// pragma solidity ^0.8.0;


// import "contracts/openzeppelin/token/ERC20/IERC20.sol";
// import "contracts/openzeppelin/token/ERC20/extensions/IERC20Metadata.sol";
// import "contracts/openzeppelin/utils/Context.sol";

// /**
//  * @dev Implementation of the {IERC20} interface.
//  *
//  * This implementation is agnostic to the way tokens are created. This means
//  * that a supply mechanism has to be added in a derived contract using {_mint}.
//  * For a generic mechanism see {ERC20PresetMinterPauser}.
//  *
//  * TIP: For a detailed writeup see our guide
//  * https://forum.openzeppelin.com/t/how-to-implement-erc20-supply-mechanisms/226[How
//  * to implement supply mechanisms].
//  *
//  * We have followed general OpenZeppelin Contracts guidelines: functions revert
//  * instead returning `false` on failure. This behavior is nonetheless
//  * conventional and does not conflict with the expectations of ERC20
//  * applications.
//  *
//  * Additionally, an {Approval} event is emitted on calls to {transferFrom}.
//  * This allows applications to reconstruct the allowance for all accounts just
//  * by listening to said events. Other implementations of the EIP may not emit
//  * these events, as it isn't required by the specification.
//  *
//  * Finally, the non-standard {decreaseAllowance} and {increaseAllowance}
//  * functions have been added to mitigate the well-known issues around setting
//  * allowances. See {IERC20-approve}.
//  */
// enum Resource{ GPU, CPU, TPU }

// // 
// struct ResourceStruct {
//     uint256 rate; 
// }

// struct ActorStruct {
//     uint cpus ; // number of cpus for actor
//     uint gpus ; // number of gpus
//     string name; // 
//     string meta; // meta hash 
//     uint start_time;
//     uint period;
// }

// struct DevState {
//     uint256 stake;
//     uint256 score;
//     uint256 block_number;
// }

// struct VoteState {
//     uint256 score;
//     uint256 block_number;
// }

// contract ModelToken is Context, IERC20, IERC20Metadata {
//     mapping(address => uint256) private _balances;

//     mapping(address => mapping(address => uint256)) private _allowances;

//     uint256 private _totalSupply;

//     string private _name;
//     string private _symbol;
//     address[] public devs;
//     mapping(address=>DevState) public dev2state;

//     mapping(address=>address[]) public voter2devs;
//     mapping(address=>address[]) public dev2voters;
    
//     ERC20Manager public token_manager = new ERC20Manager();

//     // number of block
//     uint256 block_step = 2;

//     /**
//      * @dev Sets the values for {name} and {symbol}.
//      *
//      * The default value of {decimals} is 18. To select a different value for
//      * {decimals} you should overload it.
//      *
//      * All two of these values are immutable: they can only be set once during
//      * construction.
//      */

//     constructor(string memory name_, string memory symbol_) {
//         _name = name_;
//         _symbol = symbol_;
        
//         // _mint(msg.sender, 1000000000 );
//     }

//     /**
//      * @dev Returns the name of the token.
//      */
//     function name() public view virtual override returns (string memory) {
//         return _name;
//     }

//     // addes stake to model

//     function  is_dev(address dev) public view returns(bool){
//         return bool(dev2voters[dev].length > 0);
//     }

//     function _ensure_dev(address dev) internal {
//         // push dev to devs if it doesnt exist
//         if (!is_dev(dev)) {
//             devs.push(dev);
//         }
//     }

//     function add_stake(uint256 amount)  public {
         
//         _allowances[msg.sender][address(this)] +=  amount;
//         dev2state[msg.sender].stake = allowance(msg.sender,address(this));
//         _ensure_dev(msg.sender);
//     }


        
//     function deposit() payable external {
//         // record the value sent 
//         // to the address that sent it
//         balances[msg.sender] += msg.value;
//     }
//    function withdraw(uint amount) public {
        
//         uint amount = pendingWithdrawals[msg.sender];
//         pendingWithdrawals[msg.sender] = 0;
//         msg.sender.transfer(amount);
//    }

//     uint256 public  TOKENS_PER_CALL = 1;
//     mapping(address=>uint256) public users2calls;

//     function set_tokens_per_call(uint256 ratio) internal returns (uint256){
        
//         TOKENS_PER_CALL = ratio;
//         return TOKENS_PER_CALL;
//     }

//     function bill_user_per_call(address user ) public {
//         require(_balances[user] > TOKENS_PER_CALL);
//         transferFrom(msg.sender, address(this), TOKENS_PER_CALL);
//         users2calls[msg.sender] += 1;
//     }

//     function get_stake(address user) public view returns(uint256){
//         return _allowances[user][address(this)];
//     }

//     function get_score(address user) public view returns(uint256 score){
//         for (uint i; i< dev2voters[user].length; i++) {
//             address voter = dev2voters[user][i];
//             score = score + voter2dev_vote_map[voter][user].score;
//         }

//         return score;
//     }

//     function my_score() public view returns(uint256 score){
//         score = get_score(msg.sender);
//     }

//     function my_stake() public view returns(uint256 stake){
//         stake = get_stake(msg.sender);

//         return stake;
//     }

//     function remove_stake(uint256 amount) public  returns (uint256 removed_amount){
//         if (amount > _allowances[msg.sender][address(this)]) {
//             amount = _allowances[msg.sender][address(this)];
//         }
//         removed_amount = _allowances[msg.sender][address(this)] -= amount;
//         return removed_amount;
//     }

//     uint256 public constant PERCENT_BASE = 10000;
//     uint256 public ALPHA = 0;

//     function set_alpha(uint256 alpha) public {
//         // set alpha
//         require(alpha < PERCENT_BASE, 'ALPHA needs to be Lower than Percent Base');  
//         ALPHA = alpha;
//     }

//     function set_votes(uint256[] memory votes,  address[] memory devs) public {
        
//         require(votes.length == devs.length, 'votes and devs have to be same length bro');
//         require(votes.length<PERCENT_BASE, 'the size of you voters is too big fam');
//         require(dev2state[msg.sender].stake>0, 'you need to stake greater than 0');

//         uint256 total_score = 0;
//         uint256 current_score;
//         uint256[] memory previous_normalized_scores = new uint256[](votes.length);
//         address voter = msg.sender;
//         address dev;


//         // get the previous votes and delete the map elements
//         for (uint i; i < devs.length; i++) {
//             previous_normalized_scores[i] = voter2dev_vote_map[voter][devs[i]].score;
//             delete voter2dev_vote_map[voter][dev];
//         }
//         // remove existing vote registries
//         // iterate through the devs
//         for (uint d_i; d_i< voter2devs[voter].length; d_i++) {
//             dev = voter2devs[voter][d_i];
//             // iterate through the voters
//             uint num_voters_for_dev = dev2voters[dev].length;
//             for (uint v_i; v_i<num_voters_for_dev; v_i++) {
//                 // remove the the voter from the dev registery
//                 if (voter == dev2voters[dev][v_i]) {
//                     // delete the voter from the dev2voters registery
//                     dev2voters[dev][v_i] = dev2voters[dev][dev2voters[dev].length-1];
//                     delete dev2voters[dev][dev2voters[dev].length-1];
//                     break;
//                 }
//             }
//         } 
        
//         // purge devs for voter (not efficient but it works)
//         delete voter2devs[voter];

//         for (uint i; i<votes.length; i++) {
//             dev = devs[i];
//             current_score = (votes[i] * dev2state[voter].stake) ;
//             voter2dev_vote_map[voter][dev] = VoteState({block_number: block.number, score: current_score });
//             total_score += current_score;

//         }

//         for (uint i; i<votes.length; i++) {
//             dev = devs[i];
//             _ensure_dev(dev);
//             voter2dev_vote_map[voter][dev].score = (voter2dev_vote_map[voter][dev].score*PERCENT_BASE)/(total_score);
//             voter2dev_vote_map[voter][dev].score  = ((previous_normalized_scores[i] * ALPHA + voter2dev_vote_map[voter][dev].score * (PERCENT_BASE - ALPHA) ) / PERCENT_BASE);


//             // create new registries
//             if (voter2dev_vote_map[voter][dev].score > 0 ) {
//                 voter2devs[voter].push(dev);
//                 dev2voters[dev].push(voter);
//             } 

//         }



//     }

//     function symbol() public view virtual override returns (string memory) {
//         return _symbol;
//     }

//     function decimals() public view virtual override returns (uint8) {
//         return 18;
//     }


//     function totalSupply() public view virtual override returns (uint256) {
//         return _totalSupply;
//     }

 
//     function balanceOf(address account) public view virtual override returns (uint256) {
//         return _balances[account];
//     }


//     function transfer(address to, uint256 amount) public virtual override returns (bool) {
//         address owner = _msgSender();
//         _transfer(owner, to, amount);
//         return true;
//     }


//     function allowance(address owner, address spender) public view virtual override returns (uint256) {
//         return _allowances[owner][spender];
//     }

//     function approve(address spender, uint256 amount) public virtual override returns (bool) {
//         address owner = _msgSender();
//         _approve(owner, spender, amount);
//         return true;
//     }

//     function transferFrom(
//         address from,
//         address to,
//         uint256 amount
//     ) public virtual override returns (bool) {
//         address spender = _msgSender();
//         // _spendAllowance(from, spender, amount);
//         _transfer(from, to, amount);
//         return true;
//     }

//     function increaseAllowance(address spender, uint256 addedValue) public virtual returns (bool) {
//         address owner = _msgSender();
//         _approve(owner, spender, allowance(owner, spender) + addedValue);
//         return true;
//     }

//     function decreaseAllowance(address spender, uint256 subtractedValue) public virtual returns (bool) {
//         address owner = _msgSender();
//         uint256 currentAllowance = allowance(owner, spender);
//         require(currentAllowance >= subtractedValue, "ERC20: decreased allowance below zero");
//         unchecked {
//             _approve(owner, spender, currentAllowance - subtractedValue);
//         }

//         return true;
//     }

//     function _transfer(
//         address from,
//         address to,
//         uint256 amount
//     ) internal virtual {
//         require(from != address(0), "ERC20: transfer from the zero address");
//         require(to != address(0), "ERC20: transfer to the zero address");

//         _beforeTokenTransfer(from, to, amount);

//         uint256 fromBalance = _balances[from];
//         require(fromBalance >= amount, "ERC20: transfer amount exceeds balance");
//         unchecked {
//             _balances[from] = fromBalance - amount;
//             // Overflow not possible: the sum of all balances is capped by totalSupply, and the sum is preserved by
//             // decrementing then incrementing.
//             _balances[to] += amount;
//         }

//         emit Transfer(from, to, amount);

//         _afterTokenTransfer(from, to, amount);
//     }



//     function mint(address account, uint256 amount) public returns (uint256) {
//         _mint(account, amount);
//         return balanceOf(msg.sender);
//     }
//     function _mint(address account, uint256 amount) internal virtual {
//         require(account != address(0), "ERC20: mint to the zero address");

//         _beforeTokenTransfer(address(0), account, amount);

//         _totalSupply += amount;
//         unchecked {
//             // Overflow not possible: balance + amount is at most totalSupply + amount, which is checked above.
//             _balances[account] += amount;
//         }
//         emit Transfer(address(0), account, amount);

//         _afterTokenTransfer(address(0), account, amount);
//     }

//     function _burn(address account, uint256 amount) internal virtual {
//         require(account != address(0), "ERC20: burn from the zero address");

//         _beforeTokenTransfer(account, address(0), amount);

//         uint256 accountBalance = _balances[account];
//         require(accountBalance >= amount, "ERC20: burn amount exceeds balance");
//         unchecked {
//             _balances[account] = accountBalance - amount;
//             // Overflow not possible: amount <= accountBalance <= totalSupply.
//             _totalSupply -= amount;
//         }

//         emit Transfer(account, address(0), amount);

//         _afterTokenTransfer(account, address(0), amount);
//     }

//     function _approve(
//         address owner,
//         address spender,
//         uint256 amount
//     ) internal virtual {
//         require(owner != address(0), "ERC20: approve from the zero address");
//         require(spender != address(0), "ERC20: approve to the zero address");

//         _allowances[owner][spender] = amount;
//         emit Approval(owner, spender, amount);
//     }

//     function _spendAllowance(
//         address owner,
//         address spender,
//         uint256 amount
//     ) public virtual {
//         uint256 currentAllowance = allowance(owner, spender);
//         if (currentAllowance != type(uint256).max) {
//             require(currentAllowance >= amount, "ERC20: insufficient allowance");
//             unchecked {
//                 _approve(owner, spender, currentAllowance - amount);
//             }
//         }
//     }

//     function _beforeTokenTransfer(
//         address from,
//         address to,
//         uint256 amount
//     ) internal virtual {}

//     /**
//      * @dev Hook that is called after any transfer of tokens. This includes
//      * minting and burning.
//      *
//      * Calling conditions:
//      *
//      * - when `from` and `to` are both non-zero, `amount` of ``from``'s tokens
//      * has been transferred to `to`.
//      * - when `from` is zero, `amount` tokens have been minted for `to`.
//      * - when `to` is zero, `amountba` of ``from``'s tokens have been burned.
//      * - `from` and `to` are never both zero.
//      *
//      * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
//      */
//     function _afterTokenTransfer(
//         address from,
//         address to,
//         uint256 amount
//     ) internal virtual {}



// }
