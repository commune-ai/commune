// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;

import {IERC721} from "interfaces/token/ERC721/IERC721.sol";
import {NFTOwnerState,NFTState, NFTContractState} from "contracts/model/portfolio/token/Structs.sol";
/**
 * @dev Implementation of https://eips.ethereum.org/EIPS/eip-721[ERC721] Non-Fungible Token Standard, including
 * the Metadata extension, but not including the Enumerable extension, which is available separately as
* {ERC721Enumerable}.
 */


interface IDepositNFT is  IERC721 {
    function getOwnerState(address owner) external view returns(NFTOwnerState memory);
    function getTokenState(uint256 tokenId) external view  returns(NFTState memory);
    function updateTokenState(uint256 tokenId, NFTState memory _tokenState) external ;
    function mint(address to, NFTState memory tokenState) external;
    function burn(uint256 tokenId) external;
    function getOwnerMarketValue(address owner) external view  returns (uint256) ;
    function getOwnerDepositValue(address owner) external view returns (uint256) ;
    function ownerTokenCount(address owner) external view  returns (uint256) ;
    // function tokenCount() external view  returns (uint256) ;
    // function ownerCount() external view  returns (uint256) ;
    function getAllOwnerTokenStates(address owner) external view returns(NFTState[] memory);

}
