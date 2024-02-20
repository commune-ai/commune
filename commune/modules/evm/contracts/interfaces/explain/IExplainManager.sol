// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.7;

import {ExplainSchema} from "contracts/explain/Structs.sol";
    

interface IExplainManager {
    // add explainer object to Modules List
    // replaces existing name
    function addExplain(string memory name , string memory explainURI) external ;
    function removeExplain(string memory name )  external ;
    function getExplainer(string memory name) external view returns(ExplainSchema memory );
    function getExplainers() external view returns(ExplainSchema[] memory );
}
