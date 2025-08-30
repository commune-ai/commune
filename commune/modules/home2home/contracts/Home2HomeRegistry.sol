 # start of file
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/Ownable.sol";
import "./RentToOwnAgreement.sol";

/**
 * @title Home2HomeRegistry
 * @dev Central registry for all Home2Home properties and agreements
 */
contract Home2HomeRegistry is Ownable {
    struct PropertyRecord {
        address tokenAddress;
        string propertyAddress;
        uint256 propertyValue;
        bool active;
    }
    
    // Mappings
    mapping(address => PropertyRecord) public properties;
    mapping(address => address[]) public tenantAgreements;
    mapping(address => bool) public approvedManagers;
    
    // Events
    event PropertyRegistered(address tokenAddress, string propertyAddress);
    event AgreementCreated(address agreementAddress, address propertyToken, address tenant);
    
    /**
     * @dev Register a new tokenized property
     */
    function registerProperty(
        address tokenAddress,
        string memory propertyAddress,
        uint256 propertyValue
    ) external onlyOwner {
        properties[tokenAddress] = PropertyRecord({
            tokenAddress: tokenAddress,
            propertyAddress: propertyAddress,
            propertyValue: propertyValue,
            active: true
        });
        
        emit PropertyRegistered(tokenAddress, propertyAddress);
    }
    
    /**
     * @dev Create a new rent-to-own agreement
     */
    function createAgreement(
        address propertyTokenAddress,
        address tenant,
        uint256 monthlyPayment,
        uint256 equityPercentage,
        uint256 maintenancePercentage,
        uint256 durationInMonths
    ) external returns (address) {
        require(properties[propertyTokenAddress].active, "Property not registered");
        
        RentToOwnAgreement agreement = new RentToOwnAgreement(
            propertyTokenAddress,
            tenant,
            monthlyPayment,
            equityPercentage,
            maintenancePercentage,
            durationInMonths
        );
        
        tenantAgreements[tenant].push(address(agreement));
        
        emit AgreementCreated(address(agreement), propertyTokenAddress, tenant);
        
        return address(agreement);
    }
    
    /**
     * @dev Add or remove approved property managers
     */
    function setPropertyManager(address manager, bool approved) external onlyOwner {
        approvedManagers[manager] = approved;
    }
    
    /**
     * @dev Get all agreements for a tenant
     */
    function getTenantAgreements(address tenant) external view returns (address[] memory) {
        return tenantAgreements[tenant];
    }
}
