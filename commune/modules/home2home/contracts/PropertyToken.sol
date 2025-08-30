 # start of file
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title PropertyToken
 * @dev Represents tokenized ownership of a specific property
 */
contract PropertyToken is ERC20, Ownable {
    address public propertyManager;
    uint256 public propertyValue;
    string public propertyAddress;
    
    // Property details
    struct PropertyDetails {
        uint256 squareFeet;
        uint8 bedrooms;
        uint8 bathrooms;
        uint256 yearBuilt;
        string propertyType; // "single-family", "condo", etc.
    }
    
    PropertyDetails public details;
    
    // Maintenance fund
    uint256 public maintenanceFundBalance;
    
    // Events
    event MaintenanceFundDeposit(uint256 amount);
    event MaintenanceFundWithdrawal(uint256 amount, string reason);
    event PropertyRevalued(uint256 oldValue, uint256 newValue);
    
    constructor(
        string memory _propertyAddress,
        uint256 _propertyValue,
        address _propertyManager,
        PropertyDetails memory _details
    ) ERC20("Home2Home Property Token", "H2H") {
        propertyAddress = _propertyAddress;
        propertyValue = _propertyValue;
        propertyManager = _propertyManager;
        details = _details;
        
        // Mint initial supply to property owner
        _mint(owner(), _propertyValue * 10**decimals());
    }
    
    /**
     * @dev Deposit funds into the maintenance reserve
     */
    function depositToMaintenanceFund(uint256 amount) external payable {
        require(msg.value == amount, "Amount must match sent value");
        maintenanceFundBalance += amount;
        emit MaintenanceFundDeposit(amount);
    }
    
    /**
     * @dev Withdraw from maintenance fund (property manager only)
     */
    function withdrawFromMaintenanceFund(uint256 amount, string memory reason) 
        external 
        onlyPropertyManager 
    {
        require(amount <= maintenanceFundBalance, "Insufficient funds");
        maintenanceFundBalance -= amount;
        payable(propertyManager).transfer(amount);
        emit MaintenanceFundWithdrawal(amount, reason);
    }
    
    /**
     * @dev Update property value after appraisal
     */
    function updatePropertyValue(uint256 newValue) external onlyOwner {
        uint256 oldValue = propertyValue;
        propertyValue = newValue;
        emit PropertyRevalued(oldValue, newValue);
    }
    
    /**
     * @dev Calculate token price based on current property value
     */
    function getTokenPrice() public view returns (uint256) {
        return propertyValue / (totalSupply() / 10**decimals());
    }
    
    /**
     * @dev Modifier to restrict certain functions to property manager
     */
    modifier onlyPropertyManager() {
        require(msg.sender == propertyManager, "Caller is not property manager");
        _;
    }
}
