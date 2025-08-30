 # start of file
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "./PropertyToken.sol";

/**
 * @title RentToOwnAgreement
 * @dev Manages the relationship between tenant and property
 */
contract RentToOwnAgreement {
    PropertyToken public propertyToken;
    address public tenant;
    address public landlord;
    
    uint256 public monthlyPayment;
    uint256 public equityPercentage; // Percentage of payment that goes to equity (in basis points, e.g. 2000 = 20%)
    uint256 public maintenancePercentage; // Percentage for maintenance (in basis points)
    uint256 public startDate;
    uint256 public endDate;
    
    // Payment tracking
    uint256 public lastPaymentDate;
    uint256 public totalEquityAccumulated;
    
    // Events
    event RentPaid(uint256 amount, uint256 equityAmount, uint256 maintenanceAmount);
    event AgreementTerminated(string reason);
    event EquityTokensIssued(uint256 amount);
    
    constructor(
        address _propertyTokenAddress,
        address _tenant,
        uint256 _monthlyPayment,
        uint256 _equityPercentage,
        uint256 _maintenancePercentage,
        uint256 _durationInMonths
    ) {
        propertyToken = PropertyToken(_propertyTokenAddress);
        tenant = _tenant;
        landlord = propertyToken.owner();
        monthlyPayment = _monthlyPayment;
        equityPercentage = _equityPercentage;
        maintenancePercentage = _maintenancePercentage;
        startDate = block.timestamp;
        endDate = startDate + (_durationInMonths * 30 days);
        lastPaymentDate = startDate;
    }
    
    /**
     * @dev Process monthly rent payment
     */
    function payRent() external payable {
        require(msg.sender == tenant, "Only tenant can pay rent");
        require(msg.value == monthlyPayment, "Incorrect payment amount");
        require(block.timestamp <= endDate, "Agreement has expired");
        
        // Calculate portions
        uint256 equityAmount = (monthlyPayment * equityPercentage) / 10000;
        uint256 maintenanceAmount = (monthlyPayment * maintenancePercentage) / 10000;
        uint256 rentAmount = monthlyPayment - equityAmount - maintenanceAmount;
        
        // Transfer rent portion to landlord
        payable(landlord).transfer(rentAmount);
        
        // Add to maintenance fund
        propertyToken.depositToMaintenanceFund{value: maintenanceAmount}(maintenanceAmount);
        
        // Convert equity portion to tokens
        issueEquityTokens(equityAmount);
        
        // Update state
        lastPaymentDate = block.timestamp;
        totalEquityAccumulated += equityAmount;
        
        emit RentPaid(monthlyPayment, equityAmount, maintenanceAmount);
    }
    
    /**
     * @dev Issue equity tokens to tenant based on payment
     */
    function issueEquityTokens(uint256 equityAmount) internal {
        uint256 tokenPrice = propertyToken.getTokenPrice();
        uint256 tokenAmount = (equityAmount * 10**18) / tokenPrice;
        
        // Transfer tokens from landlord to tenant
        propertyToken.transferFrom(landlord, tenant, tokenAmount);
        
        emit EquityTokensIssued(tokenAmount);
    }
    
    /**
     * @dev Get tenant's ownership percentage
     */
    function getTenantOwnershipPercentage() public view returns (uint256) {
        uint256 tenantBalance = propertyToken.balanceOf(tenant);
        uint256 totalSupply = propertyToken.totalSupply();
        return (tenantBalance * 10000) / totalSupply; // Returns basis points (e.g. 2500 = 25%)
    }
    
    /**
     * @dev Terminate the agreement early
     */
    function terminateAgreement(string memory reason) external {
        require(msg.sender == landlord || msg.sender == tenant, "Unauthorized");
        endDate = block.timestamp;
        emit AgreementTerminated(reason);
    }
}
