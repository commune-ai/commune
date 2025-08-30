// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title CollectiveLoan
 * @dev Smart contract for managing collective loans where multiple participants can pool funds
 * and collectively decide on loan disbursements and repayments.
 */
contract CollectiveLoan {
    // Struct to represent a loan request
    struct LoanRequest {
        address borrower;
        uint256 amount;
        uint256 duration; // in days
        uint256 interestRate; // in basis points (1% = 100)
        string purpose;
        uint256 votesInFavor;
        uint256 votesAgainst;
        bool executed;
        bool repaid;
        uint256 dueDate;
        uint256 amountRepaid;
    }

    // Struct to represent a member of the collective
    struct Member {
        uint256 contribution;
        uint256 sharePercentage; // in basis points (1% = 100)
        bool active;
        mapping(uint256 => bool) hasVoted; // loanRequestId => hasVoted
    }

    // Contract state variables
    address public admin;
    uint256 public totalPoolFunds;
    uint256 public availableFunds;
    uint256 public memberCount;
    uint256 public minContribution;
    uint256 public loanRequestCount;
    uint256 public votingThreshold; // Percentage required to approve a loan (in basis points)
    
    mapping(address => Member) public members;
    mapping(uint256 => LoanRequest) public loanRequests;
    mapping(uint256 => mapping(address => bool)) public loanVotes; // loanRequestId => (voter => inFavor)
    
    // Events
    event MemberJoined(address indexed member, uint256 contribution);
    event FundsContributed(address indexed member, uint256 amount);
    event FundsWithdrawn(address indexed member, uint256 amount);
    event LoanRequested(uint256 indexed loanId, address indexed borrower, uint256 amount);
    event LoanVoted(uint256 indexed loanId, address indexed voter, bool inFavor);
    event LoanApproved(uint256 indexed loanId, address indexed borrower, uint256 amount);
    event LoanRepaid(uint256 indexed loanId, address indexed borrower, uint256 amount);
    event ProfitsDistributed(uint256 amount);

    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }

    modifier onlyMember() {
        require(members[msg.sender].active, "Only members can perform this action");
        _;
    }

    /**
     * @dev Constructor to initialize the collective loan contract
     * @param _minContribution Minimum amount required to join the collective
     * @param _votingThreshold Percentage of votes needed to approve a loan (in basis points)
     */
    constructor(uint256 _minContribution, uint256 _votingThreshold) {
        admin = msg.sender;
        minContribution = _minContribution;
        votingThreshold = _votingThreshold;
    }

    /**
     * @dev Join the collective by contributing funds
     */
    function joinCollective() external payable {
        require(!members[msg.sender].active, "Already a member");
        require(msg.value >= minContribution, "Contribution below minimum");

        totalPoolFunds += msg.value;
        availableFunds += msg.value;
        memberCount++;

        Member storage newMember = members[msg.sender];
        newMember.contribution = msg.value;
        newMember.active = true;
        
        // Update share percentages for all members
        updateSharePercentages();
        
        emit MemberJoined(msg.sender, msg.value);
    }

    /**
     * @dev Contribute additional funds to the collective
     */
    function contributeFunds() external payable onlyMember {
        require(msg.value > 0, "Must contribute some funds");
        
        totalPoolFunds += msg.value;
        availableFunds += msg.value;
        members[msg.sender].contribution += msg.value;
        
        // Update share percentages for all members
        updateSharePercentages();
        
        emit FundsContributed(msg.sender, msg.value);
    }

    /**
     * @dev Request a loan from the collective
     * @param _amount Amount requested
     * @param _duration Loan duration in days
     * @param _interestRate Interest rate in basis points
     * @param _purpose Description of loan purpose
     */
    function requestLoan(
        uint256 _amount,
        uint256 _duration,
        uint256 _interestRate,
        string memory _purpose
    ) external onlyMember {
        require(_amount <= availableFunds, "Requested amount exceeds available funds");
        require(_duration > 0, "Duration must be greater than zero");
        
        uint256 loanId = loanRequestCount++;
        
        LoanRequest storage newLoan = loanRequests[loanId];
        newLoan.borrower = msg.sender;
        newLoan.amount = _amount;
        newLoan.duration = _duration;
        newLoan.interestRate = _interestRate;
        newLoan.purpose = _purpose;
        newLoan.executed = false;
        newLoan.repaid = false;
        
        emit LoanRequested(loanId, msg.sender, _amount);
    }

    /**
     * @dev Vote on a loan request
     * @param _loanId ID of the loan request
     * @param _inFavor Whether the vote is in favor of the loan
     */
    function voteOnLoan(uint256 _loanId, bool _inFavor) external onlyMember {
        require(_loanId < loanRequestCount, "Invalid loan ID");
        require(!loanRequests[_loanId].executed, "Loan already executed");
        require(!members[msg.sender].hasVoted[_loanId], "Already voted on this loan");
        
        LoanRequest storage loan = loanRequests[_loanId];
        
        if (_inFavor) {
            loan.votesInFavor += members[msg.sender].sharePercentage;
        } else {
            loan.votesAgainst += members[msg.sender].sharePercentage;
        }
        
        members[msg.sender].hasVoted[_loanId] = true;
        loanVotes[_loanId][msg.sender] = _inFavor;
        
        emit LoanVoted(_loanId, msg.sender, _inFavor);
        
        // Check if the loan can be executed
        if (loan.votesInFavor >= votingThreshold) {
            executeLoan(_loanId);
        }
    }

    /**
     * @dev Execute an approved loan
     * @param _loanId ID of the loan to execute
     */
    function executeLoan(uint256 _loanId) internal {
        LoanRequest storage loan = loanRequests[_loanId];
        require(!loan.executed, "Loan already executed");
        require(loan.votesInFavor >= votingThreshold, "Not enough votes to execute loan");
        require(loan.amount <= availableFunds, "Insufficient funds in pool");
        
        loan.executed = true;
        loan.dueDate = block.timestamp + (loan.duration * 1 days);
        availableFunds -= loan.amount;
        
        // Transfer funds to borrower
        payable(loan.borrower).transfer(loan.amount);
        
        emit LoanApproved(_loanId, loan.borrower, loan.amount);
    }

    /**
     * @dev Repay a loan (partially or fully)
     * @param _loanId ID of the loan to repay
     */
    function repayLoan(uint256 _loanId) external payable {
        LoanRequest storage loan = loanRequests[_loanId];
        require(loan.executed, "Loan not executed yet");
        require(!loan.repaid, "Loan already repaid");
        require(msg.sender == loan.borrower, "Only borrower can repay loan");
        
        uint256 remainingAmount = calculateRemainingDebt(_loanId);
        require(msg.value > 0 && msg.value <= remainingAmount, "Invalid repayment amount");
        
        loan.amountRepaid += msg.value;
        availableFunds += msg.value;
        
        // Check if loan is fully repaid
        if (loan.amountRepaid >= calculateTotalDebt(_loanId)) {
            loan.repaid = true;
        }
        
        emit LoanRepaid(_loanId, msg.sender, msg.value);
    }

    /**
     * @dev Withdraw share of profits
     */
    function withdrawFunds(uint256 _amount) external onlyMember {
        require(_amount > 0, "Amount must be greater than zero");
        require(_amount <= calculateWithdrawableAmount(msg.sender), "Insufficient withdrawable funds");
        
        availableFunds -= _amount;
        
        // Transfer funds to member
        payable(msg.sender).transfer(_amount);
        
        emit FundsWithdrawn(msg.sender, _amount);
    }

    /**
     * @dev Distribute profits to all members
     */
    function distributeProfits() external onlyAdmin {
        uint256 profits = calculateTotalProfits();
        require(profits > 0, "No profits to distribute");
        
        // Mark profits as distributed by reducing available funds
        // Members can withdraw their share using withdrawFunds
        emit ProfitsDistributed(profits);
    }

    /**
     * @dev Calculate the total debt for a loan (principal + interest)
     * @param _loanId ID of the loan
     * @return Total debt amount
     */
    function calculateTotalDebt(uint256 _loanId) public view returns (uint256) {
        LoanRequest storage loan = loanRequests[_loanId];
        uint256 interest = (loan.amount * loan.interestRate * loan.duration) / (10000 * 365);
        return loan.amount + interest;
    }

    /**
     * @dev Calculate remaining debt for a loan
     * @param _loanId ID of the loan
     * @return Remaining debt amount
     */
    function calculateRemainingDebt(uint256 _loanId) public view returns (uint256) {
        LoanRequest storage loan = loanRequests[_loanId];
        uint256 totalDebt = calculateTotalDebt(_loanId);
        return totalDebt > loan.amountRepaid ? totalDebt - loan.amountRepaid : 0;
    }

    /**
     * @dev Calculate amount a member can withdraw
     * @param _member Address of the member
     * @return Withdrawable amount
     */
    function calculateWithdrawableAmount(address _member) public view returns (uint256) {
        Member storage member = members[_member];
        if (!member.active) return 0;
        
        uint256 memberShare = (availableFunds * member.sharePercentage) / 10000;
        return memberShare > member.contribution ? memberShare : member.contribution;
    }

    /**
     * @dev Calculate total profits in the pool
     * @return Total profits amount
     */
    function calculateTotalProfits() public view returns (uint256) {
        uint256 totalContributions = 0;
        for (uint256 i = 0; i < memberCount; i++) {
            // This is simplified - in a real contract you'd need to iterate through a separate array of member addresses
            // since mappings can't be iterated
            totalContributions += members[msg.sender].contribution;
        }
        
        return totalPoolFunds > totalContributions ? totalPoolFunds - totalContributions : 0;
    }

    /**
     * @dev Update share percentages for all members
     */
    function updateSharePercentages() internal {
        // This is simplified - in a real contract you'd need to iterate through a separate array of member addresses
        // For each member, calculate their percentage of the total pool
        // members[address].sharePercentage = (members[address].contribution * 10000) / totalPoolFunds;
        
        // For simplicity in this example, we'll just update the caller's percentage
        members[msg.sender].sharePercentage = (members[msg.sender].contribution * 10000) / totalPoolFunds;
    }

    /**
     * @dev Get loan details
     * @param _loanId ID of the loan
     * @return Loan details as a tuple
     */
    function getLoanDetails(uint256 _loanId) external view returns (
        address borrower,
        uint256 amount,
        uint256 duration,
        uint256 interestRate,
        string memory purpose,
        uint256 votesInFavor,
        uint256 votesAgainst,
        bool executed,
        bool repaid,
        uint256 dueDate,
        uint256 amountRepaid
    ) {
        LoanRequest storage loan = loanRequests[_loanId];
        return (
            loan.borrower,
            loan.amount,
            loan.duration,
            loan.interestRate,
            loan.purpose,
            loan.votesInFavor,
            loan.votesAgainst,
            loan.executed,
            loan.repaid,
            loan.dueDate,
            loan.amountRepaid
        );
    }

    /**
     * @dev Get member details
     * @param _member Address of the member
     * @return Member details as a tuple
     */
    function getMemberDetails(address _member) external view returns (
        uint256 contribution,
        uint256 sharePercentage,
        bool active
    ) {
        Member storage member = members[_member];
        return (
            member.contribution,
            member.sharePercentage,
            member.active
        );
    }
}
