# Meeting Notes - August 6, 2025
## Architecture Changes and Development Direction

### Date: 2025-08-06
### Attendees: Fam, Huck, Ziggy, Bako
### Meeting Type: Architecture Planning

---

## 🏗️ Major Architectural Changes

### MCP Server Integration Strategy
**Decision**: MCP servers will be integrated as a **submodule of the module system** rather than a standalone pallet.

**Implementation Details**:
- MCP servers will be registered through the existing **module registrar**
- A new `module_type` field may be added to module primitives if differential handling is required
- The current module registrar should be capable of handling MCP server registration without major modifications

### MCP Client Architecture
**Decision**: Move MCP client to an **off-chain worker** implementation.

**Key Components**:
- **RPC Communication Layer**: Handles communication with MCP servers
- **Web2 API Server**: Provides REST/HTTP interface for traditional web developers
- **Chain Query Interface**: Queries on-chain metadata for server discovery and configuration
- **Off-chain Worker**: Manages the entire client lifecycle outside of runtime

---

## 📋 Module Registry Enhancements

### Registry Limits and Stake Management
- **Module Limit**: 128 modules maximum (configurable via on-chain parameters)
- **Stake-based Eviction**: Lowest staked module gets removed when registry is full
- **Parameter Management**: Limit must be adjustable through on-chain governance

### Asset Locking Mechanism
- **Registration Fees**: Modules must lock assets for registration
- **Freeze Mechanism**: Implement asset freezing for locked registration costs
- **Stake Requirements**: Define minimum stake requirements for module registration

### Sub-module Mining System
- **Miner Registration**: Sub-modules act as miners for registered modules
- **Subnet Architecture**: Similar to other Substrate chain subnet implementations
- **Miner Requirements**:
  - Lock-up fees for miner registration
  - Sub-map of keys to CIDs for miner code storage
  - Verification mechanisms for miner authenticity

---

## ⚖️ Arbitration and Slashing System

### Arbitration Process Flow
1. **User Complaint**: User reports issue with module/miner
2. **Arbitration Trigger**: System initiates arbitration process
3. **Module Provider Notification**: Provider informed of the issue
4. **Investigation Phase**: Provider investigates and gathers evidence
5. **Evidence Submission**: Provider must submit verifiable proof
6. **Resolution**:
   - **Proof Provided**: Bad miners slashed, provider protected
   - **No Proof**: Module provider suffers slash from lock fees

### Slashing Implementation
- **Sudo-only Commands**: All slashing operations require sudo privileges
- **Confirmation System**: Multi-step confirmation process
- **Waiting Period**: Time delay before slashing execution (TBD)
- **Verification Requirements**: All evidence must be cryptographically verifiable

---

## 🎯 Action Items and TODO List

### Immediate Tasks
- [ ] **Undo MCP Pallet Setup**: Revert current MCP server pallet implementation
- [ ] **Plan Off-chain Client**: Design architecture for off-chain worker MCP client
- [ ] **Create Test MCP Server**: Build reference implementation with IPFS storage
- [ ] **Registry Limit Implementation**: Add configurable module limit to registry
- [ ] **Asset Locking System**: Implement registration fee locking mechanism

### Medium-term Tasks
- [ ] **Mining System Design**: Define sub-module miner architecture
- [ ] **Arbitration Framework**: Implement sudo-protected slashing commands
- [ ] **Governance Integration**: Ensure parameter configurability through on-chain governance

### Future Considerations
- [ ] **Rewards and Emissions**: Coordinate with consensus team (handled by other team)
- [ ] **Consensus Layer Integration**: Plan registry interaction with consensus mechanisms

---

## 🔄 Development Impact

### Current Work Status
- **MCP-001**: Core data types completed but may need refactoring for new architecture
- **MCP Pallet**: Will be deprecated in favor of module registry integration
- **Module Registry**: Becomes primary focus for MCP server management

### Architecture Shift
- **From**: Standalone MCP pallet with on-chain client
- **To**: Module registry integration with off-chain worker client
- **Benefits**:
  - Unified module management
  - Better separation of concerns
  - Improved scalability for Web2 integration

---

## 📝 Notes and Considerations

### Technical Considerations
- Ensure backward compatibility during transition
- Plan migration strategy for any existing MCP implementations
- Consider performance implications of off-chain worker architecture
- Design robust error handling for off-chain/on-chain communication

### Security Considerations
- Implement proper access controls for sudo-only operations
- Ensure arbitration process cannot be gamed
- Verify all cryptographic proof requirements
- Plan for potential attack vectors in mining system

### Governance Considerations
- All limits and parameters must be governable
- Slashing mechanisms need community oversight
- Consider multi-sig requirements for critical operations

---

## 🔗 Related Documentation
- [Module Registry Documentation](../module-registry.md)
- [MCP Development Tickets](../archive/mcp-development-tickets.md) *(needs revision)*
- [MCP Pallet Specification](../archive/mcp-pallet.md) *(may be deprecated)*

---

**Next Meeting**: TBD
**Follow-up Required**: Architecture design document for off-chain worker client
