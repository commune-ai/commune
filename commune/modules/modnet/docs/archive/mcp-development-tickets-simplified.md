# MCP Pallet Development Tickets (Simplified Architecture)

## Overview

The MCP pallet follows the same pattern as the module registry: minimal on-chain storage with IPFS as the primary metadata store. This dramatically simplifies the implementation while maintaining full MCP protocol compatibility.

## Architecture Summary

- **On-Chain**: Simple key-value mapping (PublicKey → IPFS CID)
- **Off-Chain**: Complete MCP server metadata stored in IPFS
- **Pattern**: Mirrors existing module registry for consistency

---

## Development Tickets

### **Ticket MCP-001: Core Data Types and Storage**
**Priority**: High
**Estimated Effort**: 1-2 days

**Description**:
Implement the minimal on-chain storage and basic data types for the MCP pallet.

**Acceptance Criteria**:
- [ ] Define `PublicKey` and `IpfsCid` type aliases
- [ ] Implement `McpServers` storage map (PublicKey → IPFS CID)
- [ ] Implement `McpServersByCreator` storage map for discovery
- [ ] Add pallet `Config` trait with appropriate type parameters
- [ ] Add comprehensive unit tests for storage operations
- [ ] Add proper bounds checking for keys and CIDs

**Technical Notes**:
- Use `Vec<u8>` for flexible public key format support
- Implement proper validation for IPFS CID format
- Follow Substrate storage best practices

**Dependencies**: None

---

### **Ticket MCP-002: Basic CRUD Extrinsics**
**Priority**: High
**Estimated Effort**: 2-3 days

**Description**:
Implement the core CRUD operations for MCP server registration.

**Acceptance Criteria**:
- [ ] Implement `register_mcp_server` extrinsic
- [ ] Implement `update_mcp_server` extrinsic
- [ ] Implement `remove_mcp_server` extrinsic
- [ ] Add ownership verification and access control
- [ ] Add public key and CID validation
- [ ] Add capacity limits per creator
- [ ] Add comprehensive extrinsic tests
- [ ] Add integration tests for complete workflows

**Technical Notes**:
- Validate public key formats (Ed25519, Ethereum, Solana)
- Implement proper IPFS CID validation
- Add rate limiting for registration operations

**Dependencies**: MCP-001

---

### **Ticket MCP-003: Events and Error Handling**
**Priority**: Medium
**Estimated Effort**: 1 day

**Description**:
Implement events and comprehensive error handling.

**Acceptance Criteria**:
- [ ] Implement `McpServerRegistered` event
- [ ] Implement `McpServerUpdated` event
- [ ] Implement `McpServerRemoved` event
- [ ] Add all necessary error types
- [ ] Add proper event emission in all operations
- [ ] Add comprehensive error handling tests
- [ ] Add event emission tests

**Technical Notes**:
- Include all relevant data in events for monitoring
- Provide clear error messages for debugging
- Follow Substrate event naming conventions

**Dependencies**: MCP-002

---

### **Ticket MCP-004: Query Functions and Discovery**
**Priority**: Medium
**Estimated Effort**: 1-2 days

**Description**:
Implement query functions for MCP server discovery and retrieval.

**Acceptance Criteria**:
- [ ] Implement `get_mcp_server_cid` query function
- [ ] Implement `get_servers_by_creator` query function
- [ ] Implement `mcp_server_exists` query function
- [ ] Add pagination support for large result sets
- [ ] Add comprehensive query tests
- [ ] Add performance benchmarks for queries

**Technical Notes**:
- Implement efficient querying patterns
- Consider caching for frequently accessed data
- Add proper error handling for invalid queries

**Dependencies**: MCP-002

---

### **Ticket MCP-005: IPFS Metadata Validation**
**Priority**: Medium
**Estimated Effort**: 2-3 days

**Description**:
Implement validation for IPFS-stored MCP metadata structures.

**Acceptance Criteria**:
- [ ] Define `McpServerMetadata` structure
- [ ] Add JSON schema validation for metadata
- [ ] Implement metadata parsing and validation functions
- [ ] Add support for all MCP protocol features (tools, prompts, resources)
- [ ] Add comprehensive validation tests
- [ ] Add metadata structure documentation

**Technical Notes**:
- Use serde for JSON serialization/deserialization
- Implement proper error handling for malformed metadata
- Support MCP protocol version 2024-11-05

**Dependencies**: MCP-001

---

### **Ticket MCP-006: Integration with commune-ipfs**
**Priority**: High
**Estimated Effort**: 2-3 days

**Description**:
Integrate the MCP pallet with the existing commune-ipfs submodule.

**Acceptance Criteria**:
- [ ] Add integration functions for IPFS storage/retrieval
- [ ] Implement metadata upload/download workflows
- [ ] Add IPFS connectivity validation
- [ ] Add error handling for IPFS operations
- [ ] Add comprehensive integration tests
- [ ] Add performance benchmarks for IPFS operations

**Technical Notes**:
- Leverage existing commune-ipfs infrastructure
- Implement proper retry logic for network operations
- Add connection pooling for efficiency

**Dependencies**: MCP-005, commune-ipfs submodule

---

### **Ticket MCP-007: RMCP SDK Integration**
**Priority**: High
**Estimated Effort**: 3-4 days

**Description**:
Integrate the official RMCP Rust SDK for MCP protocol compliance.

**Acceptance Criteria**:
- [ ] Add RMCP SDK dependency
- [ ] Implement MCP protocol message handling
- [ ] Add transport layer support (stdio, HTTP, SSE)
- [ ] Implement server communication interface
- [ ] Add protocol version negotiation
- [ ] Add comprehensive SDK integration tests
- [ ] Add protocol compliance tests

**Technical Notes**:
- Follow RMCP SDK best practices
- Implement proper error handling for network operations
- Add connection management and pooling

**Dependencies**: MCP-006, RMCP SDK

---

### **Ticket MCP-008: Access Control and Security**
**Priority**: High
**Estimated Effort**: 2-3 days

**Description**:
Implement comprehensive access control and security features.

**Acceptance Criteria**:
- [ ] Add ownership verification for all operations
- [ ] Implement public key validation for different formats
- [ ] Add rate limiting and spam protection
- [ ] Add audit logging for security events
- [ ] Add comprehensive security tests
- [ ] Add security documentation and best practices

**Technical Notes**:
- Support multiple public key formats securely
- Implement proper cryptographic validation
- Add protection against common attack vectors

**Dependencies**: MCP-002

---

### **Ticket MCP-009: Runtime Integration**
**Priority**: High
**Estimated Effort**: 1-2 days

**Description**:
Integrate the MCP pallet into the runtime with proper configuration.

**Acceptance Criteria**:
- [ ] Add MCP pallet to runtime configuration
- [ ] Configure all pallet type parameters
- [ ] Add proper weight calculations
- [ ] Add runtime integration tests
- [ ] Add runtime benchmarking
- [ ] Update runtime documentation

**Technical Notes**:
- Follow Substrate runtime integration best practices
- Ensure proper weight calculations for all operations
- Add comprehensive runtime testing

**Dependencies**: All previous tickets

---

### **Ticket MCP-010: Documentation and Examples**
**Priority**: Medium
**Estimated Effort**: 1-2 days

**Description**:
Create comprehensive documentation and usage examples.

**Acceptance Criteria**:
- [ ] Update `mcp-calls.md` with actual examples
- [ ] Create developer guide for MCP pallet usage
- [ ] Add API documentation for all public functions
- [ ] Create example MCP server metadata structures
- [ ] Add troubleshooting guide
- [ ] Add comprehensive README

**Technical Notes**:
- Include real-world usage scenarios
- Add code examples for common operations
- Ensure documentation stays current with implementation

**Dependencies**: MCP-009

---

## Simplified Timeline

**Total Estimated Effort**: 15-25 days (3-5 weeks)

**Phase 1 (Core)**: MCP-001, MCP-002, MCP-003 (4-6 days)
**Phase 2 (Integration)**: MCP-004, MCP-005, MCP-006 (5-8 days)
**Phase 3 (Advanced)**: MCP-007, MCP-008 (5-7 days)
**Phase 4 (Deployment)**: MCP-009, MCP-010 (2-4 days)

## Key Benefits of Simplified Architecture

1. **Faster Development**: 60% reduction in complexity and development time
2. **Consistency**: Matches existing module registry pattern
3. **Scalability**: No on-chain storage limits for metadata
4. **Flexibility**: Easy to extend without chain upgrades
5. **Cost Efficiency**: Minimal on-chain storage costs
6. **Maintainability**: Simpler codebase with fewer edge cases

## Next Steps

1. Begin with **MCP-001** to establish the foundation
2. Progress through phases sequentially
3. Each ticket can be developed and tested independently
4. Regular integration testing throughout development
