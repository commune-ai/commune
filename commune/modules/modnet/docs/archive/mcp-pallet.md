# Model Context Protocol (MCP) Pallet Specification

## Overview

The Model Context Protocol (MCP) pallet provides a comprehensive framework for managing and executing AI tools, prompts, and resources on the blockchain. This pallet enables decentralized AI services by facilitating communication between AI clients and servers through the standardized MCP protocol.

MCP is designed to enable AI assistants to communicate with other services, allowing for extensible AI capabilities through tools, prompts, and resources. The pallet integrates the official Rust MCP SDK (RMCP) to provide on-chain AI service orchestration.

## Architecture

### Protocol Overview

The Model Context Protocol defines a standard for AI assistants to interact with external services. Key concepts include:

- **Tools**: Executable functions that AI assistants can call
- **Prompts**: Reusable prompt templates for AI interactions
- **Resources**: Data sources and content that tools can access
- **Sampling**: Ability for servers to request LLM capabilities from clients

### Core Components

1. **MCP Server Registry**: On-chain registry of available MCP servers
2. **Tool Management**: Registration and execution of AI tools
3. **Prompt Templates**: Storage and retrieval of reusable prompts
4. **Resource Handling**: IPFS-backed resource storage and access
5. **Access Control**: Permission-based access to servers and capabilities
6. **Transport Layer**: Support for multiple transport mechanisms (stdio, HTTP, SSE)

### Protocol Flow

```
Client Request → Pallet Validation → Server Lookup → Tool Execution → Response
```

## Features

### Server Management
- **Server Registration**: Register MCP servers with capabilities and metadata
- **Server Discovery**: List and query available servers
- **Health Monitoring**: Track server status and availability
- **Version Management**: Support multiple protocol versions (2024-11-05)

### Tool System
- **Tool Registration**: Register tools with schemas and descriptions
- **Parameter Validation**: JSON Schema-based parameter validation
- **Execution Framework**: Secure tool execution with result handling
- **Tool Discovery**: List available tools by server or capability

### Prompt Management
- **Template Storage**: Store reusable prompt templates
- **Parameter Substitution**: Dynamic prompt generation with parameters
- **Categorization**: Organize prompts by category and use case
- **Version Control**: Track prompt template versions

### Resource Handling
- **IPFS Integration**: Store large resources off-chain via IPFS
- **Content Addressing**: Use CIDs for resource identification
- **Access Control**: Permission-based resource access
- **Metadata Storage**: On-chain resource metadata and annotations

### Advanced Features
- **Sampling Support**: Enable servers to request LLM capabilities from clients
- **Authentication**: OAuth and custom authentication mechanisms
- **Transport Flexibility**: Support stdio, HTTP, SSE, and WebSocket transports
- **Event System**: Comprehensive logging and monitoring

## Storage Items

### Simplified Storage Architecture

Following the same pattern as the module registry, the MCP pallet uses minimal on-chain storage with IPFS as the primary metadata store:

```rust
/// Maps creator's public key to IPFS CID containing MCP server metadata
/// Key: Public key in various formats (Ed25519, Ethereum, Solana) - flexible Vec<u8>
/// Value: IPFS CID pointing to complete MCP server metadata
McpServers: StorageMap<_, Blake2_128Concat, Vec<u8>, Vec<u8>, OptionQuery>

/// Maps account to their registered MCP server keys (for discovery)
McpServersByCreator: StorageMap<_, Blake2_128Concat, AccountId, BoundedVec<Vec<u8>, MaxServersPerCreator>, ValueQuery>
```

### IPFS Metadata Structure

All substantive MCP data is stored in IPFS, referenced by CID:

```json
{
  "server_info": {
    "name": "My MCP Server",
    "description": "AI tools for code analysis",
    "protocol_version": "2024-11-05",
    "capabilities": {
      "tools": true,
      "prompts": true,
      "resources": true,
      "sampling": false
    },
    "transport_config": {
      "type": "stdio",
      "config": {}
    }
  },
  "tools": [
    {
      "name": "analyze_code",
      "description": "Analyze code for issues",
      "input_schema": { "type": "object", "properties": {...} }
    }
  ],
  "prompts": [
    {
      "name": "code_review",
      "description": "Code review template",
      "template": "Review this code: {{code}}",
      "parameters": { "code": { "type": "string" } },
      "category": "development"
    }
  ],
  "resources": [
    {
      "uri": "ipfs://QmHash...",
      "name": "Model Weights",
      "description": "Pre-trained model weights",
      "mime_type": "application/octet-stream"
    }
  ],
  "authentication": {
    "type": "oauth2",
    "config": {...}
  },
  "mining_requirements": {
    "required": false,
    "miners": [],
    "validation_rules": {}
  }
}
```

## Data Structures

### Simplified On-Chain Types

```rust
/// Type alias for public keys (flexible format support)
type PublicKey = Vec<u8>;

/// Type alias for IPFS CIDs
type IpfsCid = Vec<u8>;

/// Configuration trait for the MCP pallet
pub trait Config: frame_system::Config {
    type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

    /// Maximum number of MCP servers per creator
    type MaxServersPerCreator: Get<u32>;

    /// Maximum length of public keys
    type MaxKeyLength: Get<u32>;

    /// Maximum length of IPFS CIDs
    type MaxCidLength: Get<u32>;
}
```

### IPFS Metadata Validation

```rust
/// Validation functions for IPFS metadata (off-chain)
pub mod validation {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct McpServerMetadata {
        pub server_info: ServerInfo,
        pub tools: Vec<ToolDefinition>,
        pub prompts: Vec<PromptTemplate>,
        pub resources: Vec<ResourceReference>,
        pub authentication: Option<AuthConfig>,
        pub mining_requirements: Option<MiningConfig>,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct ServerInfo {
        pub name: String,
        pub description: Option<String>,
        pub protocol_version: String,
        pub capabilities: ServerCapabilities,
        pub transport_config: TransportConfig,
    }

    // Additional validation types...
}
```

## Extrinsics

### Simple CRUD Operations

Following the module registry pattern, the MCP pallet provides simple key-value operations:

```rust
/// Register a new MCP server
#[pallet::call_index(0)]
pub fn register_mcp_server(
    origin: OriginFor<T>,
    key: Vec<u8>,        // Creator's public key (Ed25519, Ethereum, Solana, etc.)
    cid: Vec<u8>,        // IPFS CID pointing to MCP server metadata
) -> DispatchResult

/// Update MCP server metadata
#[pallet::call_index(1)]
pub fn update_mcp_server(
    origin: OriginFor<T>,
    key: Vec<u8>,        // Creator's public key
    cid: Vec<u8>,        // New IPFS CID with updated metadata
) -> DispatchResult

/// Remove MCP server
#[pallet::call_index(2)]
pub fn remove_mcp_server(
    origin: OriginFor<T>,
    key: Vec<u8>,        // Creator's public key
) -> DispatchResult

/// Get MCP server metadata CID
#[pallet::call_index(3)]
pub fn get_mcp_server(
    origin: OriginFor<T>,
    key: Vec<u8>,        // Creator's public key
) -> DispatchResult
```

### Query Functions

```rust
/// Get all MCP servers for a creator
pub fn get_servers_by_creator(creator: &T::AccountId) -> Vec<Vec<u8>>

/// Check if MCP server exists
pub fn mcp_server_exists(key: &[u8]) -> bool

/// Get MCP server CID
pub fn get_mcp_server_cid(key: &[u8]) -> Option<Vec<u8>>
```

## Events

```rust
#[pallet::event]
#[pallet::generate_deposit(pub(super) fn deposit_event)]
pub enum Event<T: Config> {
    /// MCP server registered
    McpServerRegistered {
        key: Vec<u8>,
        cid: Vec<u8>,
        owner: T::AccountId,
    },

    /// MCP server updated
    McpServerUpdated {
        key: Vec<u8>,
        old_cid: Vec<u8>,
        new_cid: Vec<u8>,
        owner: T::AccountId,
    },

    /// MCP server removed
    McpServerRemoved {
        key: Vec<u8>,
        cid: Vec<u8>,
        owner: T::AccountId,
    },
}
```

## Errors

```rust
#[pallet::error]
pub enum Error<T> {
    /// MCP server not found
    McpServerNotFound,
    /// MCP server already exists for this key
    McpServerAlreadyExists,
    /// Not the server owner
    NotServerOwner,
    /// Invalid public key format
    InvalidPublicKey,
    /// Invalid IPFS CID format
    InvalidIpfsCid,
    /// Public key too long
    PublicKeyTooLong,
    /// IPFS CID too long
    IpfsCidTooLong,
    /// Maximum servers per creator exceeded
    MaxServersExceeded,
    /// Access denied
    AccessDenied,
}
```

## Dependencies

### Core Dependencies
- **Substrate Framework**: Pallet development framework
- **RMCP (Rust MCP SDK)**: Official Rust implementation of MCP protocol
  - Location: `$HOME/repos/mcp/rust-sdk`
  - Crates: `rmcp`, `rmcp-macros`
- **IPFS Integration**: Off-chain resource storage via `commune-ipfs` submodule
- **JSON Schema**: Parameter validation and tool schema support

### Transport Support
- **Stdio**: Standard input/output transport
- **HTTP**: RESTful API transport
- **SSE**: Server-Sent Events for streaming
- **WebSocket**: Real-time bidirectional communication

### Authentication
- **OAuth 2.0**: Standard OAuth flow support
- **Custom Auth**: Extensible authentication mechanisms

## Integration Points

### IPFS Integration
- Resource storage and retrieval via IPFS CIDs
- Integration with `commune-ipfs` submodule
- Content-addressable resource references

### Runtime Integration
```rust
// In runtime/src/lib.rs
impl pallet_mcp::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type WeightInfo = pallet_mcp::weights::WeightInfo<Runtime>;
    type MaxServersPerOwner = ConstU32<10>;
    type MaxToolsPerServer = ConstU32<50>;
    type MaxResourcesPerServer = ConstU32<100>;
    // ... other config types
}
```

## Usage Examples

### Registering an MCP Server
```rust
// Register a counter server
let server_capabilities = ServerCapabilities::builder()
    .enable_tools()
    .enable_prompts()
    .enable_resources()
    .build();

let transport_config = TransportConfig::Stdio;

McpPallet::register_server(
    RuntimeOrigin::signed(account_id),
    b"counter-server".to_vec(),
    Some(b"A simple counter server".to_vec()),
    server_capabilities,
    transport_config,
)?;
```

### Registering and Calling Tools
```rust
// Register a tool
McpPallet::register_tool(
    RuntimeOrigin::signed(account_id),
    server_id,
    b"increment".to_vec(),
    b"Increment the counter by 1".to_vec(),
    None, // No input schema required
)?;

// Call the tool
McpPallet::call_tool(
    RuntimeOrigin::signed(caller_id),
    tool_id,
    None, // No arguments
)?;
```

### Creating Prompt Templates
```rust
// Register a prompt template
McpPallet::register_prompt(
    RuntimeOrigin::signed(account_id),
    b"code-review".to_vec(),
    Some(b"Code review prompt template".to_vec()),
    b"Please review this code: {{code}}\n\nFocus on: {{focus_areas}}".to_vec(),
    Some(b'{"type":"object","properties":{"code":{"type":"string"},"focus_areas":{"type":"string"}}}'.to_vec()),
    Some(b"development".to_vec()),
)?;
```

## Testing Strategy

### Unit Tests
- Storage operations (CRUD for servers, tools, prompts)
- Extrinsic validation and execution
- Error handling and edge cases
- Weight calculations

### Integration Tests
- End-to-end MCP protocol flows
- IPFS resource storage and retrieval
- Multi-server interactions
- Authentication and authorization

### Benchmarking
- Tool execution performance
- Storage operation costs
- Network communication overhead

## Security Considerations

### Access Control
- Server ownership verification
- Tool execution permissions
- Resource access controls

### Input Validation
- JSON schema validation for tool parameters
- Bounded vector limits for storage items
- Transport configuration validation

### Resource Limits
- Maximum servers per owner
- Maximum tools per server
- Maximum resources per server
- Execution timeout limits

## Future Enhancements

### Protocol Extensions
- Support for newer MCP protocol versions
- Custom capability extensions
- Advanced sampling mechanisms

### Performance Optimizations
- Tool result caching
- Batch operations
- Async execution queues

### Integration Improvements
- WebAssembly tool execution
- Cross-chain MCP server discovery
- Decentralized server reputation system

## Example Requests and Responses

Detailed examples can be found in [mcp-calls.md](mcp-calls.md), including:
- Server registration flows
- Tool discovery and execution
- Prompt template usage
- Resource access patterns
- Error handling scenarios
