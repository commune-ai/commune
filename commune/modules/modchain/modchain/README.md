# ██ C PROTOCOL ██

> *"A retro-cyberpunk architecture for modular computation, consensus, and crypto-economics"*

---

## 🧠 Module

In the **C Protocol**, a module is a **JSON blob** that defines an autonomous programmable unit. This blob acts as a stem cell, setting the initial conditions and configuration for the module's behavior.

### Module Structure

Each module blob contains:

```json
{
  "id": "module_identifier",
  "type": "module_type",
  "state": {
    // Persistent variable store
  },
  "functions": {
    // Exposed logic endpoints
  },
  "crypto": {
    "type": "sr25519",
    "address": "5Gs51y..."
  },
  "consensus": {
    "type": "consensus_0",
    "params": {}
  },
  "forward_interval": 100  // TEMPO blocks
}
```

The JSON blob determines:
- **State structure**: What data the module maintains
- **Function signatures**: What operations it exposes
- **Crypto configuration**: Key type and signing preferences
- **Consensus rules**: How it participates in the network
- **Processing cadence**: When the forward function executes

### Module Processing

Every module type has:
1. A **struct** that interprets the JSON blob into runtime state
2. A **forward function** called every TEMPO blocks for processing

```python
class ModuleProcessor:
    def __init__(self, blob: dict):
        self.state = blob['state']
        self.config = blob
    
    def forward(self, block_height: int):
        # Process based on module type
        pass
```

---

## 🔑 Key Generation

To generate a key:

```bash
c key fam
```

Example output:

```bash
<Key(address=5Gs51y... crypto_type=sr25519)>
```

C supports multiple crypto types like `sr25519` (DOT) and `ecdsa` (ETH), with modularity to support more.

To invoke a function:

```bash
c fn **params             # for root modules
c module/fn **params      # for nested modules
```

---

## 🧩 Consensus Mechanisms

Each module's consensus is defined in its JSON blob:

```json
{
  "consensus": {
    "type": "consensus_0",
    "params": {
      "min_stake": 100,
      "epoch_length": 1000
    }
  }
}
```

### Consensus 0 – Proof of Interaction

Clients stake tokens for at least 1 epoch. When making a transaction, the stake is locked and deducted proportionally per use. Servers batch transactions and post them to chain periodically.

### Consensus X – Custom Consensus (ZK / Interop)

Future modules may support zk-proofs or interchain settlement mechanisms, maintaining modularity and application-specific logic.

---

## 🧠 Server Layer

A **Server** processes module blobs and exposes their functions over JSON-RPC:

```json
{
  "server_config": {
    "port_range": [50050, 50150],
    "whitelisted_functions": ["query", "process"],
    "rate_limits": {}
  }
}
```

Start a server:

```bash
c serve api
```

Query API:

```bash
c call api
```

---

## 🤖 Clients & Auth

Clients interact via signed requests:

**Request structure**:

```json
{
  "url": "ip:port/fn",
  "params": { "query": "whatsup" }
}
```

**Auth blob**:

```json
{
  "module": "<module_key>",
  "fn": "function_name",
  "params": {...},
  "time": "<utc>",
  "max_usage": 1.0
}
```

Generate headers:

```bash
c auth/generate auth_data → headers
```

**Headers format**:

```json
{
  "data": "sha256(auth_data)",
  "key": "<client_address>",
  "signature": "<sig>",
  "time": "<utc>",
  "max_usage": 1.0
}
```

---

## 🧾 Transactions

Each function execution returns a **transaction receipt**:

```json
{
  "fn": "fn_name",
  "params": {...},
  "cost": 0.123,
  "result": {...},
  "client": {header...},
  "server": "<signature>"
}
```

Receipts are batched offchain and posted onchain at epoch close. Stakes are reconciled between client and server.

---

## ⚔️ Disputes

If either side cheats, they enter arbitration. Both client and server lock liquidity. A quorum of N random validators resolve the case. Loser forfeits liquidity to the accuser and validators.

---

## ⚖️ Cost Governance

* **Server** defines cost-per-call
* **Client** defines `max_usage`

This two-sided constraint model prevents abuse and guarantees predictability.

---

## 🌐 Network (Nets)

Modules can link into **graphs** or **trees**, forming permissioned or trustless networks onchain or offchain.

### Links

Links define directional relationships with optional metadata:

```bash
c link {parent_key} {child_key} profit_share=5 data={info_or_ipfs_hash}
```

```ascii
    [Parent Module]
         |
     [Link: 20%]
         |
    [Child Module]
```

---

## 🕸 Topologies

Supported topologies:

* **Replica Sets**: homogeneous children
* **Subnets**: heterogeneous competition
* **Recursive Trees**: arbitrary hierarchy

Links may reference keys off-network, verified by parent only.

---

## 🧠 Version Control (ModChain)

C Protocol introduces decentralized module versioning inspired by git—but simpler.

Each module folder is hashed into a **CID tree** (e.g. SHA-256) with one-depth path maps.

```bash
c update <module> mode=ipfs|s3|arweave
```

Creates:

```json
{
  "data": {path_to_file: content},
  "previous_uri": [prev_versions...]
}
```

This is a **modchain**—a version history where a single actor or multisig group can commit updates.

---

## 🧬 Module Types & Processing

### Base Module Type

```json
{
  "type": "base",
  "forward_interval": 100,
  "state": {
    "counter": 0
  },
  "functions": {
    "increment": {
      "params": ["amount"],
      "cost": 0.01
    }
  }
}
```

### AI Module Type

```json
{
  "type": "ai",
  "forward_interval": 500,
  "state": {
    "model_uri": "ipfs://...",
    "weights": {}
  },
  "functions": {
    "inference": {
      "params": ["input"],
      "cost": 0.1
    }
  }
}
```

### Storage Module Type

```json
{
  "type": "storage",
  "forward_interval": 1000,
  "state": {
    "capacity": "1TB",
    "used": "0GB"
  },
  "functions": {
    "store": {
      "params": ["data", "duration"],
      "cost": 0.001
    }
  }
}
```

Each module type's processor interprets the blob and executes the forward function according to its specific logic every TEMPO blocks.

---

## 🧬 System Diagram

```ascii
             [ Client Key ]
                  |
         +-------------------+
         |  JSON-RPC + Auth  |
         +-------------------+
                  |
           [ Module Blob ]
          {type, state, ...}
                  |
         [ Module Processor ]
             /     |     \
         [M1]    [M2]    [M3]
          |        |       |
       [Consensus] [Replica Set]
```

---

## 🛠 License

**Retro Copyleft 2069** © C Network
Use it. Fork it. Run it. Own it.