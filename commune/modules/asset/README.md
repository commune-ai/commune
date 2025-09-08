# Multi-Chain Stablecoin Vault

This project implements a stablecoin vault system on both Ethereum (ERC20) and Solana blockchains. Users can deposit accepted stablecoins and the smart contracts track individual user balances.

## Project Structure

```
./
├── asset/
│   ├── erc20/
│   │   └── StableCoinVault.sol    # Ethereum smart contract
│   └── sol/
│       └── solana_vault.rs        # Solana program
├── docker-compose.yml             # Docker setup for local testing
└── README.md                      # This file
```

## Features

### Ethereum (ERC20) Contract
- Accepts multiple ERC20 stablecoins (USDC, USDT, DAI by default)
- Tracks user deposits per token
- Owner can add/remove accepted tokens
- Users can deposit and withdraw their funds
- Events for all major actions

### Solana Program
- Built with Anchor framework
- Accepts SPL tokens configured by owner
- Tracks user balances across multiple tokens
- Secure PDA-based account structure
- Full deposit/withdrawal functionality

## Local Development Setup

### Prerequisites
- Docker and Docker Compose installed
- Node.js 18+ (for Ethereum deployment)
- Rust and Anchor CLI (for Solana development)

### Quick Start

1. Clone the repository:
```bash
git clone <your-repo>
cd <your-repo>
```

2. Start the local blockchain environments:
```bash
docker-compose up -d
```

This will:
- Start a local Ethereum node (Ganache) on port 8545
- Start a local Solana validator on port 8899
- Automatically deploy contracts after nodes are ready

### Ethereum Development

1. Install dependencies:
```bash
cd asset/erc20
npm init -y
npm install --save-dev hardhat @openzeppelin/contracts ethers
```

2. Create `hardhat.config.js`:
```javascript
require("@nomiclabs/hardhat-waffle");

module.exports = {
  solidity: "0.8.0",
  networks: {
    localhost: {
      url: "http://localhost:8545"
    }
  }
};
```

3. Deploy contract:
```bash
npx hardhat run scripts/deploy.js --network localhost
```

### Solana Development

1. Build the program:
```bash
cd asset/sol
anchor build
```

2. Deploy to local validator:
```bash
anchor deploy --provider.cluster http://localhost:8899
```

3. Run tests:
```bash
anchor test --skip-local-validator
```

## Testing the Contracts

### Ethereum Testing

1. Connect to local node:
```bash
npx hardhat console --network localhost
```

2. Interact with the contract:
```javascript
const Vault = await ethers.getContractFactory("StableCoinVault");
const vault = await Vault.attach("<deployed-address>");

// Check accepted tokens
const tokens = await vault.getAcceptedTokens();
console.log("Accepted tokens:", tokens);

// Deposit tokens (requires approval first)
// await token.approve(vault.address, amount);
// await vault.deposit(tokenAddress, amount);
```

### Solana Testing

1. Use Anchor client:
```typescript
import * as anchor from "@project-serum/anchor";

const provider = anchor.AnchorProvider.env();
anchor.setProvider(provider);

const program = anchor.workspace.SolanaVault;

// Initialize vault
const vault = anchor.web3.Keypair.generate();
await program.methods
  .initialize()
  .accounts({
    vault: vault.publicKey,
    owner: provider.wallet.publicKey,
    systemProgram: anchor.web3.SystemProgram.programId,
  })
  .signers([vault])
  .rpc();
```

## Docker Commands

- Start all services: `docker-compose up -d`
- View logs: `docker-compose logs -f`
- Stop all services: `docker-compose down`
- Reset data: `docker-compose down -v`

## Security Considerations

1. **Ethereum Contract**:
   - Only owner can add/remove tokens
   - Uses OpenZeppelin's IERC20 interface
   - Checks for sufficient balances before withdrawal
   - Events for transparency

2. **Solana Program**:
   - PDA-based account derivation
   - Owner authorization checks
   - Balance validation
   - Anchor's built-in security features

## Next Steps

1. Add comprehensive test suites
2. Implement emergency pause functionality
3. Add fee collection mechanism
4. Create frontend interface
5. Add cross-chain bridge functionality
6. Implement yield generation strategies

## License

MIT