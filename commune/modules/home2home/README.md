 # start of file
# Home2Home: Revolutionizing Rent-to-Own through Real Estate Tokenization

```
 _    _                      ___  _    _                      
| |  | |                    |__ \| |  | |                     
| |__| | ___  _ __ ___   ___   ) | |__| | ___  _ __ ___   ___ 
|  __  |/ _ \| '_ ` _ \ / _ \ / /|  __  |/ _ \| '_ ` _ \ / _ \
| |  | | (_) | | | | | |  __// /_| |  | | (_) | | | | | |  __/
|_|  |_|\___/|_| |_| |_|\___|____|_|  |_|\___/|_| |_| |_|\___|
                                                              
         UNLOCKING OWNERSHIP, ONE RENT PAYMENT AT A TIME
```

## Overview

Home2Home is a decentralized platform that transforms traditional rental agreements into pathways to ownership through blockchain-based tokenization. This repository contains a full-stack implementation of the Home2Home concept, including smart contracts, a Next.js frontend, and a Docker-based local development environment.

## Features

- **Tokenized Property Ownership**: Each property is represented as tokens on the blockchain
- **Equity Accumulation**: Renters build equity with each payment
- **Transparent Tracking**: All ownership changes are recorded on the blockchain
- **Smart Contract Automation**: Payments are automatically split between rent, equity, and maintenance
- **User-Friendly Interface**: Intuitive web application for browsing properties and managing agreements

## Tech Stack

- **Frontend**: Next.js, Chakra UI, ethers.js
- **Blockchain**: Solidity, Hardhat, Ethereum
- **Development Environment**: Docker, Docker Compose
- **Local Blockchain**: Ganache

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Node.js](https://nodejs.org/) (v16+)
- [npm](https://www.npmjs.com/) (v7+)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/home2home.git
   cd home2home
   ```

2. Set up the application:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Start the application:
   ```bash
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

4. The application will be available at:
   - Frontend: http://localhost:3000
   - Local Ethereum Network: http://localhost:8545

### Smart Contract Deployment

Smart contracts are automatically deployed when the Docker containers start up. If you need to manually deploy them:

```bash
chmod +x scripts/deploy-contracts.sh
./scripts/deploy-contracts.sh
```

## Project Structure

```
home2home/
├── blockchain/           # Blockchain-related code
│   ├── scripts/          # Deployment scripts
│   └── test/             # Contract tests
├── contracts/            # Solidity smart contracts
├── frontend/             # Next.js application
│   ├── components/       # React components
│   ├── hooks/            # Custom React hooks
│   ├── pages/            # Next.js pages
│   ├── public/           # Static assets
│   └── styles/           # CSS styles
└── scripts/              # Utility scripts
```

## Smart Contracts

The platform includes three main smart contracts:

1. **PropertyToken**: An ERC-20 token representing ownership of a specific property
2. **RentToOwnAgreement**: Manages the relationship between tenant and property
3. **Home2HomeRegistry**: Central registry for all properties and agreements

## Development

### Running Tests

```bash
cd blockchain
npm test
```

### Modifying Smart Contracts

If you modify the smart contracts, you'll need to redeploy them:

```bash
cd blockchain
npx hardhat compile
npx hardhat run scripts/deploy.js --network localhost
```

### Frontend Development

```bash
cd frontend
npm run dev
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenZeppelin for secure smart contract libraries
- Ethereum community for blockchain infrastructure
- Chakra UI for component library
- Next.js team for the React framework

---

Home2Home - Unlocking ownership, one rent payment at a time.
