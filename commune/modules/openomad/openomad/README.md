# OpenOmad - Base Network NFT Platform

## 🚀 Full-Stack NFT Platform on Base Network

### Structure
```
openomad/
├── contracts/     # Smart contracts (Hardhat)
├── frontend/      # Next.js frontend
└── backend/       # Express.js API
```

### 🔧 Setup

#### 1. Smart Contracts
```bash
cd openomad/contracts
npm install
cp ../.env.example .env
# Edit .env with your keys
npx hardhat compile
npx hardhat run scripts/deploy.js --network base
```

#### 2. Frontend
```bash
cd openomad/frontend
npm install
cp ../.env.example .env.local
# Add contract address and WalletConnect ID
npm run dev
```

#### 3. Backend
```bash
cd openomad/backend
npm install
cp ../.env.example .env
npm run dev
```

### 🌐 Networks
- **Base Mainnet**: Chain ID 8453
- **Base Goerli**: Chain ID 84531

### ✨ Features
- ERC721 NFT minting on Base
- RainbowKit wallet connection
- Next.js 14 with App Router
- Express.js backend API
- Tailwind CSS styling

### 🎯 Deployment
1. Deploy contracts to Base
2. Update contract address in frontend/backend
3. Deploy frontend to Vercel
4. Deploy backend to your preferred host

**Built with ❤️ for Base Network**