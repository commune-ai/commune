# ☢️ ModuFi (Modular Finance) — Oracle-agnostic, Ruleset-modular Prediction Markets

**ModuFi** lets anyone spin up prediction markets where **data sources** (oracles) and **game logic** (rulesets) are both pluggable:

- **Adapters** (Chainlink / Pyth / Mock) normalize prices to 1e8 & enforce time windows.
- **Rulesets** (ClosestGuess, RangeBetting) decide winners & payout splits.
- **Markets** escrow ETH stakes, query adapter + ruleset, then pay winners.
- **Factory** creates markets with *adapter, oracleConfig, ruleset, endTime*.
- **Cyberpunk Next.js** dapp: create markets (pick adapter + ruleset), bet, finalize.
- **Docker Compose** profiles: **local** (hardhat+bootstrap), **test**, **main**. Set **EVM_RPC_HTTP** + **CHAIN_ID**.

## Quickstart
```bash
cp .env.example .env
# edit EVM_RPC_HTTP + CHAIN_ID

# Local dev (hardhat + auto-deploy + seed + web)
docker compose --profile local up -d --build
# open http://localhost:3000

# Testnet (e.g., Base Sepolia)
# set EVM_RPC_HTTP=https://base-sepolia...  CHAIN_ID=84532
docker compose --profile test up -d --build

# Mainnet (e.g., Base)
# set EVM_RPC_HTTP=https://base...  CHAIN_ID=8453
docker compose --profile main up -d --build
```

## Repo layout
```
.
├─ docker-compose.yml
├─ .env.example
├─ contracts/         # Hardhat + Solidity (adapters, rulesets, market, factory, mocks)
│  ├─ Dockerfile
│  ├─ hardhat.config.ts
│  ├─ contracts/
│  └─ scripts/
└─ web/               # Next.js cyberpunk UI (wagmi/viem)
   ├─ Dockerfile
   ├─ app/, components/, lib/
   └─ .env.local.example
```

## Security & scale
- Adapters normalize to **1e8**; enforce `publishTime ∈ [endTime, endTime+maxDelay]`.
- Fees (protocol+creator) ≤ 20%; first winner claim triggers fee payout.
- Winner calc is in rulesets; ETH market scans O(N). For big N, provide a batched ruleset or Merkle claims.

Happy hacking. 🦾
