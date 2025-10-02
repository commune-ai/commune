# ☢️ Neon Oracle Markets (Cyberpunk Edition)

Run a full oracle-agnostic prediction market stack in three modes:
- **local**: hardhat node + mock oracle + auto-seeded market
- **test**: your testnet RPC (Base Sepolia / Sepolia)
- **main**: your mainnet RPC (Base / Ethereum)

## Quickstart
```bash
# 1) Set env
cp .env.example .env
# Edit .env → set EVM_RPC_HTTP + CHAIN_ID per profile

# 2) Local dev (hardhat + web)
docker compose --profile local up -d --build
# Open http://localhost:3000

# 3) Testnet
# Set EVM_RPC_HTTP to your testnet endpoint + CHAIN_ID (e.g., 84532 for Base Sepolia)
docker compose --profile test up -d --build

# 4) Mainnet
# Set EVM_RPC_HTTP mainnet + CHAIN_ID (8453 for Base)
docker compose --profile main up -d --build
```

### Profiles & env
- `local`: spins a Hardhat node, deploys adapters/factory/registry, seeds a market, injects addresses into frontend.
- `test` / `main`: **no local node**. Frontend uses `EVM_RPC_HTTP` and `CHAIN_ID`. Contracts assumed already deployed; set addresses in `web/.env.local` or mount via compose env.

## Repo map
```
.
├─ docker-compose.yml         # multi-profile orchestration
├─ .env.example               # root env for compose
├─ contracts/                 # hardhat project (adapters, factory, market, mocks)
│  ├─ Dockerfile
│  ├─ hardhat.config.ts
│  ├─ scripts/00_local_bootstrap.ts
│  └─ (contracts/* from earlier adapter impl are implied)
└─ web/                       # cyberpunk Next.js frontend
   ├─ Dockerfile
   ├─ app/*  components/*  lib/*  tailwind setup
   └─ .env.local.example
```

## Cyberpunk UI
- CRT scanlines, chromatic aberration, animated gradient grid, neon glows
- Zero external font deps; pure CSS effects

## Change networks fast
- Set **`EVM_RPC_HTTP`** and **`CHAIN_ID`** in `.env`
- Frontend and Hardhat read them via compose env

— happy hacking.
