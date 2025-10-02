# ModuFi “Tempo” — Temporary ERC-20 for Miner/Validator Incentives

**One-line:** Validators post liquidity; miners race to ping; a randomized epoch ends; the validator pot is **liquidated** to miners holding **temporary** Tempo units earned that epoch; supply resets to zero; repeat.

## Why this works
- **Miners**: To earn rewards, a miner must **ping during the epoch**. The earlier they ping, the larger the emission they get that epoch. Only one ping per address per epoch.
- **Validators**: To secure and accelerate network liveness, validators **add liquidity** in a base token (e.g., USDC) to the current epoch’s pot. That pot is paid out to miners at epoch end.
- **Temporary token**: “Tempo” balances exist **only for the active epoch**. At settlement, holders redeem claim on the pot; the **entire Tempo supply is burned** before the next epoch begins.

## Epoch timing
Each epoch has a **non-deterministic duration** sampled from `block.prevrandao` and bounded by `[minEpoch, maxEpoch]`. You configure `tempoTarget` and `tempoJitterBps` to center and widen the distribution.

## Emissions
- Miners call `ping()` once per epoch.
- Mint = `baseEmission + (timeWeight * timeLeft / epochLength)`.
- Earlier pings earn more.

## Liquidity / liquidation
- Validators call `provideLiquidity(amount)`.
- At epoch end, anyone calls `settleEpoch()`:
  1. Finalizes epoch pot.
  2. Snapshots total Tempo supply.
  3. Holders call `redeem(epochId)` to burn Tempo & withdraw pro-rata base tokens.
  4. Starts new epoch.

## Design choices
- **Transfers disabled** → no last-block MEV games, minimal storage.
- **Pull-based redeem** → no loops.
- **One ping per address per epoch**.
- **Randomized epochs** for tempo unpredictability.

## Usage
```bash
forge build
forge test
```

Deploy example:
```bash
cast send <deployer> --create contracts/TempoToken.sol \
  "constructor(address,uint40,uint40,uint40,uint16,uint128,uint128)" \
  <BASE_ASSET> 60 300 180 2000 1000000000000000000 1000000000000000000
```

- `minEpoch=60s, maxEpoch=300s, tempoTarget=180s, tempoJitterBps=2000 (±20%)`.
- `baseEmission=1e18, timeWeight=1e18`.

Workflow:
1. Validators deposit with `provideLiquidity`.
2. Miners ping with `ping()`.
3. Anyone settles with `settleEpoch()`.
4. Miners redeem with `redeem(epochId)`.
5. Repeat.
