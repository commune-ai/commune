import { ethers } from 'hardhat';
import * as fs from 'fs'; import * as path from 'path';

async function writeEnv(obj: Record<string,string>, file: string) { fs.writeFileSync(file, Object.entries(obj).map(([k,v])=>`${k}=${v}`).join('\n')); }

async function main(){
  const [deployer, a, b] = await ethers.getSigners();
  const now = (await ethers.provider.getBlock('latest'))!.timestamp;

  // Deploy registry + factory
  const Registry = await ethers.getContractFactory('Registry');
  const registry = await Registry.deploy(); await registry.waitForDeployment();

  const Factory = await ethers.getContractFactory('ModuFiFactory');
  const factory = await Factory.deploy(deployer.address, 200, await registry.getAddress()); await factory.waitForDeployment();

  // Adapters
  const Chainlink = await ethers.getContractFactory('ChainlinkAdapter'); const chainlink = await Chainlink.deploy(); await chainlink.waitForDeployment();
  const Pyth = await ethers.getContractFactory('PythAdapter'); const pyth = await Pyth.deploy(); await pyth.waitForDeployment();
  const Mock = await ethers.getContractFactory('MockAdapter'); const mock = await Mock.deploy(); await mock.waitForDeployment();

  // Rulesets
  const Closest = await ethers.getContractFactory('ClosestGuessRuleset'); const closest = await Closest.deploy(); await closest.waitForDeployment();
  const Range5 = await ethers.getContractFactory('RangeBettingRuleset'); const range = await Range5.deploy(500); await range.waitForDeployment(); // ±5%

  // Seed mock oracle
  const oracleId = ethers.id('ETH-USD:LOCAL');
  await (await mock.devSet(oracleId, 3000n * 10n**8n, now + 3600)).wait();
  const cfg = ethers.AbiCoder.default.encode(['bytes32'], [oracleId]);

  // Create market (closest-guess) ends in 1h, 3h window, 2 min players
  const end = now + 3600;
  await (await factory.createETHMarketModular(await mock.getAddress(), cfg, await closest.getAddress(), end, 0, 3*3600, 2)).wait();

  // Bets
  const mktAddr = (await (await ethers.getContractAt('Registry', await registry.getAddress())).getAll())[0];
  const mkt = await ethers.getContractAt('ModuFiMarketETH', mktAddr);
  await (await mkt.connect(a).placeBetETH(3001n * 10n**8n, { value: ethers.parseEther('0.5') })).wait();
  await (await mkt.connect(b).placeBetETH(2999n * 10n**8n, { value: ethers.parseEther('0.5') })).wait();

  // Write web/.env.local for local UI
  const webEnv = path.resolve(__dirname, '..', '..', 'web', '.env.local');
  await writeEnv({
    NEXT_PUBLIC_FACTORY: await factory.getAddress(),
    NEXT_PUBLIC_REGISTRY: await registry.getAddress(),
    NEXT_PUBLIC_ADAPTER_MOCK: await mock.getAddress(),
    NEXT_PUBLIC_ADAPTER_CHAINLINK: await chainlink.getAddress(),
    NEXT_PUBLIC_ADAPTER_PYTH: await pyth.getAddress(),
    NEXT_PUBLIC_RULESET_CLOSEST: await closest.getAddress(),
    NEXT_PUBLIC_RULESET_RANGE5: await range.getAddress(),
    NEXT_PUBLIC_LOCAL_ORACLE_ID: oracleId,
    NEXT_PUBLIC_CHAIN_ID: '31337',
    NEXT_PUBLIC_RPC_HTTP: 'http://localhost:8545'
  }, webEnv);

  console.log('Local ModuFi bootstrap complete. Market:', mktAddr);
}

main().catch((e)=>{ console.error(e); process.exit(1); });
