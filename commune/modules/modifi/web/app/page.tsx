"use client";
import { useEffect, useMemo, useState } from 'react';
import { useAccount, useConnect, useDisconnect, usePublicClient, useSwitchChain, useWalletClient } from 'wagmi';
import { injected } from 'wagmi/connectors';
import { encodeAbiParameters, parseEther } from 'viem';
import { FactoryABI, MarketABI, RegistryABI } from '@/lib/abi';
import { buildOracleConfig } from '@/lib/oracles';

const ENV = {
  FACTORY: process.env.NEXT_PUBLIC_FACTORY as `0x${string}`,
  REGISTRY: process.env.NEXT_PUBLIC_REGISTRY as `0x${string}`,
  CHAIN_ID: Number(process.env.NEXT_PUBLIC_CHAIN_ID||'0'),
  RPC: process.env.NEXT_PUBLIC_RPC_HTTP || ''
} as const;

export default function Home(){
  const { connectAsync } = useConnect({ connector: injected() });
  const { isConnected, address, chainId } = useAccount();
  const { disconnect } = useDisconnect();
  const publicClient = usePublicClient();
  const { switchChain } = useSwitchChain();
  const { data: wallet } = useWalletClient();

  const [status,setStatus] = useState('');
  const [markets,setMarkets] = useState<`0x${string}`[]>([]);

  // create form
  const [oracle,setOracle] = useState<'mock'|'chainlink'|'pyth'>('mock');
  const [feed,setFeed] = useState('');
  const [pythId,setPythId] = useState('');
  const [ruleset,setRuleset] = useState<'closest'|'range5'>('closest');
  const [customRuleset,setCustomRuleset] = useState('');
  const [hours,setHours] = useState(6); const [minPlayers,setMinPlayers] = useState(2);

  // interact form
  const [market,setMarket] = useState<`0x${string}`|''>('');
  const [guess,setGuess] = useState('300000000000'); const [stake,setStake] = useState('0.05');

  const mustSwitch = useMemo(()=> ENV.CHAIN_ID && chainId && ENV.CHAIN_ID!==chainId,[chainId]);

  useEffect(()=>{ load(); },[publicClient]);

  async function ensureChain(){ if(mustSwitch){ try{ switchChain({ chainId: ENV.CHAIN_ID }); }catch{} } }
  async function connect(){ await connectAsync(); await ensureChain(); }

  async function load(){ if(!publicClient||!ENV.REGISTRY) return; try{ const all = await publicClient.readContract({ address: ENV.REGISTRY, abi: RegistryABI, functionName:'getAll' }) as `0x${string}`[]; setMarkets(all);}catch{} }

  function rulesetAddress(){
    if (customRuleset) return customRuleset as `0x${string}`;
    return (ruleset==='closest' ? (process.env.NEXT_PUBLIC_RULESET_CLOSEST as `0x${string}`) : (process.env.NEXT_PUBLIC_RULESET_RANGE5 as `0x${string}`));
  }

  async function create(){
    if(!wallet||!ENV.FACTORY) return setStatus('Connect wallet'); await ensureChain();
    try{
      const end = Math.floor(Date.now()/1000)+hours*3600;
      const { adapter, cfg } = buildOracleConfig(oracle, { feed, pythId });
      const raddr = rulesetAddress(); if(!raddr) return setStatus('Missing ruleset address');
      const hash = await wallet.writeContract({ address: ENV.FACTORY, abi: FactoryABI, functionName:'createETHMarketModular', args:[adapter, cfg, raddr, BigInt(end), 0, BigInt(3*3600), BigInt(minPlayers)]});
      setStatus('Create tx: '+hash); await publicClient?.waitForTransactionReceipt({ hash }); await load();
    }catch(e:any){ setStatus(e?.shortMessage||e?.message||'create failed'); }
  }

  async function place(){ if(!wallet||!market) return setStatus('Pick market'); await ensureChain(); try{ const hash = await wallet.writeContract({ address: market, abi: MarketABI, functionName:'placeBetETH', args:[BigInt(guess)], value: parseEther(stake) }); setStatus('Bet tx: '+hash);}catch(e:any){ setStatus(e?.shortMessage||e?.message||'bet failed'); } }
  async function finalize(){ if(!wallet||!market) return setStatus('Pick market'); await ensureChain(); try{
      let aux: `0x${string}`='0x'; if(oracle==='pyth'){ const j = prompt('Pyth payloads JSON array?')||'[]'; const arr = JSON.parse(j) as string[]; aux = encodeAbiParameters([{type:'bytes[]'}],[arr]) as `0x${string}`; }
      const hash = await wallet.writeContract({ address: market, abi: MarketABI, functionName:'finalize', args:[aux] }); setStatus('Finalize tx: '+hash);
    }catch(e:any){ setStatus(e?.shortMessage||e?.message||'finalize failed'); } }

  return (
    <div className="grid gap-6">
      <div className="card"><div className="flex items-center justify-between">
        <div>
          <div className="text-sm opacity-80">RPC: {ENV.RPC||'http://localhost:8545'}</div>
          <div className="text-sm opacity-70">Factory: {ENV.FACTORY||'—'} | ChainId: {ENV.CHAIN_ID||'—'}</div>
        </div>
        <div className="flex items-center gap-2">
          {!isConnected ? <button className="btn btn-primary" onClick={connect}>Connect</button> : <>
            {mustSwitch && <button className="btn" onClick={ensureChain}>Switch Chain</button>}
            <button className="btn" onClick={()=>disconnect()}>Disconnect</button>
          </>}
        </div>
      </div></div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-lg font-semibold mb-2">Create Market</h2>
          <label className="text-sm opacity-80">Oracle</label>
          <select className="select" value={oracle} onChange={e=>setOracle(e.target.value as any)}>
            <option value="mock">Mock (local)</option>
            <option value="chainlink">Chainlink</option>
            <option value="pyth">Pyth</option>
          </select>
          {oracle==='chainlink' && <><label className="text-sm opacity-80 mt-3">Feed</label><input className="input" placeholder="0x..." value={feed} onChange={e=>setFeed(e.target.value)} /></>}
          {oracle==='pyth' && <><label className="text-sm opacity-80 mt-3">Pyth Price ID</label><input className="input" placeholder="0x..." value={pythId} onChange={e=>setPythId(e.target.value)} /></>}

          <label className="text-sm opacity-80 mt-3">Ruleset</label>
          <select className="select" value={ruleset} onChange={e=>setRuleset(e.target.value as any)}>
            <option value="closest">Closest Guess</option>
            <option value="range5">Range (±5%)</option>
          </select>
          <label className="text-sm opacity-80 mt-3">Custom Ruleset (optional)</label>
          <input className="input" placeholder="0xYourRuleset" value={customRuleset} onChange={e=>setCustomRuleset(e.target.value)} />

          <div className="grid grid-cols-2 gap-3 mt-3">
            <div><label className="text-sm opacity-80">Hours</label><input className="input" type="number" min={1} value={hours} onChange={e=>setHours(parseInt(e.target.value||'1',10))} /></div>
            <div><label className="text-sm opacity-80">Min Players</label><input className="input" type="number" min={2} value={minPlayers} onChange={e=>setMinPlayers(parseInt(e.target.value||'2',10))} /></div>
          </div>
          <button className="btn btn-primary mt-3" onClick={create}>Create</button>
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold mb-2">Place / Finalize</h2>
          <label className="text-sm opacity-80">Market</label>
          <select className="select" value={market} onChange={e=>setMarket(e.target.value as any)}>
            <option value="">Select…</option>
            {markets.map(m => (<option key={m} value={m}>{m}</option>))}
          </select>
          <label className="text-sm opacity-80 mt-3">Guess (1e8)</label>
          <input className="input" value={guess} onChange={e=>setGuess(e.target.value)} />
          <label className="text-sm opacity-80 mt-3">Stake (ETH)</label>
          <input className="input" value={stake} onChange={e=>setStake(e.target.value)} />
          <div className="flex gap-2 mt-3">
            <button className="btn btn-primary" onClick={place}>Place Bet</button>
            <button className="btn" onClick={finalize}>Finalize</button>
          </div>
        </div>
      </div>

      <div className="card"><div className="text-sm">Status: {status||'—'}</div></div>
    </div>
  );
}
