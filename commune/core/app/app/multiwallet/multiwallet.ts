# Folder: multiwallet

## File: multiwallet/types.ts
```ts
import type { HexString } from '@polkadot/util/types';

export type WalletKind = 'metamask' | 'subwallet' | 'phantom';

export interface ConnectedAccount {
  kind: WalletKind;
  address: string;          // EVM (0x…), SS58, or base58 (Solana)
  publicKeyHex?: string;     // Hex public key when available (e.g., Phantom)
  chainId?: string;          // EVM chain id (0x…)
  source?: string;           // Substrate extension source
}

export interface SignResult {
  signatureHex: HexString;   // 0x-hex signature
  publicKeyHex?: HexString;   // optional
}

export interface WalletAdapter {
  detect(): Promise<boolean> | boolean;
  connect(): Promise<ConnectedAccount>;
  signMessage(account: ConnectedAccount, message: string | Uint8Array): Promise<SignResult>;
  verify(account: ConnectedAccount, message: string | Uint8Array, signatureHex: string): Promise<boolean>;
}
```

## File: multiwallet/utils.ts
```ts
import { hexToU8a, u8aToHex } from '@polkadot/util';
import { keccak_256 as keccak256 } from '@noble/hashes/sha3';
import { secp256k1 } from '@noble/curves/secp256k1';
import bs58 from 'bs58';

const te = new TextEncoder();
const td = new TextDecoder();

export function toBytes(message: string | Uint8Array): Uint8Array {
  return typeof message === 'string' ? te.encode(message) : message;
}
export function encode(message: string): Uint8Array { return te.encode(message); }
export function decode(bytes: Uint8Array): string { return td.decode(bytes); }

export function isHex(x: string): boolean { return /^0x[0-9a-fA-F]+$/.test(x); }
export function fromHex(hex: string): Uint8Array { return hexToU8a(hex); }
export function toHex(u8: Uint8Array): `0x${string}` { return u8aToHex(u8) as `0x${string}`; }

export function isBase58(x: string): boolean { try { bs58.decode(x); return true; } catch { return false; } }

export function evmHashPersonal(message: Uint8Array): Uint8Array {
  const prefix = `\x19Ethereum Signed Message:\n${message.length}`;
  return keccak256(Buffer.concat([Buffer.from(prefix), Buffer.from(message)]));
}

export function evmAddressFromPubKey(pubkey: Uint8Array): string {
  const uncompressed = secp256k1.ProjectivePoint.fromHex(pubkey).toRawBytes(false);
  const hash = keccak256(uncompressed.slice(1));
  const addr = '0x' + Buffer.from(hash.slice(-20)).toString('hex');
  // EIP-55 checksum
  const lower = addr.slice(2).toLowerCase();
  const hash2 = Buffer.from(keccak256(Buffer.from(lower))).toString('hex');
  let out = '0x';
  for (let i = 0; i < lower.length; i++) {
    out += parseInt(hash2[i], 16) >= 8 ? lower[i].toUpperCase() : lower[i];
  }
  return out;
}
```

## File: multiwallet/metamask.ts
```ts
import { hexToU8a } from '@polkadot/util';
import { secp256k1 } from '@noble/curves/secp256k1';
import { ConnectedAccount, SignResult, WalletAdapter } from './types';
import { evmAddressFromPubKey, evmHashPersonal, toBytes } from './utils';

function normalizeEthAddress(addr: string): string { return addr.startsWith('0x') ? addr : '0x' + addr; }

export class MetaMaskAdapter implements WalletAdapter {
  detect(): boolean { return typeof (window as any).ethereum !== 'undefined'; }

  async connect(): Promise<ConnectedAccount> {
    const eth = (window as any).ethereum;
    if (!eth) throw new Error('MetaMask (window.ethereum) not found');
    const [address] = await eth.request({ method: 'eth_requestAccounts' });
    const chainId = await eth.request({ method: 'eth_chainId' });
    return { kind: 'metamask', address: normalizeEthAddress(address), chainId };
  }

  async signMessage(account: ConnectedAccount, message: string | Uint8Array): Promise<SignResult> {
    const eth = (window as any).ethereum;
    if (!eth) throw new Error('MetaMask not available');
    const bytes = toBytes(message);
    const signatureHex: string = await eth.request({ method: 'personal_sign', params: [hexToU8a('0x' + Buffer.from(bytes).toString('hex')), account.address] });
    // Above hexToU8a on a 0x string returns Uint8Array; but personal_sign expects hex string or utf8 string; pass 0x-hex directly
    // Simpler:
    const sig: string = await eth.request({ method: 'personal_sign', params: ['0x' + Buffer.from(bytes).toString('hex'), account.address] });
    return { signatureHex: sig as `0x${string}` };
  }

  async verify(account: ConnectedAccount, message: string | Uint8Array, signatureHex: string): Promise<boolean> {
    const bytes = toBytes(message);
    const sig = signatureHex.startsWith('0x') ? signatureHex.slice(2) : signatureHex;
    const sigBytes = Buffer.from(sig, 'hex');
    if (sigBytes.length !== 65) return false;
    const r = sigBytes.slice(0, 32);
    const s = sigBytes.slice(32, 64);
    let v = sigBytes[64];
    if (v < 27) v += 27;
    const msgHash = evmHashPersonal(bytes);
    try {
      const pub = secp256k1.recoverPublicKey(msgHash, Buffer.concat([r, s]), v - 27);
      const recovered = evmAddressFromPubKey(pub);
      return recovered.toLowerCase() === account.address.toLowerCase();
    } catch {
      return false;
    }
  }
}
```

## File: multiwallet/subwallet.ts
```ts
import { web3Enable, web3Accounts, web3FromSource } from '@polkadot/extension-dapp';
import { signatureVerify } from '@polkadot/util-crypto';
import { ConnectedAccount, SignResult, WalletAdapter } from './types';
import { toBytes } from './utils';

export class SubWalletAdapter implements WalletAdapter {
  async detect(): Promise<boolean> {
    const exts = await web3Enable('MultiWallet');
    return exts.some((e) => e.name?.toLowerCase().includes('subwallet')) || exts.length > 0;
  }

  async connect(): Promise<ConnectedAccount> {
    const extensions = await web3Enable('MultiWallet');
    if (extensions.length === 0) throw new Error('No Polkadot extension found');
    const accounts = await web3Accounts();
    if (accounts.length === 0) throw new Error('No Substrate accounts found');
    const acc = accounts[0];
    return { kind: 'subwallet', address: acc.address, source: acc.meta?.source as string };
  }

  async signMessage(account: ConnectedAccount, message: string | Uint8Array): Promise<SignResult> {
    const accounts = await web3Accounts();
    const acc = accounts.find((a) => a.address === account.address) ?? accounts[0];
    const injector = await web3FromSource(acc.meta?.source as string);
    const signRaw = injector?.signer?.signRaw;
    if (!signRaw) throw new Error('Substrate signer.signRaw unavailable');
    const bytes = toBytes(message);
    const result = await signRaw({ address: acc.address, data: '0x' + Buffer.from(bytes).toString('hex'), type: 'bytes' });
    return { signatureHex: result.signature as `0x${string}` };
  }

  async verify(account: ConnectedAccount, message: string | Uint8Array, signatureHex: string): Promise<boolean> {
    try {
      const { isValid } = signatureVerify(toBytes(message), signatureHex, account.address);
      return isValid;
    } catch { return false; }
  }
}
```

## File: multiwallet/phantom.ts
```ts
import * as ed25519 from '@noble/ed25519';
import bs58 from 'bs58';
import { ConnectedAccount, SignResult, WalletAdapter } from './types';
import { isBase58, isHex, toBytes } from './utils';

export class PhantomAdapter implements WalletAdapter {
  detect(): boolean { return typeof (window as any).phantom?.solana !== 'undefined'; }

  async connect(): Promise<ConnectedAccount> {
    const ph = (window as any).phantom?.solana;
    if (!ph) throw new Error('Phantom (window.phantom.solana) not found');
    const resp = await ph.connect();
    const pubkey = resp.publicKey?.toBytes?.() ?? resp.publicKey?.toBuffer?.();
    if (!pubkey) throw new Error('Phantom: missing public key bytes');
    const publicKeyHex = ('0x' + Buffer.from(pubkey).toString('hex')) as `0x${string}`;
    const address = resp.publicKey.toBase58();
    return { kind: 'phantom', address, publicKeyHex };
  }

  async signMessage(account: ConnectedAccount, message: string | Uint8Array): Promise<SignResult> {
    const ph = (window as any).phantom?.solana;
    if (!ph) throw new Error('Phantom not available');
    const bytes = toBytes(message);
    const { signature, publicKey } = await ph.signMessage(bytes, 'utf8');
    const pubHex = ('0x' + Buffer.from(publicKey.toBytes()).toString('hex')) as `0x${string}`;
    return { signatureHex: ('0x' + Buffer.from(signature).toString('hex')) as `0x${string}`, publicKeyHex: pubHex };
  }

  async verify(account: ConnectedAccount, message: string | Uint8Array, signatureHex: string): Promise<boolean> {
    let pubBytes: Uint8Array | undefined;
    if (account.publicKeyHex && isHex(account.publicKeyHex)) {
      pubBytes = Buffer.from(account.publicKeyHex.slice(2), 'hex');
    } else if (isBase58(account.address)) {
      pubBytes = bs58.decode(account.address);
    }
    if (!pubBytes) return false;
    try {
      return await ed25519.verify(Buffer.from(signatureHex.slice(2), 'hex'), toBytes(message), pubBytes);
    } catch { return false; }
  }
}
```

## File: multiwallet/index.ts
```ts
export * from './types';
export * from './utils';
export { MetaMaskAdapter } from './metamask';
export { SubWalletAdapter } from './subwallet';
export { PhantomAdapter } from './phantom';

import { WalletKind, WalletAdapter, ConnectedAccount, SignResult } from './types';
import { MetaMaskAdapter } from './metamask';
import { SubWalletAdapter } from './subwallet';
import { PhantomAdapter } from './phantom';
import { encode, decode } from './utils';

const adapters: Record<WalletKind, WalletAdapter> = {
  metamask: new MetaMaskAdapter(),
  subwallet: new SubWalletAdapter(),
  phantom: new PhantomAdapter(),
};

export class MultiWallet {
  static async detect(kind: WalletKind): Promise<boolean> {
    const a = adapters[kind];
    const res = a.detect();
    return res instanceof Promise ? await res : res;
  }

  static async connect(kind: WalletKind): Promise<ConnectedAccount> {
    return adapters[kind].connect();
  }

  static async signMessage(kind: WalletKind, account: ConnectedAccount, message: string | Uint8Array): Promise<SignResult> {
    return adapters[kind].signMessage(account, message);
  }

  static async verify(kind: WalletKind, account: ConnectedAccount, message: string | Uint8Array, signatureHex: string): Promise<boolean> {
    return adapters[kind].verify(account, message, signatureHex);
  }

  // pass-through helpers for convenience
  static encode = encode;
  static decode = decode;
}
```

## File: multiwallet/README.md
```md
# Multiwallet (MetaMask / SubWallet / Phantom)

Modular adapters + shared utils. No private keys handled in-app.

## Install

```bash
npm i @polkadot/extension-dapp @polkadot/util @polkadot/util-crypto \
       @noble/curves @noble/hashes bs58
```

## Usage

```ts
import { MultiWallet } from './multiwallet';

const ok = await MultiWallet.detect('metamask');
const acct = await MultiWallet.connect('metamask');
const msg = 'hello';
const { signatureHex } = await MultiWallet.signMessage('metamask', acct, msg);
const isValid = await MultiWallet.verify('metamask', acct, msg, signatureHex);
```

Swap `'metamask'` for `'phantom'` or `'subwallet'` to target those wallets.
