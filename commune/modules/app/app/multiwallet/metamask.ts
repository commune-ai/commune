import { toBytes, evmHashPersonal, evmAddressFromPubKey } from './utils';
import { secp256k1 } from '@noble/curves/secp256k1';


export async function connectMetaMask() {
const eth = (window as any).ethereum;
if (!eth) throw new Error('MetaMask not found');
const [address] = await eth.request({ method: 'eth_requestAccounts' });
const chainId = await eth.request({ method: 'eth_chainId' });
return { kind: 'metamask', address, chainId };
}


export async function signMetaMask(account: any, message: string | Uint8Array) {
const eth = (window as any).ethereum;
const sig = await eth.request({ method: 'personal_sign', params: [message, account.address] });
return { signatureHex: sig };
}


export async function verifyMetaMask(account: any, message: string | Uint8Array, signatureHex: string): Promise<boolean> {
const msgBytes = toBytes(message);
const msgHash = evmHashPersonal(msgBytes);


const sigBytes = Buffer.from(signatureHex.replace(/^0x/, ''), 'hex');
const r = sigBytes.slice(0, 32);
const s = sigBytes.slice(32, 64);
let v = sigBytes[64];
if (v < 27) v += 27;


try {
const pub = secp256k1.recoverPublicKey(msgHash, Buffer.concat([r, s]), v - 27);
const recovered = evmAddressFromPubKey(pub);
return recovered.toLowerCase() === account.address.toLowerCase();
} catch {
return false;
}
}