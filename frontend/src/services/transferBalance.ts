import { ApiPromise, WsProvider } from '@polkadot/api';

export async function transferBalance(sender: string, recipient: string, amount: number): Promise<string> {
    const provider = new WsProvider('wss://commune.api.onfinality.io/public-ws');
    const api = await ApiPromise.create({ provider });

    const transfer = api.tx.balances.transfer(recipient, amount);
    const hash = await transfer.signAndSend(sender);

    return hash.toString();
}
