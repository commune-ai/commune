import { web3Enable, web3Accounts } from "@polkadot/extension-dapp";

export async function enablePolkadotExtension(appName: string) {
  const extensions = await web3Enable(appName);

  if (extensions.length === 0) {
    throw new Error("No Polkadot extension installed");
  }

  const allAccounts = await web3Accounts();

  if (allAccounts.length === 0) {
    throw new Error("No Polkadot accounts available");
  }

  return allAccounts[0];
}
