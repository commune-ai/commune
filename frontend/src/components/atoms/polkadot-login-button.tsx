"use client"
import React, { useState } from "react";
import { web3Enable, web3Accounts } from "@polkadot/extension-dapp";
// import { web3Accounts } from "@polkadot/extension-dapp";
import { InjectedAccountWithMeta } from "@polkadot/extension-inject/types";

const PolkadotWallet = () => {

    const [accounts, setAccounts] = useState<InjectedAccountWithMeta[]>([]);
    const [selectedAccount, setSelectedAccount] = useState<InjectedAccountWithMeta | null>(null);
    const [extensionAvailable, setExtensionAvailable] = useState<boolean>(true);

    async function connectWallet() {
        if (typeof window !== 'undefined') {
            try {
                const extensions = await web3Enable("CommuneAI");
                if (extensions.length === 0) {
                    console.error("Install Polkadot wallet extension");
                    setExtensionAvailable(false);
                    return;
                }

                setExtensionAvailable(true);
                const allAccounts = await web3Accounts();
                setAccounts(allAccounts as InjectedAccountWithMeta[]);
            } catch (error) {
                console.error("Error connecting to wallet", error);
            }
        } else {
            console.error('Cannot connect wallet');
        }
    }

    const handleConnectClick = () => {
        connectWallet();
    };

    const handleAccountSelection = (accountIndex: number) => {
        setSelectedAccount(accounts[accountIndex]);
    };

    return (
        <>
            {
                !extensionAvailable && (
                    <div className="bg-red-100 p-8 rounded-md shadow-md flex flex-col gap-4">
                        <p>Please install the Polkadot{".js"} extension to continue.</p>
                        <a href="https://polkadot.js.org/extension/" target="_blank" rel="noopener noreferrer" className="rounded-md px-4 py-2 bg-red-500 text-white ">
                            Get Polkadot{".js"} Extension
                        </a>
                    </div>
                )
            }

            <div className=" flex items-center justify-center h-[20svh]">
                {
                    accounts.length === 0 && (
                        <div className="bg-neutral-100 p-8 rounded-md shadow-md flex flex-col gap-4">
                            <button onClick={handleConnectClick} className="rounded-md px-4 py-2 bg-neutral-800 text-white ">
                                Connect with Polkadot{".js"}
                            </button>
                        </div>
                    )
                }

                {
                    accounts.length > 0 && !selectedAccount ? (
                        <div className="shadow-md p-8 bg-neutral-100">
                            <select onChange={(e) => handleAccountSelection(parseInt(e.target.value, 10))} className="rounded-md p-1">
                                <option value="" key="empty" hidden disabled selected>
                                    Choose your account
                                </option>
                                {
                                    accounts.map((account, index) => (
                                        <option key={index} value={index}>
                                            {account.meta.name} ({account.address})
                                        </option>
                                    ))
                                }
                            </select>
                        </div>
                    ) : null
                }
                {
                    selectedAccount && (
                        <div className="bg-neutral-100 p-8 rounded-md shadow-lg flex flex-col gap-4">
                            <p>Account Name: <span className="text-black">{selectedAccount.meta.name}</span></p>
                            <p>You are signed as <span className="text-black">{selectedAccount.address}</span></p>

                        </div>
                    )
                }
            </div>
        </>
    );
}

export default PolkadotWallet;
