import React, { useState } from "react";
import Image from "next/image";
import { ApiPromise, WsProvider } from "@polkadot/api";
import { web3Enable, web3Accounts } from "@polkadot/extension-dapp";
import Modal from "antd/es/modal/Modal";

export default function PolkadotWalletButton() {

    const [extensionAvailable, setExtensionAvailable] = useState<boolean>(false);

    const connectWallet = async () => {
        if (typeof window !== 'undefined') {
            try {
                await web3Enable('Commune AI');
                const accounts = await web3Accounts();
                const provider = new WsProvider('wss://rpc.polkadot.io');
                const polkadotAPI = await ApiPromise.create({ provider });
                const address = accounts[0].address;
                await polkadotAPI.query.system.account(address);
            } catch (error) {
                console.error('Error', error);
            }
        } else {
            console.error('Cannot connect wallet');
        }
    };

    return (
        <>
            <Modal open={extensionAvailable} onCancel={() => setExtensionAvailable(false)} footer={null} width={500}>
                <div className="p-8 flex flex-col gap-4">
                    <p>Please install the Polkadot{".js"} extension to continue.</p>
                    <a href="https://polkadot.js.org/extension/" target="_blank" rel="noopener noreferrer"
                        className="rounded-md px-4 py-2 bg-red-500 text-white hover:text-white hover:bg-red-400 transition-all duration-300">
                        Get Polkadot{".js"} Extension
                    </a>
                </div>
            </Modal>
            <div className=' mb-[20px] bg-blue-700 rounded-lg shadow-lg hover:shadow-2xl text-center hover:bg-blue-600 duration-200 
                text-white font-sans font-semibold px-2 py-2 w-full flex justify-center items-center cursor-pointer' onClick={() => connectWallet()}>
                <Image style={{ width: "auto", height: "2.7rem", marginRight: "0.25rem" }} src="/svg/polkadot.svg" alt="My Site Logo" width={42} height={42} />
                <span>Connect Wallet</span>
            </div>
        </>
    );
}
