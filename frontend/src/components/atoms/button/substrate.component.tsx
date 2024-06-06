import React, { useState, useEffect } from 'react';
import { ApiPromise, WsProvider } from '@polkadot/api';

const SubstrateComponent: React.FC = () => {

	const [api, setApi] = useState<ApiPromise | null>(null);
	const [, setChainInfo] = useState('');
	const [, setNodeName] = useState('');

	useEffect(() => {
		const connectToSubstrate = async () => {
			const provider = new WsProvider('wss://rpc.polkadot.io');
			const substrateApi = await ApiPromise.create({ provider });
			setApi(substrateApi);
		};

		connectToSubstrate();
	}, []);

	const getChainInfo = async () => {
		if (api) {
			const chain = await api.rpc.system.chain();
			setChainInfo(chain.toString())
			const nodeName = await api.rpc.system.name();
			setNodeName(nodeName.toString())
			console.log(`Connected to chain ${chain} using ${nodeName}`);
		}
	};

	return (
		<div className='flex flex-col'>
			<button onClick={getChainInfo} className=' bg-blue-700 rounded-lg shadow-lg hover:shadow-2xl text-center hover:bg-blue-600 duration-200 text-white hover:text-white font-sans font-semibold justify-center px-2 py-2 hover:border-blue-300 hover:border-2 hover:border-solid cursor-pointer'>Get Chain Info</button>
		</div>
	);
};

export default SubstrateComponent;
