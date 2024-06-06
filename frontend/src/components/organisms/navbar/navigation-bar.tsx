import { Fragment, useState, useEffect } from 'react'
import classNames from "classnames";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Disclosure, Menu, Transition } from '@headlessui/react'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import { ApiPromise, WsProvider } from '@polkadot/api';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { Modal, Popover } from 'antd';
import { AiFillWallet } from 'react-icons/ai';
import { FaSpinner } from 'react-icons/fa6';
import DiscordIcon from "@/components/atoms/discord-icon";
import GitHubIcon from "@/components/atoms/github-icon";
import TwitterIcon from "@/components/atoms/twitter-icon";
import ThemeToggler from "@/components/templates/theme-toggler";
import { usePolkadot } from "@/context"
import { useColorContext } from '@/context/color-widget-provider';
import { truncateWalletAddress } from '@/utils';
import Button from '@/utils/button';
import ColorPicker from '@/utils/colorWidget';
import classes from './navigation-bar.module.css';
import styles from './navigation-bar.module.css'
import LogoImage from '../../../../public/img/frontpage/comai-logo.png'
import ColorPickImage from '../../../../public/img/icon/colorWidget.jpg'
import CommunityImage from '../../../../public/img/icon/communtiy.png'
import DocsImage from '../../../../public/img/icon/docs.png'
import ModuleImage from '../../../../public/img/icon/module.png'
import SatelliteImage from '../../../../public/img/icon/satellite.png'

const navigation = [
	{ name: 'Modules', href: '/commune-modules', current: false, icon: ModuleImage },
	{ name: 'Telemetry', href: '/telemetry', current: false, icon: SatelliteImage },
	{ name: 'Docs', href: '/docs/introduction', current: false, icon: DocsImage },
	// { name: '📄Whitepaper', href: 'https://ai-secure.github.io/DMLW2022/assets/papers/7.pdf' },
]

const community = [
	{ name: 'Discord', href: 'https://discord.gg/communeai' },
	{ name: 'Twitter', href: 'https://twitter.com/communeaidotorg' },
	{ name: 'Github', href: 'https://github.com/commune-ai' },
]

export default function NavigationBar() {

	const [isShowConnectWithSubstrateModalOpen, setIsShowConnectWithSubstrateModalOpen] = useState(false)
	const router = useRouter();

	//polkadot js login
	const { isInitialized, handleConnect, selectedAccount } = usePolkadot()

	// const [isMetamaskLogedin, setIsMetamaskLogedin] = useState(false)
	// const [address, setAddress] = useState<string | undefined>('')
	const handleConnectWithSubstrateModalCancel = () => {
		setIsShowConnectWithSubstrateModalOpen(false)
	}
	const [api, setApi] = useState<ApiPromise | null>(null);
	const [chainInfo, setChainInfo] = useState('');
	const [nodeName, setNodeName] = useState('');

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
			if (chain) {
				setChainInfo(chain?.toString())
			}
			const nodeName = await api.rpc.system.name();
			if (nodeName) {
				setNodeName(nodeName?.toString())
			}
			console.log(`Connected to chain ${chain} using ${nodeName}`);
		}
	};

	const [currentColor, setCurrentColor] = useState<string>('#000000');

	const handleColorChange = (color: string) => {
		setCurrentColor(color);
	};

	const [openColorWidgetPopOver, setColorWidgetOpen] = useState(false);

	const handleOpenColorWidgetChange = (newOpen: boolean) => {
		setColorWidgetOpen(newOpen);
	};

	const { color, changeColor } = useColorContext();

	return (
		<>
			<div className="min-h-full">
				<Disclosure as="nav" className={`dark:bg-[${color}] border-b-2 border-slate-500 shadow-md py-4`} style={{ backgroundColor: color }}>
					{({ open }) => (
						<>
							<div className="mx-auto px-4 lg:px-8">
								<div className="flex h-16 items-center justify-between">
									<div className="flex items-center">
										<Link className={classes.brand} href="/">
											<Image
												style={{ width: "auto", height: "4rem", marginRight: "-0.25rem" }}
												src={LogoImage}
												alt="Commune Logo"
												width={64}
												height={64}
											/>
										</Link>

										<div className="hidden xl:block">
											<div className="flex">
												{
													navigation.map((item) => (
														<a
															key={item.name}
															href={item.href}
															className={classNames(classes.link, 'flex text-white dark:text-[#32CD32] dark:hover:text-[#6bcd32] p-0 lg:pl-4')}
															aria-current={item.current ? 'page' : undefined}
														>
															<Image src={item.icon} alt='communityimage' width={40} height={30} className='ml-3 mr-1' />
															<span className={`flex items-center justify-center ${styles.fontStyle}`}>
																{item.name}
															</span>
														</a>
													))
												}
											</div>
										</div>
										<Menu as="div" className="flex relative ml-3">

											<Menu.Button className={classNames(classes.link, classes.fontStyle, 'text-white flex items-center justify-center dark:text-[#32CD32] dark:hover:text-[#6bcd32] p-0')} aria-haspopup="true" aria-expanded="false" role="button" >
												<Image src={CommunityImage} alt='communityimage' width={30} height={30} className='ml-3 mr-1' />
												Community
											</Menu.Button>

											<Transition
												as={Fragment}
												enter="transition ease-out duration-100"
												enterFrom="transform opacity-0 scale-95"
												enterTo="transform opacity-100 scale-100"
												leave="transition ease-in duration-75"
												leaveFrom="transform opacity-100 scale-100"
												leaveTo="transform opacity-0 scale-95"
											>
												<Menu.Items className="dark:bg-[#242556] dark:text-[#32CD32] absolute right-0 z-10 mt-8 w-39 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
													<Menu.Item>
														<Link
															className={classes.dropdownLink}
															href="https://discord.gg/communeai"
															target="_blank"
															rel="noopener noreferrer"
														>
															<div style={{ display: "flex", alignItems: "center", }}>
																<span><DiscordIcon /></span>
																<span className="ml-1 mr-2">Discord</span>
															</div>
														</Link>
													</Menu.Item>
													<Menu.Item>
														<Link
															className={classes.dropdownLink}
															href="https://twitter.com/communeaidotorg"
															target="_blank"
															rel="noopener noreferrer"
														>
															<div style={{ display: "flex", alignItems: "center", }}>
																<TwitterIcon />
																<span className="ml-1 mr-2">Twitter</span>
															</div>
														</Link>
													</Menu.Item>
													<Menu.Item>
														<Link
															className={classes.dropdownLink}
															href="https://github.com/commune-ai"
															target="_blank"
															rel="noopener noreferrer"
														>
															<div style={{ display: "flex", alignItems: "center", }}>
																<GitHubIcon />
																<span className="ml-1 mr-2">Github</span>
															</div>
														</Link>
													</Menu.Item>
												</Menu.Items>
											</Transition>

										</Menu>

									</div>

									<div className="hidden md:block">
										<div className="flex items-center relative">

											{
												isInitialized && selectedAccount ? (

													<div className="flex items-center bg-white rounded-full shadow px-4 py-2 ml-2 w-[180px]">
														<button className="flex items-center cursor-pointer">
															<Image
																width={35}
																height={15}
																className="cursor-pointer"
																alt="Tailwind CSS Navbar component"
																src="/img/polkadot.png" />
															<span className="ml-2 font-mono dark:text-black">
																{truncateWalletAddress(selectedAccount.address)}
															</span>
														</button>
													</div>
												)
													:
													<Menu as="div" className="flex">
														<div>
															<Menu.Button style={{ marginLeft: '0.35rem', width: '300px' }} className={classNames(classes.link, classes.fontStyle, 'text-white dark:text-[#32CD32] dark:hover:text-[#6bcd32] p-0 w-full')}>Choose wallet</Menu.Button>
														</div>
														<Transition
															as={Fragment}
															enter="transition ease-out duration-100"
															enterFrom="transform opacity-0 scale-95"
															enterTo="transform opacity-100 scale-100"
															leave="transition ease-in duration-75"
															leaveFrom="transform opacity-100 scale-100"
															leaveTo="transform opacity-0 scale-95"
														>
															<Menu.Items className="dark:bg-[#242556] dark:text-white absolute right-0 z-10 mt-8 w-[15.5rem] origin-top-right rounded-md bg-white py-1 px-5 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none flex items-start justify-center flex-col">
																<Menu.Item>
																	{
																		isInitialized && selectedAccount ? (
																			<div className="flex items-center">

																				<div className="relative flex items-center bg-white rounded-full shadow px-4 py-2">
																					<button className="flex items-center cursor-pointer">
																						<AiFillWallet size={24} className="text-purple dark:text-black" />
																						<span className="ml-2 font-mono dark:text-black">
																							{truncateWalletAddress(selectedAccount.address)}
																						</span>
																					</button>
																				</div>
																			</div>
																		) : (
																			<div className="flex items-center gap-x-2 w-full">
																				{!isInitialized && <FaSpinner className="spinner" />}
																				<Button
																					size="large"
																					variant="primary"
																					className='flex items-center justify-center'
																					onClick={handleConnect}
																					isDisabled={!isInitialized}
																				>
																					<AiFillWallet size={18} />
																					Connect with Polkadot
																				</Button>
																			</div>
																		)
																	}
																</Menu.Item>

																<Menu.Item>
																	<ConnectButton.Custom>
																		{({
																			account,
																			chain,
																			openAccountModal,
																			openChainModal,
																			openConnectModal,
																			authenticationStatus,
																			mounted,
																		}) => {
																			const ready = mounted && authenticationStatus !== 'loading';
																			// setIsMetamaskLogedin(true);
																			// setAddress(account?.displayName)
																			const connected =
																				ready &&
																				account &&
																				chain &&
																				(!authenticationStatus ||
																					authenticationStatus === 'authenticated');
																			return (
																				<div
																					{...(!ready && {
																						'aria-hidden': true,
																						'style': {
																							opacity: 0,
																							pointerEvents: 'none',
																							userSelect: 'none',
																						},
																					})}
																				>
																					{(() => {
																						if (!connected) {
																							return (

																								<Button
																									size="large"
																									variant="primary"
																									className='flex items-center justify-center mt-2 w-full'
																									onClick={openConnectModal}
																								>
																									<AiFillWallet size={18} />
																									Connect with Metamask
																								</Button>
																							);
																						}

																						if (chain.unsupported) {
																							return (
																								<button onClick={openChainModal} type="button" style={{ boxShadow: 'rgb(0 0 0 / 98%) 3px 3px 3px 3px' }}>
																									Wrong network
																								</button>
																							);
																						}

																						return (
																							<div style={{ display: 'flex', gap: 12 }} className='flex items-center flex-col justify-center'>
																								<button
																									onClick={openChainModal}
																									style={{ display: 'flex', alignItems: 'center' }}
																									type="button"
																								>
																									{chain.hasIcon && (
																										<div
																											style={{
																												background: chain.iconBackground,
																												width: 12,
																												height: 12,
																												borderRadius: 999,
																												overflow: 'hidden',
																												marginRight: 4,
																											}}
																										>
																											{chain.iconUrl && (
																												<Image
																													alt={chain.name ?? 'Chain icon'}
																													src={chain.iconUrl}
																													style={{ width: 12, height: 12 }}
																													width={12}
																													height={12}
																												/>
																											)}
																										</div>
																									)}
																									{chain.name}
																								</button>
																								<button type="button" style={{ color: 'darkcyan' }}>
																									Connected
																								</button>
																								<button onClick={openAccountModal} type="button">
																									{account.displayName}
																									{account.displayBalance
																										? ` (${account.displayBalance})`
																										: ''}
																								</button>
																							</div>
																						);
																					})()}
																				</div>
																			);
																		}}
																	</ConnectButton.Custom>
																</Menu.Item>
															</Menu.Items>
														</Transition>
													</Menu>
											}

											<Popover
												title={<ColorPicker onChange={handleColorChange} />}
												trigger="click"
												open={openColorWidgetPopOver}
												onOpenChange={handleOpenColorWidgetChange}
											>
												<Image src={ColorPickImage} alt='image' className='rounded-xl cursor-pointer' style={{ width: '100px', height: '60px' }} />
											</Popover>

											<ThemeToggler />
											{/* 
											<div className='hidden xl:block dark:text-[#32CD32]'>
												<HamburgerModal />
											</div> */}

										</div>
									</div>

									<div className="flex xl:hidden">
										<Disclosure.Button
											className="
												relative inline-flex items-center justify-center rounded-md bg-gray-800 p-2 
												text-gray-400 hover:text-[#25c2a0] focus:outline-none focus:ring-2 
												focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-800
											"
										>
											<span className="absolute -inset-0.5" />
											<span className="sr-only">Open main menu</span>
											{open ? (
												<XMarkIcon className="block h-6 w-6" aria-hidden="true" />
											) : (
												<Bars3Icon className="block h-6 w-6" aria-hidden="true" />
											)}
										</Disclosure.Button>
									</div>
								</div>
							</div>

							<Disclosure.Panel className="xl:hidden">
								<div className="space-y-1 pb-3 pt-2 px-3">
									{navigation.map((item) => (
										<Disclosure.Button
											key={item.name}
											as="a"
											href={item.href}
											className={classNames(
												item.current ? 'bg-gray-900 text-white' : 'dark:text-white hover:text-[#25c2a0]',
												'block rounded-md px-3 py-2', classes.link
											)}
											aria-current={item.current ? 'page' : undefined}
										>
											{item.name}
										</Disclosure.Button>
									))}
								</div>
								<div className="border-t border-gray-700 pb-3 pt-4">

									<div className="flex items-center px-5">
										<div className={classNames(classes.themeTogglerWrapper)}>
											<ThemeToggler />
										</div>
									</div>

									<div className="mt-3 space-y-1 px-2">
										{community.map((item) => (
											<Disclosure.Button
												key={item.name}
												as="a"
												href={item.href}
												target='_blank'
												className="block rounded-md px-3 py-2 dark:text-white text-lg"
											>
												{item.name}
											</Disclosure.Button>
										))}
									</div>
								</div>
							</Disclosure.Panel>
						</>
					)}
				</Disclosure>
			</div>
			{
				isShowConnectWithSubstrateModalOpen
				&&
				<Modal open={isShowConnectWithSubstrateModalOpen} onCancel={handleConnectWithSubstrateModalCancel} footer={null} >
					<div className="flex flex-col">
						<button
							onClick={getChainInfo}
							className="
								w-1/2 mx-auto bg-blue-700 rounded-lg shadow-lg hover:shadow-2xl text-center hover:bg-blue-600 
								duration-200 text-white hover:text-white font-sans font-semibold justify-center px-2 py-2 
								hover:border-blue-300 hover:border-2 hover:border-solid cursor-pointer
							"
						>
							Get Chain Info
						</button>
						{
							chainInfo && nodeName &&
							<div className="flex items-center justify-evenly mt-4">
								Connected to chain <span className="text-cyan-500">{chainInfo}</span> using <span className="text-cyan-500">{nodeName}</span>
							</div>
						}
					</div>
				</Modal>
			}

		</>
	)
}
