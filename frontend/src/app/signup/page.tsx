"use client"
import React, { useState } from 'react';
import Image from 'next/image';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { Input, Button, Checkbox, Typography } from 'antd';
import { AiFillWallet } from 'react-icons/ai';
import { FaSpinner } from 'react-icons/fa6';
import { GithubLoginButton, GoogleLoginButton, MetamaskLoginButton } from 'react-social-login-buttons';
import { usePolkadot } from '@/context';
import { truncateWalletAddress } from '@/utils/tokenPrice';

const { Text } = Typography;

const SignUp: React.FC = () => {
    const [name, setName] = useState<string>('');
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [agree, setAgree] = useState<boolean>(false);
    const [passwordError, setPasswordError] = useState<string | null>(null);
    const { isInitialized, handleConnect, selectedAccount } = usePolkadot()

    const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setName(event.target.value);
    };

    const handleEmailChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setEmail(event.target.value);
    };

    const handlePasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        setPassword(value);
        setPasswordError(value.length < 8 ? 'Password is too short - must be at least 8 chars.' : null);
    };

    const handleAgreeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setAgree(event.target.checked);
    };

    const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (password.length >= 8 && agree) {
            console.log('Form submitted:', { name, email, password, agree });
        }
    };

    return (
        <div className="flex min-h-screen items-center justify-center dark:bg-[#161C3B] transition-all bg-[url(/img/dots-bg.svg)] dark:bg-[url(/img/dot-bg-dark.svg)]">
            <div className="flex w-full max-w-7xl h-[700px] rounded-xl shadow-lg overflow-hidden bg-white dark:bg-[#121828]">
                <div className="w-full p-8 md:w-1/2">
                    <div className="flex flex-col items-center dark:text-white">
                        <Image src="/img/frontpage/comai-logo.png" alt="logo" width={80} height={80} />
                        <Typography.Title level={2} className="mt-8 font-bold dark:text-white">Sign up</Typography.Title>
                        <div className="mt-2 flex items-baseline font-medium">
                            <Text className='dark:text-white'>Already have an account?</Text>
                            <a className="ml-2 text-blue-500" href="/signin">Sign in</a>
                        </div>
                    </div>
                    <form name="signupForm" noValidate onSubmit={handleSubmit} className="mt-8 flex flex-col">
                        <Input
                            placeholder="Name"
                            value={name}
                            onChange={handleNameChange}
                            className="mb-4 rounded-lg"
                        />
                        <Input
                            placeholder="Email"
                            value={email}
                            onChange={handleEmailChange}
                            className="mb-4 rounded-lg"
                        />
                        <Input.Password
                            placeholder="Password"
                            value={password}
                            onChange={handlePasswordChange}
                            className="mb-4"
                            status={passwordError ? 'error' : ''}
                        />
                        {passwordError && <Text type="danger">{passwordError}</Text>}
                        <div className="flex items-center mt-4">
                            <Checkbox value={agree} onChange={({ target: { value } }) => setAgree(value)} className='dark:text-white'>
                                I agree to the terms and conditions
                            </Checkbox>
                        </div>
                        <Button
                            type="primary"
                            htmlType="submit"
                            className="mt-4 w-full"
                            disabled={password.length < 8 || !agree}
                        >
                            Sign up
                        </Button>
                        <div className="flex items-center my-4 dark:text-white">
                            <div className="flex-1 border-t border-gray-300" />
                            <Text className="mx-4 dark:text-white">Or continue with</Text>
                            <div className="flex-1 border-t border-gray-300" />
                        </div>
                        <div className="flex justify-between items-center">
                            <GoogleLoginButton onClick={() => console.log('--------------')} />
                            <GithubLoginButton onClick={() => console.log('--------------')} />
                        </div>
                        <div className="flex justify-between items-center ">
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
                                            className='w-1/2'
                                        >
                                            {(() => {
                                                if (!connected) {
                                                    return (
                                                        <MetamaskLoginButton onClick={openConnectModal} className='w-full' />
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
                            {
                                isInitialized && selectedAccount ? (
                                    <div className="flex items-center w-[288px] h-[60px]">

                                        <div className="relative flex items-center bg-white rounded-sm shadow px-4 py-2 w-full h-full">
                                            <button className="flex items-center cursor-pointer">
                                                <AiFillWallet size={24} className="text-purple dark:text-black" />
                                                <span className="ml-2 font-mono dark:text-black">
                                                    {truncateWalletAddress(selectedAccount.address)}
                                                </span>
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex items-center w-[281px] h-[50px]">
                                        {!isInitialized && <FaSpinner className="spinner" />}
                                        <Button
                                            size="large"
                                            className='flex items-center justify-start w-full h-full'
                                            onClick={handleConnect}
                                            disabled={!isInitialized}
                                        >
                                            <Image
                                                width={25}
                                                height={15}
                                                className="cursor-pointer mr-2"
                                                alt="Tailwind CSS Navbar component"
                                                src="/img/polkadot.png" />
                                            Connect with Polkadot
                                        </Button>
                                    </div>
                                )
                            }
                        </div>
                    </form>
                </div>
                <div className="relative hidden md:flex md:w-1/2 items-center justify-center p-8">
                    <svg
                        className="pointer-events-none absolute inset-0"
                        viewBox="0 0 960 540"
                        width="100%"
                        height="100%"
                        preserveAspectRatio="xMidYMax slice"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <g className="opacity-20" fill="none" stroke="currentColor" strokeWidth="100">
                            <circle r="234" cx="196" cy="23" />
                            <circle r="234" cx="790" cy="491" />
                        </g>
                    </svg>
                    <svg className="absolute -right-64 -top-64 opacity-20" viewBox="0 0 220 192" fill="none">
                        <defs>
                            <pattern id="837c3e70-6c3a-44e6-8854-cc48c737b659" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
                                <rect x="0" y="0" width="4" height="4" fill="currentColor" />
                            </pattern>
                        </defs>
                        <rect width="220" height="192" fill="url(#837c3e70-6c3a-44e6-8854-cc48c737b659)" />
                    </svg>
                    <div className="relative z-10 text-center text-white max-w-lg">
                        <Typography.Title level={1} className='dark:text-white'>Join our community</Typography.Title>
                        <Text className="text-lg mt-4 dark:text-white">Commune is a protocol that aims to connect all developer tools into one network, fostering a more shareable, reusable, and open economy. It follows an inclusive design philosophy that is based on being maximally unopinionated. This means that developers can leverage Commune as a versatile set of tools alongside their existing projects and have the freedom to incorporate additional tools they find valuable.</Text>
                        {/* <div className="mt-8 flex justify-center">
                            <AvatarGroup>
                                <Avatar src="/img/people/male-16.jpg" />
                                <Avatar src="/img/people/male-09.jpg" />
                                <Avatar src="/img/people/female-11.jpg" />
                                <Avatar src="/img/people/female-18.jpg" />
                            </AvatarGroup>
                            <Text className="ml-4 dark:text-white">More than 17k people joined us, it's your turn</Text>
                        </div> */}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SignUp;
