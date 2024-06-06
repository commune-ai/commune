"use client"
import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { ConnectButton } from "@rainbow-me/rainbowkit";
import Modal from "antd/es/modal/Modal";
import { usePolkadot } from "@/context"
import { toast } from 'react-toastify';
import { AiFillWallet} from "react-icons/ai"
import { FaSpinner } from "react-icons/fa6"
import { truncateWalletAddress } from "@/utils"
import GitHubLogin from "react-github-login";
import GithubImage from "../../../public/svg/github-mark.svg";
import MetaMaskImage from "../../../public/svg/metamask.svg";
import PolkadotImage from "../../../public/svg/polkadot.svg";

const words: string[] = [
  "developers.",
  "designers.",
  "creators.",
  "everyone.",
  "<END>",
];
const colour: string[] = [
  "text-[#00000]",
  "text-[#ffb4ed] dark:text-[#FFD6F5]",
  "text-[#FF8F8F]  dark:text-[#FF8F8F]",
  "text-[#ffef40] dark:text-[#FFF7A1]",
];

const TITLE = "Commune AI";
const TAGLINE = "Renovating the way we build software for ";

export default function HomepageHeader() {

  const [index, setIndex] = useState(0);
  const [subIndex, setSubIndex] = useState(0);
  const [blink, setBlink] = useState(true);
  const [reverse, setReverse] = useState(false);
  const [isShowAuthModalOpen, setIsShowAuthModalOpen] = useState(false)

  const { isInitialized, handleConnect, selectedAccount } = usePolkadot()

  const [isLoggedIn, setIsLoggedIn] = useState(false);
  console.log('-----------This is the loginstatus--------', isLoggedIn)
  const [metamaskAddress, setMetamaskAddress] = useState<string | undefined>('')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  // state of the scroll position and header height
  const [scrollPosition] = useState(0);
  console.log('-------------This is the scrollPosition------------', scrollPosition)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const headerRef = useRef<any>(null);
  const [headerHeight, setHeaderHeight] = useState(20);
  console.log('---------------This is teh headerHeight------------', headerHeight)
  // typeWriter effect
  // give me the context of this whole useEffect
  useEffect(() => {
    if (index === words.length) return; // if end of words, return
    // if subIndex is equal to the length of the word + 1 and index is not the last word and not reverse
    if (subIndex === words[index].length + 1 && index !== words.length - 1 && !reverse) {
      setReverse(true);
      return;
    }
    // if subIndex is equal to 0 and reverse is true
    if (subIndex === 0 && reverse) {
      setReverse(false);
      setIndex((prev) => prev + 1);
      return;
    }
    // if reverse is true, subIndex is not 0 and index is not the last word
    if (index === words.length - 1)
      setIndex(() => 0)
    // if reverse is true, subIndex is not 0 and index is not the last word
    // if reverse is false, subIndex is not the length of the word and index is not the last word
    const timeout = setTimeout(() => {
      setSubIndex((prev) => prev + (reverse ? -1 : 1));
    }, Math.max(reverse ? 75 : subIndex === words[index].length ? 1000 :
      75, 25));
    return () => clearTimeout(timeout);
  }, [subIndex, index, reverse]);


  // blinker effect
  useEffect(() => {
    const timeout2 = setTimeout(() => {
      setBlink((prev) => !prev);
    }, 250);
    if (index === words.length) return;

    return () => clearTimeout(timeout2);
  }, [blink]);

  // Handle scroll position
  // const handleScroll = () => {
  //   const position = window.pageYOffset;
  //   setScrollPosition(position);
  // };

  // Add scroll event listener to window
  // useEffect(() => {
  //   window.addEventListener('scroll', handleScroll, { passive: true });

  //   return () => {
  //     window.removeEventListener('scroll', handleScroll);
  //   };
  // }, []);

  // Get header height on mount and when window is resized
  // This is to offset the scroll position so that the header
  useEffect(() => {
    if (headerRef?.current) {
      setHeaderHeight(headerRef.current.clientHeight);
    }
  }, [headerRef.current]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onGitHubLoginSuccess = (response: any) => {

    setIsShowAuthModalOpen(false)

    const accessToken = response.code;

    const getUserInfo = async (accessToken: string) => {

      const response = await fetch('https://api.github.com/user', {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });

      if (response.ok) {

        const userInfo = await response.json();
        setIsLoggedIn(true)
        // Handle user information (userInfo)
        console.log('------github account------', userInfo);

      } else {
        // Handle error
        console.error('Failed to fetch user information from GitHub API');
      }
    };

    // Call this function with the access token obtained after successful login
    getUserInfo(accessToken);

  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onGitHubLoginFailure = (response: any) => {
    console.log('------the data from github-----failed-----', response);
  }

  //const dispatch = useDispatch<any>()

  useEffect(()=>{
    if(metamaskAddress){
      console.log('-----------This is an wallet address---------', metamaskAddress);
      toast.success(`You are logged in with ${metamaskAddress}`, { autoClose: 2000 });
      //dispatch(saveMetaMaskAddress(metamaskAddress));
    }
  }, [metamaskAddress])

  return (
    <header
      ref={headerRef}
      className={`
        relative 
        z-10 
        h-[100vh]
        dark:bg-gray-900 
        p-[4rem] 
        text-center 
        overflow-hidden 
        duration-500
        bg-[url(/img/dots-bg.svg)] dark:bg-[url(/img/dot-bg-dark.svg)]
        `}
    >          
      <div className="absolute right-0 top-0 z-[-1] opacity-30 lg:opacity-100">
        <svg width="450" height="556" viewBox="0 0 450 556" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle
            cx="277"
            cy="63"
            r="225"
            fill="url(#paint0_linear_25:217)"
          />
          <circle
            cx="17.9997"
            cy="182"
            r="18"
            fill="url(#paint1_radial_25:217)"
          />
          <circle
            cx="76.9997"
            cy="288"
            r="34"
            fill="url(#paint2_radial_25:217)"
          />
          <circle
            cx="325.486"
            cy="302.87"
            r="180"
            transform="rotate(-37.6852 325.486 302.87)"
            fill="url(#paint3_linear_25:217)"
          />
          <circle
            opacity="0.8"
            cx="184.521"
            cy="315.521"
            r="132.862"
            transform="rotate(114.874 184.521 315.521)"
            stroke="url(#paint4_linear_25:217)"
          />
          <circle
            opacity="0.8"
            cx="356"
            cy="290"
            r="179.5"
            transform="rotate(-30 356 290)"
            stroke="url(#paint5_linear_25:217)"
          />
          <circle
            opacity="0.8"
            cx="191.659"
            cy="302.659"
            r="133.362"
            transform="rotate(133.319 191.659 302.659)"
            fill="url(#paint6_linear_25:217)"
          />
          <defs>
            <linearGradient
              id="paint0_linear_25:217"
              x1="-54.5003"
              y1="-178"
              x2="222"
              y2="288"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" />
              <stop offset="1" stopColor="#4A6CF7" stopOpacity="0" />
            </linearGradient>
            <radialGradient
              id="paint1_radial_25:217"
              cx="0"
              cy="0"
              r="1"
              gradientUnits="userSpaceOnUse"
              gradientTransform="translate(17.9997 182) rotate(90) scale(18)"
            >
              <stop offset="0.145833" stopColor="#4A6CF7" stopOpacity="0" />
              <stop offset="1" stopColor="#4A6CF7" stopOpacity="0.08" />
            </radialGradient>
            <radialGradient
              id="paint2_radial_25:217"
              cx="0"
              cy="0"
              r="1"
              gradientUnits="userSpaceOnUse"
              gradientTransform="translate(76.9997 288) rotate(90) scale(34)"
            >
              <stop offset="0.145833" stopColor="#4A6CF7" stopOpacity="0" />
              <stop offset="1" stopColor="#4A6CF7" stopOpacity="0.08" />
            </radialGradient>
            <linearGradient
              id="paint3_linear_25:217"
              x1="226.775"
              y1="-66.1548"
              x2="292.157"
              y2="351.421"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" />
              <stop offset="1" stopColor="#4A6CF7" stopOpacity="0" />
            </linearGradient>
            <linearGradient
              id="paint4_linear_25:217"
              x1="184.521"
              y1="182.159"
              x2="184.521"
              y2="448.882"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" />
              <stop offset="1" stopColor="white" stopOpacity="0" />
            </linearGradient>
            <linearGradient
              id="paint5_linear_25:217"
              x1="356"
              y1="110"
              x2="356"
              y2="470"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" />
              <stop offset="1" stopColor="white" stopOpacity="0" />
            </linearGradient>
            <linearGradient
              id="paint6_linear_25:217"
              x1="118.524"
              y1="29.2497"
              x2="166.965"
              y2="338.63"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" />
              <stop offset="1" stopColor="#4A6CF7" stopOpacity="0" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      <div className="absolute bottom-0 left-0 z-[-1] opacity-30 lg:opacity-100">
        <svg width="364" height="201" viewBox="0 0 364 201" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M5.88928 72.3303C33.6599 66.4798 101.397 64.9086 150.178 105.427C211.155 156.076 229.59 162.093 264.333 166.607C299.076 171.12 337.718 183.657 362.889 212.24"
            stroke="url(#paint0_linear_25:218)"
          />
          <path
            d="M-22.1107 72.3303C5.65989 66.4798 73.3965 64.9086 122.178 105.427C183.155 156.076 201.59 162.093 236.333 166.607C271.076 171.12 309.718 183.657 334.889 212.24"
            stroke="url(#paint1_linear_25:218)"
          />
          <path
            d="M-53.1107 72.3303C-25.3401 66.4798 42.3965 64.9086 91.1783 105.427C152.155 156.076 170.59 162.093 205.333 166.607C240.076 171.12 278.718 183.657 303.889 212.24"
            stroke="url(#paint2_linear_25:218)"
          />
          <path
            d="M-98.1618 65.0889C-68.1416 60.0601 4.73364 60.4882 56.0734 102.431C120.248 154.86 139.905 161.419 177.137 166.956C214.37 172.493 255.575 186.165 281.856 215.481"
            stroke="url(#paint3_linear_25:218)"
          />
          <circle
            opacity="0.8"
            cx="214.505"
            cy="60.5054"
            r="49.7205"
            transform="rotate(-13.421 214.505 60.5054)"
            stroke="url(#paint4_linear_25:218)"
          />
          <circle cx="220" cy="63" r="43" fill="url(#paint5_radial_25:218)" />
          <defs>
            <linearGradient
              id="paint0_linear_25:218"
              x1="184.389"
              y1="69.2405"
              x2="184.389"
              y2="212.24"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" stopOpacity="0" />
              <stop offset="1" stopColor="#4A6CF7" />
            </linearGradient>
            <linearGradient
              id="paint1_linear_25:218"
              x1="156.389"
              y1="69.2405"
              x2="156.389"
              y2="212.24"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" stopOpacity="0" />
              <stop offset="1" stopColor="#4A6CF7" />
            </linearGradient>
            <linearGradient
              id="paint2_linear_25:218"
              x1="125.389"
              y1="69.2405"
              x2="125.389"
              y2="212.24"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" stopOpacity="0" />
              <stop offset="1" stopColor="#4A6CF7" />
            </linearGradient>
            <linearGradient
              id="paint3_linear_25:218"
              x1="93.8507"
              y1="67.2674"
              x2="89.9278"
              y2="210.214"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" stopOpacity="0" />
              <stop offset="1" stopColor="#4A6CF7" />
            </linearGradient>
            <linearGradient
              id="paint4_linear_25:218"
              x1="214.505"
              y1="10.2849"
              x2="212.684"
              y2="99.5816"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4A6CF7" />
              <stop offset="1" stopColor="#4A6CF7" stopOpacity="0" />
            </linearGradient>
            <radialGradient
              id="paint5_radial_25:218"
              cx="0"
              cy="0"
              r="1"
              gradientUnits="userSpaceOnUse"
              gradientTransform="translate(220 63) rotate(90) scale(43)"
            >
              <stop offset="0.145833" stopColor="white" stopOpacity="0" />
              <stop offset="1" stopColor="white" stopOpacity="0.08" />
            </radialGradient>
          </defs>
        </svg>
      </div>

      <Image
        src="/gif/logo/commune.webp"
        alt="Commune Logo"
        className='block lg:hidden'
        width={5000}
        height={5000}
      />
      <div className="px-10 py-5 m-auto">
        <div className='flex lg:flex-row flex-col mt-[100px]'>
          <div className='m-auto w-full lg:w-1/2 flex flex-col items-center justify-center'>
            <div className='w-auto sm:w-[710px] sm:h-[250px]'>
              <h1 className="text-4xl sm:text-6xl sm:pb-3 dark:text-[#32CD32]">{TITLE}</h1>
              <div className='hidden sm:block'>
                <p className="hero__subtitle text-xl sm:text-4xl dark:text-[#32CD32]">{TAGLINE}
                  <br />
                  <span
                    className={`hero__subtitle text-4xl ${colour[index]} font-semibold mb-5`}
                  >
                    {`${words[index].substring(0, subIndex)}${blink ? "|" : ""}`}
                  </span>
                </p>
              </div>
            </div>
            {/* 
              <div className='w-[10rem] h-[5rem]'>
                <div
                  className='
                    bg-blue-700 rounded-lg shadow-lg hover:shadow-2xl text-center 
                    hover:bg-blue-600 duration-200 text-xl text-white hover:text-white 
                    font-sans font-semibold justify-center px-2 py-2 cursor-pointer
                  '
                  onClick={() => setIsShowAuthModalOpen(true)}
                >
                  Get Started
                </div>
              </div>
              */}
              
          </div>
          <div className='hidden lg:block w-full lg:w-1/2 h-full lg:-mr-44 '>
            <Image src="/gif/logo/commune.gif" width={500} height={500} alt="Commune Logo" className='' />
          </div>
        </div>
      </div>
      {
        isShowAuthModalOpen &&
        <Modal open={isShowAuthModalOpen} onCancel={() => setIsShowAuthModalOpen(false)} footer={null} width={500}>
          <div className='flex items-center justify-center'>
            <span style={{ fontWeight: '500', alignItems: 'center', display: 'flex', fontSize: '2rem' }}>
              Connect to Commune AI
            </span>
          </div>
          <div className='flex items-center justify-evenly mt-14 mb-14 flex-col'>
            <div className='flex w-full items-center justify-evenly cursor-pointer'>
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
                  // Note: If your app doesn't use authentication, you
                  // can remove all 'authenticationStatus' checks
                  const ready = mounted && authenticationStatus !== 'loading';
                  account?.address&&setMetamaskAddress(account.address)
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
                            <div className='
                                flex items-center justify-center hover:bg-gray-300 p-2 w-[105.77px] h-[105.77px] rounded-md dark:text-white hover: text-black
                              '
                              style={{ flexDirection: 'column', border: '1px solid gray' }} onClick={openConnectModal}
                            >
                              <Image src={MetaMaskImage} alt='login with Metamask' width={50} height={50} className='cursor-pointer mb-1' />
                              <button type="button">
                                Metamask
                              </button>
                            </div>
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
              <div className='
                  flex items-center justify-center p-2 rounded-md hover:bg-gray-300 w-[105.77px] h-[105.77px]
                '
                style={{ flexDirection: 'column', border: '1px solid gray' }}
              >
                <Image src={GithubImage} alt='login with Github' width={50} height={50} className='cursor-pointer mb-1' />
                <GitHubLogin clientId='8386c0df1514607054e7'
                  buttonText="Github"
                  style={{ marginTop: '8px' }}
                  onSuccess={onGitHubLoginSuccess}
                  onFailure={onGitHubLoginFailure}
                  redirectUri={'http://localhost:3000/modules'}
                />
              </div>
             
                
                {
                  isInitialized && selectedAccount ? (
                  <div className="flex items-center">
                    <div className="relative flex items-center bg-white rounded-full shadow px-4 py-2">
                      <button className="flex items-center cursor-pointer">
                        <AiFillWallet size={24} className="text-purple" />
                        <span className="ml-2 font-mono">
                          {truncateWalletAddress(selectedAccount.address)}
                        </span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="
                  transition-all duration-300 flex items-center justify-center flex-col border-[1px] 
                  border-[gray] p-2 rounded-md hover:bg-gray-300 w-[105.77px] h-[105.77px]
                "
              >
                  <div className="flex items-center gap-x-2">
                    {!isInitialized && <FaSpinner className="spinner" />}
                    <button onClick={handleConnect} className="w-full h-full flex justify-center items-center flex-col">
                      <Image className="w-[60px] h-[60px]" width={50} height={50} src={PolkadotImage} alt="Polkadot" />
                      <span>Comwallet</span>
                    </button>
                  </div>
              </div>

                )}
            </div>
          </div>
        </Modal>
      }
    </header>
  );
}

export const getHeaderClasses = (position: number, height: number) => {
  if (position > height / 2) {
    return "rounded-b-lg shadow-lg mx-5";
  }
  return "";
};
