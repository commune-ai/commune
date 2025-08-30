 # start of file
import { createContext, useContext, useState, useEffect } from 'react';
import { ethers } from 'ethers';
import Web3Modal from 'web3modal';

const Web3Context = createContext();

export function Web3Provider({ children }) {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [account, setAccount] = useState(null);
  const [chainId, setChainId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [contracts, setContracts] = useState({
    registry: null,
    propertyToken: null,
  });

  // Initialize web3modal
  const initWeb3Modal = async () => {
    try {
      const web3Modal = new Web3Modal({
        network: "localhost",
        cacheProvider: true,
        providerOptions: {},
      });
      return web3Modal;
    } catch (error) {
      console.error("Failed to initialize Web3Modal", error);
      return null;
    }
  };

  // Connect to wallet
  const connect = async () => {
    try {
      const web3Modal = await initWeb3Modal();
      if (!web3Modal) return;

      const connection = await web3Modal.connect();
      const ethersProvider = new ethers.providers.Web3Provider(connection);
      const ethersSigner = ethersProvider.getSigner();
      const accounts = await ethersProvider.listAccounts();
      const network = await ethersProvider.getNetwork();

      setProvider(ethersProvider);
      setSigner(ethersSigner);
      setAccount(accounts[0]);
      setChainId(network.chainId);
      setIsConnected(true);

      // Load contract instances
      await loadContracts(ethersProvider);

      // Setup event listeners
      connection.on("accountsChanged", (accounts) => {
        setAccount(accounts[0]);
        loadContracts(ethersProvider);
      });

      connection.on("chainChanged", (chainId) => {
        window.location.reload();
      });

      return true;
    } catch (error) {
      console.error("Failed to connect to wallet", error);
      return false;
    }
  };

  // Disconnect wallet
  const disconnect = async () => {
    try {
      const web3Modal = await initWeb3Modal();
      if (web3Modal) {
        web3Modal.clearCachedProvider();
      }
      setProvider(null);
      setSigner(null);
      setAccount(null);
      setChainId(null);
      setIsConnected(false);
      setContracts({
        registry: null,
        propertyToken: null,
      });
    } catch (error) {
      console.error("Failed to disconnect wallet", error);
    }
  };

  // Load contract instances
  const loadContracts = async (provider) => {
    try {
      // In a real app, we would load ABIs and addresses from deployment artifacts
      // For now, we'll use placeholders
      
      // Attempt to fetch deployment info
      let deploymentInfo;
      try {
        const response = await fetch('/deployment.json');
        deploymentInfo = await response.json();
      } catch (error) {
        console.error("Failed to load deployment info", error);
        return;
      }

      // Load contract ABIs
      const registryAbi = []; // This would be the actual ABI
      const propertyTokenAbi = []; // This would be the actual ABI

      // Create contract instances
      const registry = new ethers.Contract(
        deploymentInfo.registryAddress,
        registryAbi,
        provider
      );

      const propertyToken = new ethers.Contract(
        deploymentInfo.samplePropertyTokenAddress,
        propertyTokenAbi,
        provider
      );

      setContracts({
        registry,
        propertyToken,
      });
    } catch (error) {
      console.error("Failed to load contracts", error);
    }
  };

  // Auto-connect if cached provider exists
  useEffect(() => {
    const autoConnect = async () => {
      const web3Modal = await initWeb3Modal();
      if (web3Modal && web3Modal.cachedProvider) {
        connect();
      }
    };

    autoConnect();
  }, []);

  const value = {
    provider,
    signer,
    account,
    chainId,
    isConnected,
    contracts,
    connect,
    disconnect,
  };

  return <Web3Context.Provider value={value}>{children}</Web3Context.Provider>;
}

export function useWeb3() {
  return useContext(Web3Context);
}
