let provider;
let signer;
let aggregatorContract;
let stablecoinContract;

const AGGREGATOR_ADDRESS = 'YOUR_DEPLOYED_CONTRACT_ADDRESS';
const STABLECOIN_ADDRESS = 'YOUR_STABLECOIN_ADDRESS';

const AGGREGATOR_ABI = [
    'function deposit(uint256 _amount) external',
    'function withdraw(uint256 _shares) external',
    'function harvestYield() external',
    'function claimPoolShares() external',
    'function balanceOf(address account) external view returns (uint256)',
    'function getTotalValue() external view returns (uint256)',
    'function totalYieldGenerated() external view returns (uint256)',
    'function getPoolSharesAvailable() external view returns (uint256)',
    'function totalPoolShares() external view returns (uint256)',
    'function userPoolShares(address user) external view returns (uint256)',
    'function totalDeposited() external view returns (uint256)'
];

const ERC20_ABI = [
    'function approve(address spender, uint256 amount) external returns (bool)',
    'function balanceOf(address account) external view returns (uint256)',
    'function decimals() external view returns (uint8)'
];

let currentTab = 'deposit';

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    event.target.classList.add('active');
    document.getElementById(tabName).classList.add('active');
    currentTab = tabName;
}

async function connectWallet() {
    try {
        if (typeof window.ethereum === 'undefined') {
            alert('Please install MetaMask!');
            return;
        }

        provider = new ethers.providers.Web3Provider(window.ethereum);
        await provider.send('eth_requestAccounts', []);
        signer = provider.getSigner();
        
        const address = await signer.getAddress();
        document.getElementById('walletInfo').innerHTML = `<span>Wallet:</span> ${address.substring(0, 6)}...${address.substring(38)}`;
        
        aggregatorContract = new ethers.Contract(AGGREGATOR_ADDRESS, AGGREGATOR_ABI, signer);
        stablecoinContract = new ethers.Contract(STABLECOIN_ADDRESS, ERC20_ABI, signer);
        
        await updateStats();
        
    } catch (error) {
        console.error('Error connecting wallet:', error);
        alert('Failed to connect wallet');
    }
}

async function updateStats() {
    try {
        if (!aggregatorContract) return;
        
        const address = await signer.getAddress();
        
        const tvl = await aggregatorContract.getTotalValue();
        const totalYield = await aggregatorContract.totalYieldGenerated();
        const userBalance = await aggregatorContract.balanceOf(address);
        const poolSharesAvailable = await aggregatorContract.getPoolSharesAvailable();
        const totalPoolShares = await aggregatorContract.totalPoolShares();
        const userPoolShares = await aggregatorContract.userPoolShares(address);
        
        document.getElementById('tvl').textContent = `$${ethers.utils.formatUnits(tvl, 6)}`;
        document.getElementById('totalYield').textContent = `$${ethers.utils.formatUnits(totalYield, 6)}`;
        document.getElementById('userBalance').textContent = `${ethers.utils.formatEther(userBalance)} YAT`;
        document.getElementById('poolShares').textContent = `${ethers.utils.formatEther(poolSharesAvailable)}`;
        
        document.getElementById('totalPoolShares').textContent = ethers.utils.formatEther(totalPoolShares);
        document.getElementById('userPoolShares').textContent = ethers.utils.formatEther(userPoolShares);
        
        const poolPercentage = (parseFloat(ethers.utils.formatEther(totalPoolShares)) / 21000000) * 100;
        document.getElementById('poolProgressBar').style.width = `${poolPercentage}%`;
        document.getElementById('poolProgressBar').textContent = `${poolPercentage.toFixed(2)}%`;
        
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

async function deposit() {
    try {
        const amount = document.getElementById('depositAmount').value;
        if (!amount || parseFloat(amount) <= 0) {
            alert('Please enter a valid amount');
            return;
        }
        
        const amountWei = ethers.utils.parseUnits(amount, 6);
        
        const approveTx = await stablecoinContract.approve(AGGREGATOR_ADDRESS, amountWei);
        await approveTx.wait();
        
        const depositTx = await aggregatorContract.deposit(amountWei);
        await depositTx.wait();
        
        alert('Deposit successful!');
        document.getElementById('depositAmount').value = '';
        await updateStats();
        
    } catch (error) {
        console.error('Error depositing:', error);
        alert('Deposit failed: ' + error.message);
    }
}

async function withdraw() {
    try {
        const shares = document.getElementById('withdrawAmount').value;
        if (!shares || parseFloat(shares) <= 0) {
            alert('Please enter a valid amount');
            return;
        }
        
        const sharesWei = ethers.utils.parseEther(shares);
        
        const withdrawTx = await aggregatorContract.withdraw(sharesWei);
        await withdrawTx.wait();
        
        alert('Withdrawal successful!');
        document.getElementById('withdrawAmount').value = '';
        await updateStats();
        
    } catch (error) {
        console.error('Error withdrawing:', error);
        alert('Withdrawal failed: ' + error.message);
    }
}

async function harvestYield() {
    try {
        const harvestTx = await aggregatorContract.harvestYield();
        await harvestTx.wait();
        
        alert('Yield harvested successfully!');
        await updateStats();
        
    } catch (error) {
        console.error('Error harvesting:', error);
        alert('Harvest failed: ' + error.message);
    }
}

async function claimPoolShares() {
    try {
        const claimTx = await aggregatorContract.claimPoolShares();
        await claimTx.wait();
        
        alert('Pool shares claimed successfully!');
        await updateStats();
        
    } catch (error) {
        console.error('Error claiming:', error);
        alert('Claim failed: ' + error.message);
    }
}

window.addEventListener('load', async () => {
    if (typeof window.ethereum !== 'undefined') {
        await connectWallet();
        
        setInterval(updateStats, 30000);
    }
});

window.ethereum?.on('accountsChanged', () => {
    window.location.reload();
});

window.ethereum?.on('chainChanged', () => {
    window.location.reload();
});