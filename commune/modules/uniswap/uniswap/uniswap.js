const { ethers } = require("ethers");
const { AlphaRouter } = require("@uniswap/smart-order-router");
const { Token, CurrencyAmount, TradeType, Percent } = require("@uniswap/sdk-core");
const fetch = require("node-fetch"); // For historical data API calls

class UniswapTrader {
  constructor({
    providerUrl = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID",
    privateKey = "YOUR_PRIVATE_KEY",
    fromTokenAddress = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", // UNI
    toTokenAddress = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", // WETH
    fromDecimals = 18,
    toDecimals = 18,
    slippagePercent = 1, // 1% default slippage
  }) {
    // Setup provider and wallet
    this.provider = new ethers.providers.JsonRpcProvider(providerUrl);
    this.wallet = new ethers.Wallet(privateKey, this.provider);
    this.router = new AlphaRouter({ chainId: 1, provider: this.provider });

    // Define tokens
    this.fromToken = new Token(1, fromTokenAddress, fromDecimals, "FROM", "From Token");
    this.toToken = new Token(1, toTokenAddress, toDecimals, "TO", "To Token");

    // Config
    this.swapRouterAddress = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"; // Uniswap V3 Router
    this.slippage = new Percent(slippagePercent, 100);
    this.approved = false; // Track approval status
  }

  // Approve token for Uniswap to spend
  async approve(amount) {
    const tokenContract = new ethers.Contract(this.fromToken.address, [
      "function approve(address spender, uint256 amount) external returns (bool)",
    ], this.wallet);

    const amountWei = ethers.utils.parseUnits(amount.toString(), this.fromToken.decimals);
    const tx = await tokenContract.approve(this.swapRouterAddress, amountWei);
    await tx.wait();
    this.approved = true;
    console.log(`Approved ${amount} ${this.fromToken.symbol} for Uniswap`);
    return true;
  }

  // Get current price (fromToken -> toToken)
  async getCurrentPrice() {
    const amountIn = CurrencyAmount.fromRawAmount(this.fromToken, ethers.utils.parseUnits("1", this.fromToken.decimals));
    const route = await this.router.route(amountIn, this.toToken, TradeType.EXACT_INPUT, {
      slippageTolerance: this.slippage,
    });

    if (!route) throw new Error("No route found");
    const price = route.quote.toExact();
    console.log(`Current ${this.fromToken.symbol} price: ${price} ${this.toToken.symbol}`);
    return price; // Returns price as a string (e.g., "0.01234" WETH per UNI)
  }

  // Sell tokens incrementally
  async sell(amount, minPrice = null) {
    if (!this.approved) throw new Error("Token not approved yet. Call approve() first.");

    const currentPrice = ethers.utils.parseEther(await this.getCurrentPrice());
    if (minPrice && currentPrice.lt(ethers.utils.parseEther(minPrice.toString()))) {
      console.log(`Price ${ethers.utils.formatEther(currentPrice)} < ${minPrice}. Skipping sale.`);
      return false;
    }

    const amountIn = CurrencyAmount.fromRawAmount(this.fromToken, ethers.utils.parseUnits(amount.toString(), this.fromToken.decimals));
    const route = await this.router.route(amountIn, this.toToken, TradeType.EXACT_INPUT, {
      recipient: this.wallet.address,
      slippageTolerance: this.slippage,
      deadline: Math.floor(Date.now() / 1000) + 60 * 20, // 20 min deadline
    });

    if (!route) throw new Error("No swap route found");

    const tx = {
      data: route.methodParameters.calldata,
      to: this.swapRouterAddress,
      value: route.methodParameters.value,
      from: this.wallet.address,
    };

    const txResponse = await this.wallet.sendTransaction(tx);
    const receipt = await txResponse.wait();
    console.log(`Sold ${amount} ${this.fromToken.symbol}. Tx: ${receipt.transactionHash}`);
    return receipt.transactionHash;
  }

  // Get historical price data from CoinGecko (daily, adjustable)
  async getHistoricalPrices(days = 365) {
    const url = `https://api.coingecko.com/api/v3/coins/uniswap/market_chart?vs_currency=eth&days=${days}&interval=daily`;
    const response = await fetch(url);
    const data = await response.json();

    if (!data.prices) throw new Error("Failed to fetch historical data");

    const historicalData = data.prices.map(([timestamp, price]) => ({
      date: new Date(timestamp).toISOString().split("T")[0],
      price: price, // Price in ETH (WETH equivalent)
    }));

    console.log(`Fetched ${historicalData.length} days of UNI/ETH historical data`);
    return historicalData;
  }

  // Run the bot with price checking and incremental selling
  async runBot({
    minPrice = "0.01", // Minimum price in toToken (e.g., WETH)
    amountPerTrade = "100", // Amount to sell per trade
    intervalSeconds = 60, // Check interval
  }) {
    console.log(`Starting Uniswap bot for ${this.fromToken.symbol}/${this.toToken.symbol}...`);

    if (!this.approved) {
      await this.approve(ethers.utils.parseUnits("1000", this.fromToken.decimals)); // Approve 1000 tokens by default
    }

    const checkAndSell = async () => {
      try {
        const price = await this.getCurrentPrice();
        const priceWei = ethers.utils.parseEther(price);

        if (priceWei.gte(ethers.utils.parseEther(minPrice))) {
          console.log(`Price ${price} >= ${minPrice}. Selling ${amountPerTrade} ${this.fromToken.symbol}...`);
          await this.sell(amountPerTrade, minPrice);
        } else {
          console.log(`Price ${price} < ${minPrice}. Waiting...`);
        }
      } catch (error) {
        console.error("Bot error:", error.message);
      }
    };

    // Run immediately, then every interval
    await checkAndSell();
    setInterval(checkAndSell, intervalSeconds * 1000);
  }
}

// Example usage
async function main() {
  const trader = new UniswapTrader({
    providerUrl: "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID",
    privateKey: "YOUR_PRIVATE_KEY",
    fromTokenAddress: "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", // UNI
    toTokenAddress: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", // WETH
    slippagePercent: 1,
  });

  // Get historical prices
  const historical = await trader.getHistoricalPrices(30); // Last 30 days
  console.log("Historical Prices:", historical.slice(0, 5)); // First 5 days

  // Get current price
  const price = await trader.getCurrentPrice();
  console.log("Current Price:", price);

  // Sell manually
  await trader.sell("50"); // Sell 50 UNI

  // Run the bot
  trader.runBot({
    minPrice: "0.01", // Sell if UNI >= 0.01 WETH
    amountPerTrade: "100", // Sell 100 UNI per trade
    intervalSeconds: 60, // Check every minute
  });
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = UniswapTrader;