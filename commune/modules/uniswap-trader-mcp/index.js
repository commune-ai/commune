const { McpServer } = require("@modelcontextprotocol/sdk/server/mcp.js");
const { StdioServerTransport } = require("@modelcontextprotocol/sdk/server/stdio.js");
const { z } = require("zod");
const ethers = require("ethers");
const { 
  Token,
  CurrencyAmount,
  TradeType,
  Percent,
  SwapRouter
} = require("@uniswap/sdk-core");
const { AlphaRouter, SwapType } = require("@uniswap/smart-order-router");

// Define minimal ERC20 ABI with decimals function added
const ERC20ABI = [
  "function balanceOf(address account) external view returns (uint256)",
  "function approve(address spender, uint256 amount) external returns (bool)",
  "function symbol() external view returns (string)",
  "function decimals() external view returns (uint8)"
];

// Define minimal SwapRouter ABI for Uniswap V3 (only exactInput and exactOutput)
const SwapRouterABI = [
  "function exactInput(tuple(address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum, address[] path) params) external payable returns (uint256 amountOut)",
  "function exactOutput(tuple(address recipient, uint256 deadline, uint256 amountOut, uint256 amountInMaximum, address[] path) params) external payable returns (uint256 amountIn)",
  "function multicall(bytes[] calldata data) external payable returns (bytes[] memory results)"
];

// Define minimal WETH9 ABI for deposit and withdraw
const WETHABI = [
  "function deposit() external payable",
  "function withdraw(uint256 wad) external",
  "function balanceOf(address account) external view returns (uint256)"
];

// Load environment variables and chain configurations
require('dotenv').config();
const CHAIN_CONFIGS = require('./chainConfigs');

// Import utilities from ethers.utils for v5
const { parseUnits, formatUnits } = ethers.utils;

const WALLET_PRIVATE_KEY = process.env.WALLET_PRIVATE_KEY;
if (!WALLET_PRIVATE_KEY) {
  throw new Error("WALLET_PRIVATE_KEY environment variable is required");
}

// Initialize MCP server
const server = new McpServer({
  name: "Uniswap Trader MCP",
  version: "1.0.0",
  description: "An MCP server for AI agents to automate trading strategies on Uniswap DEX across multiple blockchains"
});

// Get provider and router for a specific chain
function getChainContext(chainId) {
  const config = CHAIN_CONFIGS[chainId];
  if (!config) {
    const supportedChains = Object.entries(CHAIN_CONFIGS)
      .map(([id, { name }]) => `${id} - ${name}`)
      .join(', ');
    throw new Error(`Unsupported chainId: ${chainId}. Supported chains: ${supportedChains}`);
  }
  const provider = new ethers.providers.JsonRpcProvider(config.rpcUrl);
  const router = new AlphaRouter({ chainId, provider });
  return { provider, router, config };
}

// Create a token instance, fetching decimals for ERC-20 tokens
async function createToken(chainId, address, provider, symbol = "UNKNOWN", name = "Unknown Token") {
  const config = CHAIN_CONFIGS[chainId];
  if (!address || address.toLowerCase() === "native") {
    return new Token(chainId, config.weth, 18, symbol, name); // Native token defaults to 18 decimals
  }
  const tokenContract = new ethers.Contract(address, ERC20ABI, provider);
  const decimals = await tokenContract.decimals();
  console.log('=>', decimals)
  return new Token(chainId, ethers.utils.getAddress(address), decimals, symbol, name);
}

// Check wallet balance, throw error if zero
async function checkBalance(provider, wallet, tokenAddress, isNative = false) {
  if (isNative) {
    const balance = await provider.getBalance(wallet.address);
    if (balance.isZero()) {
      throw new Error(`Zero ${CHAIN_CONFIGS[provider.network.chainId].name} native token balance. Please deposit funds to ${wallet.address}.`);
    }
  } else {
    const tokenContract = new ethers.Contract(tokenAddress, ERC20ABI, provider);
    const balance = await tokenContract.balanceOf(wallet.address);
    if (balance.isZero()) {
      const symbol = await tokenContract.symbol();
      throw new Error(`Zero ${symbol} balance. Please deposit funds to ${wallet.address}.`);
    }
  }
}

// Tool: Get price quote with Smart Order Router
server.tool(
  "getPrice",
  "Get a price quote for a Uniswap swap, supporting multi-hop routes",
  {
    chainId: z.number().default(1).describe("Chain ID (1: Ethereum, 10: Optimism, 137: Polygon, 42161: Arbitrum, 42220: Celo, 56: BNB Chain, 43114: Avalanche, 8453: Base)"),
    tokenIn: z.string().describe("Input token address ('NATIVE' for native token like ETH)"),
    tokenOut: z.string().describe("Output token address ('NATIVE' for native token like ETH)"),
    amountIn: z.string().optional().describe("Exact input amount (required for exactIn trades)"),
    amountOut: z.string().optional().describe("Exact output amount (required for exactOut trades)"),
    tradeType: z.enum(["exactIn", "exactOut"]).default("exactIn").describe("Trade type: exactIn requires amountIn, exactOut requires amountOut")
  },
  async ({ chainId, tokenIn, tokenOut, amountIn, amountOut, tradeType }) => {
    try {
      const { provider, router, config } = getChainContext(chainId);
      
      const tokenA = await createToken(chainId, tokenIn, provider);
      const tokenB = await createToken(chainId, tokenOut, provider);

      if (tradeType === "exactIn" && !amountIn) {
        throw new Error("amountIn is required for exactIn trades");
      }
      if (tradeType === "exactOut" && !amountOut) {
        throw new Error("amountOut is required for exactOut trades");
      }

      const amount = tradeType === "exactIn" ? amountIn : amountOut;
      const decimals = tradeType === "exactIn" ? tokenA.decimals : tokenB.decimals;
      const amountWei = parseUnits(amount, decimals).toString();
      const route = await router.route(
        CurrencyAmount.fromRawAmount(
          tradeType === "exactIn" ? tokenA : tokenB,
          amountWei
        ),
        tradeType === "exactIn" ? tokenB : tokenA,
        tradeType === "exactIn" ? TradeType.EXACT_INPUT : TradeType.EXACT_OUTPUT,
        {
          recipient: ethers.constants.AddressZero,
          slippageTolerance: new Percent(5, 1000),
          deadline: Math.floor(Date.now() / 1000) + 20 * 60,
          type: SwapType.SWAP_ROUTER_02,
        }
      );
      if (!route) throw new Error("No route found");

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            chainId,
            tradeType,
            price: route.trade.executionPrice.toSignificant(6),
            inputAmount: route.trade.inputAmount.toSignificant(6),
            outputAmount: route.trade.outputAmount.toSignificant(6),
            minimumReceived: route.trade.minimumAmountOut(new Percent(5, 1000)).toSignificant(6),
            maximumInput: route.trade.maximumAmountIn(new Percent(5, 1000)).toSignificant(6),
            route: route.trade.swaps.map(swap => ({
              tokenIn: swap.inputAmount.currency.address,
              tokenOut: swap.outputAmount.currency.address,
              fee: swap.route.pools[0].fee
            })),
            estimatedGas: route.estimatedGasUsed.toString()
          }, null, 2)
        }]
      };
    } catch (error) {
      throw new Error(`Failed to get price: ${error.message}. Check network connection.`);
    }
  }
);

// Tool: Execute swap with Smart Order Router
server.tool(
  "executeSwap",
  "Execute a swap on Uniswap with optimal multi-hop routing",
  {
    chainId: z.number().default(1).describe("Chain ID (1: Ethereum, 10: Optimism, 137: Polygon, 42161: Arbitrum, 42220: Celo, 56: BNB Chain, 43114: Avalanche, 8453: Base)"),
    tokenIn: z.string().describe("Input token address ('NATIVE' for native token like ETH)"),
    tokenOut: z.string().describe("Output token address ('NATIVE' for native token like ETH)"),
    amountIn: z.string().optional().describe("Exact input amount (required for exactIn trades)"),
    amountOut: z.string().optional().describe("Exact output amount (required for exactOut trades)"),
    tradeType: z.enum(["exactIn", "exactOut"]).default("exactIn").describe("Trade type: exactIn requires amountIn, exactOut requires amountOut"),
    slippageTolerance: z.number().optional().default(0.5).describe("Slippage tolerance in percentage"),
    deadline: z.number().optional().default(20).describe("Transaction deadline in minutes")
  },
  async ({ chainId, tokenIn, tokenOut, amountIn, amountOut, tradeType, slippageTolerance, deadline }) => {
    try {
      const { provider, router, config } = getChainContext(chainId);
      const wallet = new ethers.Wallet(WALLET_PRIVATE_KEY, provider);

      const isNativeIn = !tokenIn || tokenIn.toLowerCase() === "native";
      const isNativeOut = !tokenOut || tokenOut.toLowerCase() === "native";
      
      const tokenA = await createToken(chainId, isNativeIn ? config.weth : tokenIn, provider);
      const tokenB = await createToken(chainId, isNativeOut ? config.weth : tokenOut, provider);

      if (tradeType === "exactIn" && !amountIn) {
        throw new Error("amountIn is required for exactIn trades");
      }
      if (tradeType === "exactOut" && !amountOut) {
        throw new Error("amountOut is required for exactOut trades");
      }

      const amount = tradeType === "exactIn" ? amountIn : amountOut;
      const decimals = tradeType === "exactIn" ? tokenA.decimals : tokenB.decimals;
      const amountWei = parseUnits(amount, decimals).toString();
      
      const route = await router.route(
        CurrencyAmount.fromRawAmount(
          tradeType === "exactIn" ? tokenA : tokenB,
          amountWei
        ),
        tradeType === "exactIn" ? tokenB : tokenA,
        tradeType === "exactIn" ? TradeType.EXACT_INPUT : TradeType.EXACT_OUTPUT,
        {
          recipient: isNativeOut ? wallet.address : config.swapRouter,
          slippageTolerance: new Percent(Math.floor(slippageTolerance * 100), 10000),
          deadline: Math.floor(Date.now() / 1000) + (deadline * 60),
          type: SwapType.SWAP_ROUTER_02,
        }
      );

      if (!route) throw new Error("No route found");

      // Check balance before swap
      await checkBalance(provider, wallet, isNativeIn ? null : tokenA.address, isNativeIn);

      const swapRouter = new ethers.Contract(config.swapRouter, SwapRouterABI, wallet);
      const wethContract = new ethers.Contract(config.weth, WETHABI, wallet);

      // Approve token if not native input
      if (!isNativeIn) {
        const tokenContract = new ethers.Contract(tokenA.address, ERC20ABI, wallet);
        const approvalTx = await tokenContract.approve(config.swapRouter, ethers.constants.MaxUint256);
        await approvalTx.wait();
      }

      let tx;
      if (isNativeOut && tradeType === "exactOut") {
        // Execute swap to receive WETH
        tx = await wallet.sendTransaction({
          to: config.swapRouter,
          data: route.methodParameters.calldata,
          value: route.methodParameters.value,
          gasLimit: route.estimatedGasUsed.mul(12).div(10),
          gasPrice: (await provider.getGasPrice()).mul(2)
        });
        const receipt = await tx.wait();
        
        // Convert WETH to native token
        const wethAmount = route.trade.outputAmount.quotient;
        const withdrawTx = await wethContract.withdraw(wethAmount);
        await withdrawTx.wait();
      } else {
        // Execute swap directly, native input handled by SwapRouter via value
        tx = await wallet.sendTransaction({
          to: config.swapRouter,
          data: route.methodParameters.calldata,
          value: isNativeIn && tradeType === "exactIn" ? amountWei : route.methodParameters.value,
          gasLimit: route.estimatedGasUsed.mul(12).div(10),
          gasPrice: (await provider.getGasPrice()).mul(2)
        });
        await tx.wait();
      }

      const receipt = await tx.wait();

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            chainId,
            txHash: receipt.transactionHash,
            tradeType,
            amountIn: route.trade.inputAmount.toSignificant(6),
            outputAmount: route.trade.outputAmount.toSignificant(6),
            minimumReceived: route.trade.minimumAmountOut(new Percent(Math.floor(slippageTolerance * 100), 10000)).toSignificant(6),
            maximumInput: route.trade.maximumAmountIn(new Percent(Math.floor(slippageTolerance * 100), 10000)).toSignificant(6),
            fromToken: isNativeIn ? "NATIVE" : tokenIn,
            toToken: isNativeOut ? "NATIVE" : tokenOut,
            route: route.trade.swaps.map(swap => ({
              tokenIn: swap.inputAmount.currency.address,
              tokenOut: swap.outputAmount.currency.address,
              fee: swap.route.pools[0].fee
            })),
            gasUsed: receipt.gasUsed.toString()
          }, null, 2)
        }]
      };
    } catch (error) {
      throw new Error(`Swap failed: ${error.message}. Check wallet funds and network connection.`);
    }
  }
);

// Prompt: Generate swap suggestion with Smart Order Router
server.prompt(
  "suggestSwap",
  { 
    amount: z.string().describe("Amount to swap"),
    token: z.string().describe("Starting token address ('NATIVE' for native token like ETH)"),
    tradeType: z.enum(["exactIn", "exactOut"]).default("exactIn").describe("Trade type")
  },
  ({ amount, token, tradeType }) => ({
    messages: [{
      role: "user",
      content: {
        type: "text",
        text: `Suggest the best token swap for ${amount} at ${token} on Uniswap V3 using smart order routing. Consider liquidity, fees, and optimal multi-hop paths. Trade type: ${tradeType}.`
      }
    }]
  })
);

// Start the server without Infura check
async function startServer() {
  try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
  } catch (error) {
    console.error(`Failed to start server: ${error.message}`);
    process.exit(1);
  }
}

startServer();