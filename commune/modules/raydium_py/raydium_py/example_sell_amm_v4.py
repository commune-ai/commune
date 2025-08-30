from raydium.amm_v4 import sell

if __name__ == "__main__":
    pair_address = ""
    percentage = 100
    slippage = 5
    sell(pair_address, percentage, slippage)