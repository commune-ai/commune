from raydium.cpmm import buy

if __name__ == "__main__":
    pair_address = ""
    sol_in = .1
    slippage = 5
    buy(pair_address, sol_in, slippage)