import argparse
import commune as c

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Transfer all balance to a specific address.')
parser.add_argument('--address', type=str, help='The address to transfer to.')
parser.add_argument('--min', type=float, default=0.5, help='Min stake amount to left. Default is 0.5.')
parser.add_argument('--timeout', type=float, default=10, help='The timeout for the task. Default is 10.')

# Parse command-line arguments
args = parser.parse_args()

def unstake_fn(address):
    # Unstake all balance from the wallet
    staked = c.get_stake(address)
    print(f"Staked amount is {staked}")
    if (staked > 0):
        c.unstake(staked, address)
    
    # Transfer all balance to the specific address
    balance = c.get_balance(address)
    print(f"Total balance is {balance}")

    if (balance > args.min):
        c.transfer(args.address, balance - args.min, address)


for address in c.keys():
    c.print(f"Unstaking and transferring all balance from {address} to {args.address}...")
    futures = c.submit(unstake_fn, address, timeout=args.timeout)

for future in c.as_completed(futures, timeout=args.timeout): 
    result = future.result()
    c.print(result)    

