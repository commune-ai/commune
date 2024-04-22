import argparse
import commune as c

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Transfer all balance to a specific address.')
parser.add_argument('--address', type=str, help='The address to transfer to.')

parser.add_argument('--min', type=float, default=0.5, help='Min stake amount to left. Default is 0.5.')

# Parse command-line arguments
args = parser.parse_args()

wallet_addresses = c.keys()
print(wallet_addresses)

for address in wallet_addresses:
    try:
        # Unstake all balance from the wallet
        staked = c.get_stake(address, update=True)
        print(f"Staked amount is {staked}")
        if (staked > 0):
            c.unstake(key=address)
        
        # Transfer all balance to the specific address
        balance = c.get_balance(address)
        print(f"Total balance is {balance}")

        if (balance > args.min):
            c.transfer(args.address, balance - args.min, address)

    except Exception as e:
        # Code that will run if the exception is raised
        print(f"An error occurred: {e}, Address: {address}")
