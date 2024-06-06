import requests
import csv

# Function to fetch SOL transaction data for a specific address
def fetch_transactions(address):
    url = f"https://api.mainnet-beta.solana.com/v1/account/transactions/{address}"
    response = requests.get(url)
    data = response.json()
    return data

# Function to convert SOL amount to USD
def sol_to_usd(amount_in_sol):
    sol_price_response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd")
    sol_price_data = sol_price_response.json()
    sol_price_usd = sol_price_data['solana']['usd']
    # Convert SOL amount to USD
    amount_in_usd = amount_in_sol * sol_price_usd
    return amount_in_usd

# Address to fetch transactions for
address = "5P3mxkzywBnLWPXiqDiuFSeJVmTugZQAaWz17jPVyjwx"

# Fetch transactions
transactions = fetch_transactions(address)

# Write data to CSV file
with open('transactions.csv', 'w', newline='') as csvfile:
    fieldnames = ['Sender Address', 'Amount (SOL)', 'Amount (USD)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for transaction in transactions:
        sender_address = transaction['sender']
        amount_sol = transaction['amount']
        amount_usd = sol_to_usd(amount_sol)
        writer.writerow({'Sender Address': sender_address, 'Amount (SOL)': amount_sol, 'Amount (USD)': amount_usd})

print("CSV file generated successfully.")
