import requests
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)

# Mock API URL for fetching token prices (replace with real API URL)
TOKEN_PRICE_API_URL = "https://api.coingecko.com/api/v3/simple/price"

# Token list for price analysis
TOKENS = ["BTC", "ETH", "XRP"]

# Store historical prices
historical_prices = {token: [] for token in TOKENS}

def fetch_token_price(token):
    try:
        response = requests.get(f"{TOKEN_PRICE_API_URL}/{token}")
        if response.status_code == 200:
            return response.json().get('price')
        else:
            logging.error("Failed to fetch price for %s. Status Code: %d", token, response.status_code)
    except Exception as e:
        logging.error("Error fetching price for %s: %s", token, e)
    return None

def analyze_prices(token, current_price):
    if not historical_prices[token]:
        logging.info("No historical data for analysis for %s", token)
        return

    last_price = historical_prices[token][-1]
    if current_price > last_price:
        logging.info("Price uptrend for %s", token)
    elif current_price < last_price:
        logging.info("Price downtrend for %s", token)
    else:
        logging.info("Price stable for %s", token)

def update_price_history(token, price):
    historical_prices[token].append(price)
    logging.info("Updated historical prices for %s", token)

def run_price_check():
    for token in TOKENS:
        current_price = fetch_token_price(token)
        if current_price is not None:
            logging.info("Current price of %s: %s", token, current_price)
            analyze_prices(token, current_price)
            update_price_history(token, current_price)
        else:
            logging.error("Failed to get current price for %s", token)

# Run the price check function
run_price_check()
