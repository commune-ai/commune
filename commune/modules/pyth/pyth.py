import requests
import sseclient

class PythPriceFetcher:
    BASE_URL = 'https://hermes.pyth.network'

    def __init__(self):
        self.session = requests.Session()

    def get_latest_price(self, price_ids):
        endpoint = f"{self.BASE_URL}/v2/updates/price/latest"
        params = [('ids[]', price_id) for price_id in price_ids]
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()['parsed']

    def stream_price_updates(self, price_ids, callback):
        url = f"{self.BASE_URL}/v2/updates/price/stream"
        params = '&'.join([f'ids[]={price_id}' for price_id in price_ids])
        full_url = f"{url}?{params}"

        with self.session.get(full_url, stream=True) as response:
            response.raise_for_status()
            client = sseclient.SSEClient(response)

            for event in client.events():
                data = event.data
                callback(data)

    # Example usage:
    def test(self):
        fetcher = PythPriceFetcher()
        
        btc_eth_price_ids = [
            "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43", # BTC/USD
            "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace"  # ETH/USD
        ]

        latest_prices = fetcher.get_latest_price(btc_eth_price_ids)
        print("Latest Prices:", latest_prices)

        def print_stream_update(data):
            print("Stream update:", data)

        fetcher.stream_price_updates([btc_eth_price_ids[0]], print_stream_update)
