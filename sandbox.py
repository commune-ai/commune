import requests
import commune as c

class Web(c.Module):
    @staticmethod
    def get_text_from_url(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None
        
    def test(self):
        print("test"


url = "https://cryptomarketpool.com/use-web3-py-in-python-to-call-uniswap/"

web_util = Web()
text = web_util.get_text_from_url(url)

if text:
    c.print(text)
else:
    c.print("Failed to retrieve the text from the URL.")
