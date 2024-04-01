# Project - Web Module for Commune

**Overview:**
The project is a Web module for Commune, which help to perform web scraping, making http requests, fetching components from HTML, and processing the response.

## Requirements
- Python 3.6 and above
- Commune
- Requests
- BeautifulSoup
- AsyncIO
- aiohttp

## Usage

Create an instance of the `Web` class to use its methods.

**Methods:**

- `async_request`: This async method sends a request to a URL and return the response status and its text content if successful.
- `request`: This method sends a request according to the mode specified. The `mode` can be `request` or `asyncio`.
- `html`: This method fetches the HTML context of a page using the request method.
- `get_text`: This method get text from `p` tag of a specified page.
- `get_components`: This method fetches the components from HTML tags as passed in argument from a specified page(if successful).
- `rget`: This method performs a get request on a specified URL.
- `rpost`: This method performs a post request on a specified URL.
- `google_search`: Fetches search results from Google for a specified keyword.
- 'bing_search' : Fetches search results from Bing for a specified keyword.
- `yahoo_search`: Fetches search results from Yahoo for a specified keyword.
- `webpage`: Fetches the webpage from a specified URL.
- `soup`: Converts the HTML content to a BeautifulSoup object for easy manipulation and extraction.
- 'find': Finds and returns the first tag that match the argument tag in the soup.
- `install`: Installs the required dependencies.
- `url2text`: Extracts images and text from a specified URL and returns the data in a JSON format.

**Running the code:**

This module can be run directly after cloning the repository by running:

```bash
python web.py
```

## Disclaimer:
This module is built for educational purposes only, do not use it for illegal activity or production systems without proper testing. Please use it responsibly and ethically.