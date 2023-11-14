import requests

class SearchService:
    def __init__(self, search_engine_url):
        self.search_engine_url = search_engine_url

    def perform_search(self, query):
        # can call an external search engine API here
        # For example, let's say we're using a hypothetical API that takes a 'q' parameter for the query
        response = requests.get(self.search_engine_url, params={'q': query})
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
