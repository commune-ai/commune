import commune as c
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import random

class Surfy:
    """
    A module that can surf the web and fetch context based on a query without using an API key.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Surfy module with configurable parameters.
        Args:
            **kwargs: Arbitrary keyword arguments to configure the instance
        """
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        self.search_engines = [
            {'name': 'Google', 'url': 'https://www.google.com/search?q={}&num={}', 'selector': 'div.g'},
            {'name': 'DuckDuckGo', 'url': 'https://html.duckduckgo.com/html/?q={}', 'selector': 'div.result'}
        ]
        
    def _get_random_user_agent(self):
        """
        Return a random user agent from the list to avoid detection.
        """
        return random.choice(self.user_agents)
    
    def _get_search_engine(self):
        """
        Return a random search engine configuration.
        """
        return random.choice(self.search_engines)
    
    def search(self, query, num_results=5):
        """
        Search the web for the given query and return results.
        
        Args:
            query (str): The search query
            num_results (int): Number of results to return
            
        Returns:
            list: A list of search results with titles, snippets, and URLs
        """
        engine = self._get_search_engine()
        headers = {'User-Agent': self._get_random_user_agent()}
        
        try:
            # Format the search URL with the query
            search_url = engine['url'].format(quote_plus(query), num_results)
            
            # Send the request
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = []
            if engine['name'] == 'Google':
                for result in soup.select(engine['selector'])[:num_results]:
                    title_elem = result.select_one('h3')
                    link_elem = result.select_one('a')
                    snippet_elem = result.select_one('div.VwiC3b')
                    
                    if title_elem and link_elem and snippet_elem:
                        title = title_elem.get_text()
                        link = link_elem.get('href')
                        if link.startswith('/url?q='):
                            link = link.split('/url?q=')[1].split('&')[0]
                        snippet = snippet_elem.get_text()
                        
                        results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet
                        })
            elif engine['name'] == 'DuckDuckGo':
                for result in soup.select(engine['selector'])[:num_results]:
                    title_elem = result.select_one('h2')
                    link_elem = result.select_one('a.result__a')
                    snippet_elem = result.select_one('a.result__snippet')
                    
                    if title_elem and link_elem and snippet_elem:
                        title = title_elem.get_text()
                        link = link_elem.get('href')
                        snippet = snippet_elem.get_text()
                        
                        results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet
                        })
            
            return results
        
        except Exception as e:
            c.print(f'Error searching with {engine["name"]}: {str(e)}', color='red')
            # If one search engine fails, try with another
            if len(self.search_engines) > 1:
                self.search_engines = [se for se in self.search_engines if se != engine]
                return self.search(query, num_results)
            return []
    
    def fetch_page_content(self, url):
        """
        Fetch and parse content from a specific URL.
        
        Args:
            url (str): The URL to fetch content from
            
        Returns:
            str: The main text content of the page
        """
        headers = {'User-Agent': self._get_random_user_agent()}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        
        except Exception as e:
            c.print(f'Error fetching content from {url}: {str(e)}', color='red')
            return ""
    
    def forward(self, query, *extra_query,  num_results=5, fetch_content=False, max_content_length=5000):
        """
        Search the web for information based on a query and return relevant context.
        
        Args:
            query (str): The search query
            num_results (int): Number of search results to consider
            fetch_content (bool): Whether to fetch full content from result URLs
            max_content_length (int): Maximum length of content to return per result
            
        Returns:
            dict: Search results and extracted context
        """
        # Search for results
        results = self.search(query, num_results)
        query = query + " " + " ".join(extra_query)
        
        context = []
        
        # Extract snippets from search results
        for result in results:
            context.append(f"Title: {result['title']}")
            context.append(f"URL: {result['url']}")
            context.append(f"Snippet: {result['snippet']}")
            context.append("---")
        
        # Optionally fetch full content from URLs
        if fetch_content and results:
            for i, result in enumerate(results):
                if i >= 3:  # Limit to first 3 results to avoid too many requests
                    break
                    
                content = self.fetch_page_content(result['url'])
                if content:
                    # Truncate content if it's too long
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    
                    context.append(f"\nFull content from {result['title']}:")
                    context.append(content)
                    context.append("---")
        
        return {
            'query': query,
            'results': results,
            'context': '\n'.join(context)
        }
