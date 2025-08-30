import commune as c
import requests
import json
import os
import time
from typing import List, Dict, Union, Optional, Any
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

print = c.print

class WebScraper:
    """
    A tool for scraping the web and retrieving relevant information based on queries
    without relying on search engine APIs.
    
    This tool provides functionality to:
    - Search the web using direct scraping of search engines
    - Extract and format relevant information from search results
    - Return context that can be used for further processing
    - Cache results to minimize redundant requests
    """
    
    def __init__(self, 
                 search_engine: str = 'bing',
                 cache_dir: str = '~/.commune/web_scraper_cache',
                 cache_expiry: int = 3600,  # 1 hour
                 use_selenium: bool = True,
                 headless: bool = True,
                 **kwargs):
        """
        Initialize the WebScraper tool.
        
        Args:
            search_engine: Search engine to use ('google', 'bing', 'duckduckgo')
            cache_dir: Directory to store cached results
            cache_expiry: Time in seconds before cache entries expire
            use_selenium: Whether to use Selenium for JavaScript-heavy sites
            headless: Whether to run browser in headless mode (Selenium only)
            **kwargs: Additional configuration parameters
        """
        self.search_engine = search_engine.lower()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.cache_expiry = cache_expiry
        self.use_selenium = use_selenium
        self.headless = headless
        self.driver = None
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up search engine configurations
        self.engine_configs = {
            'google': {
                'search_url': 'https://www.google.com/search?q={query}&num={num_results}',
                'result_selector': 'div.g',
                'title_selector': 'h3',
                'link_selector': 'a',
                'snippet_selector': 'div.VwiC3b',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            },
            'bing': {
                'search_url': 'https://www.bing.com/search?q={query}&count={num_results}',
                'result_selector': '.b_algo',
                'title_selector': 'h2',
                'link_selector': 'a',
                'snippet_selector': '.b_caption p',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            },
            'duckduckgo': {
                'search_url': 'https://html.duckduckgo.com/html/?q={query}',
                'result_selector': '.result',
                'title_selector': '.result__title',
                'link_selector': '.result__url',
                'snippet_selector': '.result__snippet',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            }
        }
        
        # Validate configuration
        if self.search_engine not in self.engine_configs:
            raise ValueError(f"Unsupported search engine: {self.search_engine}. Supported engines: {list(self.engine_configs.keys())}")
    
    def _initialize_selenium(self):
        """
        Initialize Selenium WebDriver if not already initialized.
        """
        if self.driver is not None:
            return
        
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument(f"user-agent={self.engine_configs[self.search_engine]['user_agent']}")
            
            # Initialize Chrome WebDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            print("Selenium WebDriver initialized successfully", color="green")
        except Exception as e:
            print(f"Failed to initialize Selenium WebDriver: {e}", color="red")
            self.use_selenium = False
    
    def _close_selenium(self):
        """
        Close Selenium WebDriver if it's open.
        """
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            finally:
                self.driver = None
    
    def forward(self,
                query: str,
                num_results: int = 5,
                include_snippets: bool = True,
                include_links: bool = True,
                filter_domains: Optional[List[str]] = None,
                safe_search: bool = True,
                use_cache: bool = False,
                cache_key: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Search the web for information related to the query.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            include_snippets: Whether to include text snippets in results
            include_links: Whether to include links in results
            filter_domains: List of domains to include/exclude (e.g., ['wikipedia.org'])
            safe_search: Whether to enable safe search filtering
            use_cache: Whether to use cached results if available
            cache_key: Custom key for caching (defaults to query hash)
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing:
            - success: Whether the search was successful
            - results: List of search results
            - context: Extracted context from results
            - query: Original search query
            - source: Search engine used
        """
        if verbose:
            c.print(f"Searching for: {query}", color="cyan")
        
        # Generate cache key if not provided
        if use_cache and not cache_key:
            cache_key = f"{self.search_engine}_{hash(query)}_{num_results}"
        
        # Check cache first if enabled
        if use_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                if verbose:
                    c.print(f"Retrieved results from cache", color="green")
                return cached_result
        
        try:
            # Perform search based on engine
            if self.use_selenium:
                # Initialize Selenium if needed
                self._initialize_selenium()
                results = self._search_with_selenium(query, num_results, safe_search, filter_domains)
            else:
                results = self._search_with_requests(query, num_results, safe_search, filter_domains)
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = {}
                
                if include_links and 'link' in result:
                    processed_result['url'] = result['link']
                
                if 'title' in result:
                    processed_result['title'] = result['title']
                
                if include_snippets and 'snippet' in result:
                    processed_result['snippet'] = result['snippet']
                
                processed_results.append(processed_result)
            
            # Extract context from results
            context = self._extract_context(processed_results)
            
            # Prepare response
            response = {
                "success": True,
                "results": processed_results,
                "context": context,
                "query": query,
                "source": self.search_engine
            }
            
            # Cache the results if enabled
            if use_cache:
                self._save_to_cache(cache_key, response)
            
            if verbose:
                c.print(f"Found {len(processed_results)} results", color="green")
                if len(processed_results) > 0:
                    c.print("Top result:", color="blue")
                    c.print(f"Title: {processed_results[0].get('title', 'N/A')}")
                    if include_snippets:
                        c.print(f"Snippet: {processed_results[0].get('snippet', 'N/A')}")
            
            return response
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            if verbose:
                c.print(error_msg, color="red")
            
            return {
                "success": False,
                "error": error_msg,
                "query": query,
                "source": self.search_engine
            }
        finally:
            # Close Selenium if we're not caching the driver
            if self.use_selenium and not getattr(self, 'keep_driver_alive', False):
                self._close_selenium()
    
    def _search_with_selenium(self, query, num_results, safe_search, filter_domains):
        """
        Perform a search using Selenium WebDriver.
        
        Args:
            query: Search query
            num_results: Number of results to return
            safe_search: Whether to enable safe search
            filter_domains: Domains to filter
            
        Returns:
            List of search results
        """
        config = self.engine_configs[self.search_engine]
        
        # Add domain filtering if specified
        search_query = query
        if filter_domains:
            site_query = ' OR '.join([f'site:{domain}' for domain in filter_domains])
            search_query = f"({search_query}) {site_query}"
        
        # Add safe search if enabled
        if safe_search and self.search_engine == 'google':
            search_query += " &safe=active"
        
        # Format the search URL
        search_url = config['search_url'].format(query=search_query.replace(' ', '+'), num_results=num_results)
        
        # Navigate to the search page
        self.driver.get(search_url)
        time.sleep(2)  # Allow page to load
        
        # Extract search results
        results = []
        result_elements = self.driver.find_elements(By.CSS_SELECTOR, config['result_selector'])
        
        for element in result_elements[:num_results]:
            try:
                title_element = element.find_element(By.CSS_SELECTOR, config['title_selector'])
                title = title_element.text.strip()
                
                link_element = title_element.find_element(By.CSS_SELECTOR, config['link_selector'])
                link = link_element.get_attribute('href')
                
                snippet = ""
                try:
                    snippet_element = element.find_element(By.CSS_SELECTOR, config['snippet_selector'])
                    snippet = snippet_element.text.strip()
                except Exception:
                    pass  # Snippet might not be available
                
                results.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet
                })
                
                if len(results) >= num_results:
                    break
            except Exception as e:
                print(f"Error extracting result: {e}", color="yellow")
        
        return results
    
    def _search_with_requests(self, query, num_results, safe_search, filter_domains):
        """
        Perform a search using requests and BeautifulSoup.
        
        Args:
            query: Search query
            num_results: Number of results to return
            safe_search: Whether to enable safe search
            filter_domains: Domains to filter
            
        Returns:
            List of search results
        """
        config = self.engine_configs[self.search_engine]
        
        # Add domain filtering if specified
        search_query = query
        if filter_domains:
            site_query = ' OR '.join([f'site:{domain}' for domain in filter_domains])
            search_query = f"({search_query}) {site_query}"
        
        # Add safe search if enabled
        if safe_search and self.search_engine == 'google':
            search_query += " &safe=active"
        
        # Format the search URL
        search_url = config['search_url'].format(query=search_query.replace(' ', '+'), num_results=num_results)
        
        # Set up headers to mimic a browser
        headers = {
            'User-Agent': config['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': f"https://www.{self.search_engine}.com/",
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Send request
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        results = []
        result_elements = soup.select(config['result_selector'])
        
        for element in result_elements[:num_results]:
            try:
                title_element = element.select_one(config['title_selector'])
                title = title_element.get_text().strip() if title_element else 'No title'
                
                link_element = title_element.select_one(config['link_selector']) if title_element else None
                link = link_element.get('href') if link_element else ''
                
                # For Google, links might be relative or in a special format
                if self.search_engine == 'google' and link and not link.startswith('http'):
                    if link.startswith('/url?q='):
                        link = link.split('/url?q=')[1].split('&')[0]
                
                snippet_element = element.select_one(config['snippet_selector'])
                snippet = snippet_element.get_text().strip() if snippet_element else ''
                
                results.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet
                })
                
                if len(results) >= num_results:
                    break
            except Exception as e:
                print(f"Error extracting result: {e}", color="yellow")
        
        return results
    
    def _extract_context(self, results):
        """
        Extract and format context from search results.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context = []
        for i, result in enumerate(results):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description available')
            url = result.get('url', '')
            
            entry = f"[{i+1}] {title}\n"
            if snippet:
                entry += f"{snippet}\n"
            if url:
                entry += f"Source: {url}\n"
            
            context.append(entry)
        
        return "\n".join(context)
    
    def _get_from_cache(self, key):
        """
        Retrieve results from cache if available and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached results or None if not found/expired
        """
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is expired
        if self.cache_expiry > 0:
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age > self.cache_expiry:
                return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, key, data):
        """
        Save results to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            c.print(f"Failed to save to cache: {e}", color="yellow")
    
    def search_and_summarize(self,
                            query: str,
                            model: Optional[str] = None,
                            num_results: int = 5,
                            max_tokens: int = 300,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Search the web and generate a concise summary of the results.
        
        Args:
            query: Search query
            model: Model to use for summarization (uses default if None)
            num_results: Number of search results to consider
            max_tokens: Maximum length of summary
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing:
            - success: Whether the operation was successful
            - summary: Generated summary
            - query: Original search query
            - results: Raw search results
        """
        # First, search the web
        search_results = self.forward(
            query=query,
            num_results=num_results,
            verbose=verbose
        )
        
        if not search_results["success"]:
            return search_results
        
        try:
            # Get the model for summarization
            llm = c.module('dev.model.openrouter')()
            
            # Create prompt for summarization
            context = search_results["context"]
            prompt = f"""
            Based on the following search results for the query "{query}", provide a concise, 
            informative summary of the key information. Focus on factual information and 
            include the most important points from multiple sources if available.
            
            SEARCH RESULTS:
            {context}
            
            SUMMARY:
            """
            
            # Generate summary
            summary = llm.forward(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens
            )
            
            if verbose:
                c.print("Generated summary:", color="green")
                c.print(summary)
            
            return {
                "success": True,
                "summary": summary,
                "query": query,
                "results": search_results["results"]
            }
            
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            if verbose:
                c.print(error_msg, color="red")
            
            return {
                "success": False,
                "error": error_msg,
                "query": query,
                "results": search_results["results"]
            }

# Example usage
if __name__ == "__main__":
    scraper = WebScraper(use_selenium=True, headless=True)
    results = scraper.forward("latest AI developments", num_results=5)
    print(results["context"])
