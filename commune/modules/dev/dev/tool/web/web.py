import commune as c
import requests
import json
import os
import time
from typing import List, Dict, Union, Optional, Any, Set
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin, urlparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

print = c.print

class WebScraper:
    """
    An enhanced web scraping tool with depth crawling, better content extraction,
    and improved information retrieval capabilities.
    
    This tool provides functionality to:
    - Search the web using direct scraping of search engines
    - Crawl pages to a specified depth for comprehensive data extraction
    - Extract structured content from web pages
    - Handle JavaScript-rendered content
    - Cache results to minimize redundant requests
    - Extract specific data patterns (emails, phones, links, etc.)
    """
    
    def __init__(self, 
                 search_engine: str = 'bing',
                 cache_dir: str = '~/.commune/web_scraper_cache',
                 cache_expiry: int = 3600,  # 1 hour
                 use_selenium: bool = True,
                 headless: bool = True,
                 max_workers: int = 5,
                 request_delay: float = 0.5,
                 **kwargs):
        """
        Initialize the WebScraper tool.
        
        Args:
            search_engine: Search engine to use ('google', 'bing', 'duckduckgo')
            cache_dir: Directory to store cached results
            cache_expiry: Time in seconds before cache entries expire
            use_selenium: Whether to use Selenium for JavaScript-heavy sites
            headless: Whether to run browser in headless mode (Selenium only)
            max_workers: Maximum number of concurrent workers for parallel processing
            request_delay: Delay between requests to avoid rate limiting
            **kwargs: Additional configuration parameters
        """
        self.search_engine = search_engine.lower()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.cache_expiry = cache_expiry
        self.use_selenium = use_selenium
        self.headless = headless
        self.driver = None
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.visited_urls = set()
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Enhanced search engine configurations
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
        
        # Content extraction patterns
        self.extraction_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}',
            'social_media': {
                'twitter': r'(?:https?://)?(?:www\.)?twitter\.com/[a-zA-Z0-9_]+',
                'linkedin': r'(?:https?://)?(?:www\.)?linkedin\.com/(?:in|company)/[a-zA-Z0-9-]+',
                'facebook': r'(?:https?://)?(?:www\.)?facebook\.com/[a-zA-Z0-9.]+',
                'instagram': r'(?:https?://)?(?:www\.)?instagram\.com/[a-zA-Z0-9_.]+'
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
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize Chrome WebDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
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
                query: str = 'what is the weather today',
                num_results: int = 5,
                page_depth: int = 0,
                include_snippets: bool = True,
                include_links: bool = True,
                extract_patterns: Optional[List[str]] = None,
                filter_domains: Optional[List[str]] = None,
                safe_search: bool = True,
                use_cache: bool = False,
                cache_key: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Search the web for information with depth crawling capability.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            page_depth: How many levels deep to crawl from search results (0 = search only)
            include_snippets: Whether to include text snippets in results
            include_links: Whether to include links in results
            extract_patterns: List of patterns to extract ('email', 'phone', 'social_media')
            filter_domains: List of domains to include/exclude (e.g., ['wikipedia.org'])
            safe_search: Whether to enable safe search filtering
            use_cache: Whether to use cached results if available
            cache_key: Custom key for caching (defaults to query hash)
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing:
            - success: Whether the search was successful
            - results: List of search results with extracted content
            - context: Extracted context from results
            - query: Original search query
            - source: Search engine used
            - extracted_data: Any patterns extracted from pages
        """
        if verbose:
            c.print(f"Searching for: {query} (depth: {page_depth})", color="cyan")
        
        # Reset visited URLs for new search
        self.visited_urls = set()
        
        # Generate cache key if not provided
        if use_cache and not cache_key:
            cache_key = hashlib.md5(f"{self.search_engine}_{query}_{num_results}_{page_depth}".encode()).hexdigest()
        
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
                self._initialize_selenium()
                results = self._search_with_selenium(query, num_results, safe_search, filter_domains)
            else:
                results = self._search_with_requests(query, num_results, safe_search, filter_domains)
            
            # Process results with depth crawling if requested
            processed_results = []
            extracted_data = {'emails': set(), 'phones': set(), 'social_media': {}}
            
            if page_depth > 0:
                # Crawl pages to specified depth
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for result in results:
                        if 'link' in result and result['link']:
                            future = executor.submit(self._crawl_page, result['link'], page_depth, extract_patterns)
                            futures.append((future, result))
                    
                    for future, original_result in futures:
                        try:
                            crawl_data = future.result(timeout=30)
                            # Merge crawled data with original result
                            enhanced_result = {**original_result, **crawl_data}
                            processed_results.append(enhanced_result)
                            
                            # Aggregate extracted patterns
                            if 'extracted_patterns' in crawl_data:
                                for pattern_type, values in crawl_data['extracted_patterns'].items():
                                    if pattern_type == 'social_media':
                                        for platform, links in values.items():
                                            if platform not in extracted_data['social_media']:
                                                extracted_data['social_media'][platform] = set()
                                            extracted_data['social_media'][platform].update(links)
                                    else:
                                        extracted_data[pattern_type].update(values)
                        except Exception as e:
                            if verbose:
                                c.print(f"Error crawling page: {e}", color="yellow")
                            processed_results.append(original_result)
            else:
                # No depth crawling, just process search results
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
            
            # Convert sets to lists for JSON serialization
            serializable_extracted = {
                'emails': list(extracted_data['emails']),
                'phones': list(extracted_data['phones']),
                'social_media': {k: list(v) for k, v in extracted_data['social_media'].items()}
            }
            
            # Prepare response
            response = {
                "success": True,
                "results": processed_results,
                "context": context,
                "query": query,
                "source": self.search_engine,
                "page_depth": page_depth,
                "extracted_data": serializable_extracted if extract_patterns else None
            }
            
            # Cache the results if enabled
            if use_cache:
                self._save_to_cache(cache_key, response)
            
            if verbose:
                c.print(f"Found {len(processed_results)} results", color="green")
                if page_depth > 0:
                    c.print(f"Crawled to depth {page_depth}", color="blue")
                if extract_patterns and any(serializable_extracted.values()):
                    c.print(f"Extracted data: {serializable_extracted}", color="cyan")
            
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
    
    def _crawl_page(self, url: str, depth: int, extract_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Crawl a page to extract content and optionally follow links to specified depth.
        
        Args:
            url: URL to crawl
            depth: How many levels deep to crawl
            extract_patterns: Patterns to extract from the page
            
        Returns:
            Dictionary with extracted content and data
        """
        if url in self.visited_urls or depth < 0:
            return {}
        
        self.visited_urls.add(url)
        time.sleep(self.request_delay)  # Rate limiting
        
        try:
            if self.use_selenium and self.driver:
                return self._crawl_with_selenium(url, depth, extract_patterns)
            else:
                return self._crawl_with_requests(url, depth, extract_patterns)
        except Exception as e:
            print(f"Error crawling {url}: {e}", color="yellow")
            return {'error': str(e)}
    
    def _crawl_with_requests(self, url: str, depth: int, extract_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Crawl a page using requests library.
        """
        headers = {
            'User-Agent': self.engine_configs[self.search_engine]['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        content = self._extract_main_content(soup)
        
        # Extract patterns if requested
        extracted_patterns = {}
        if extract_patterns:
            text = soup.get_text()
            extracted_patterns = self._extract_patterns_from_text(text, extract_patterns)
        
        # Extract links for further crawling
        links = []
        if depth > 1:
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                if self._is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                    links.append(absolute_url)
        
        result = {
            'url': url,
            'content': content,
            'extracted_patterns': extracted_patterns,
            'links': links[:10]  # Limit links to prevent explosion
        }
        
        # Recursively crawl child pages if depth allows
        if depth > 1 and links:
            child_results = []
            with ThreadPoolExecutor(max_workers=min(3, self.max_workers)) as executor:
                futures = [executor.submit(self._crawl_page, link, depth - 1, extract_patterns) for link in links[:5]]
                for future in as_completed(futures, timeout=20):
                    try:
                        child_result = future.result()
                        if child_result:
                            child_results.append(child_result)
                    except Exception:
                        pass
            result['child_pages'] = child_results
        
        return result
    
    def _crawl_with_selenium(self, url: str, depth: int, extract_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Crawl a page using Selenium for JavaScript-rendered content.
        """
        self.driver.get(url)
        time.sleep(2)  # Allow JavaScript to render
        
        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # Extract main content
        content = self._extract_main_content(soup)
        
        # Extract patterns if requested
        extracted_patterns = {}
        if extract_patterns:
            text = soup.get_text()
            extracted_patterns = self._extract_patterns_from_text(text, extract_patterns)
        
        # Extract links for further crawling
        links = []
        if depth > 1:
            link_elements = self.driver.find_elements(By.TAG_NAME, 'a')
            for element in link_elements:
                try:
                    href = element.get_attribute('href')
                    if href and self._is_valid_url(href) and href not in self.visited_urls:
                        links.append(href)
                except Exception:
                    pass
        
        result = {
            'url': url,
            'content': content,
            'extracted_patterns': extracted_patterns,
            'links': links[:10]
        }
        
        # Recursively crawl child pages if depth allows
        if depth > 1 and links:
            child_results = []
            for link in links[:5]:  # Limit to prevent explosion
                child_result = self._crawl_page(link, depth - 1, extract_patterns)
                if child_result:
                    child_results.append(child_result)
            result['child_pages'] = child_results
        
        return result
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract the main content from a web page.
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        content = {
            'title': '',
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'tables': []
        }
        
        # Extract title
        title = soup.find('title')
        if title:
            content['title'] = title.get_text().strip()
        
        # Extract headings
        for i in range(1, 4):  # h1, h2, h3
            headings = soup.find_all(f'h{i}')
            for heading in headings:
                text = heading.get_text().strip()
                if text:
                    content['headings'].append({'level': i, 'text': text})
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 50:  # Filter out very short paragraphs
                content['paragraphs'].append(text)
        
        # Extract lists
        lists = soup.find_all(['ul', 'ol'])
        for lst in lists:
            items = [li.get_text().strip() for li in lst.find_all('li')]
            if items:
                content['lists'].append(items)
        
        # Extract tables (simplified)
        tables = soup.find_all('table')
        for table in tables[:3]:  # Limit to 3 tables
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            if rows:
                content['tables'].append(rows)
        
        return content
    
    def _extract_patterns_from_text(self, text: str, patterns: List[str]) -> Dict[str, Any]:
        """
        Extract specified patterns from text.
        """
        results = {}
        
        if 'email' in patterns:
            emails = re.findall(self.extraction_patterns['email'], text)
            results['emails'] = list(set(emails))
        
        if 'phone' in patterns:
            phones = re.findall(self.extraction_patterns['phone'], text)
            results['phones'] = list(set(phones))
        
        if 'social_media' in patterns:
            social_results = {}
            for platform, pattern in self.extraction_patterns['social_media'].items():
                matches = re.findall(pattern, text)
                if matches:
                    social_results[platform] = list(set(matches))
            results['social_media'] = social_results
        
        return results
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid for crawling.
        """
        try:
            parsed = urlparse(url)
            # Check if it's a valid HTTP/HTTPS URL
            if parsed.scheme not in ['http', 'https']:
                return False
            # Skip certain file types
            excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.dmg']
            if any(url.lower().endswith(ext) for ext in excluded_extensions):
                return False
            return True
        except Exception:
            return False
    
    def _search_with_selenium(self, query, num_results, safe_search, filter_domains):
        """
        Perform a search using Selenium WebDriver.
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
            
            # Add extracted content if available
            if 'content' in result and result['content']:
                content = result['content']
                if content.get('paragraphs'):
                    entry += f"Content preview: {content['paragraphs'][0][:200]}...\n"
            
            # Add extracted patterns if available
            if 'extracted_patterns' in result and result['extracted_patterns']:
                patterns = result['extracted_patterns']
                if patterns.get('emails'):
                    entry += f"Emails found: {', '.join(patterns['emails'][:3])}\n"
                if patterns.get('phones'):
                    entry += f"Phones found: {', '.join(patterns['phones'][:3])}\n"
            
            context.append(entry)
        
        return "\n".join(context)
    
    def _get_from_cache(self, key):
        """
        Retrieve results from cache if available and not expired.
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
                            page_depth: int = 1,
                            max_tokens: int = 500,
                            extract_patterns: Optional[List[str]] = None,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Search the web and generate a comprehensive summary of the results.
        
        Args:
            query: Search query
            model: Model to use for summarization (uses default if None)
            num_results: Number of search results to consider
            page_depth: Depth to crawl for more comprehensive data
            max_tokens: Maximum length of summary
            extract_patterns: Patterns to extract from pages
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing:
            - success: Whether the operation was successful
            - summary: Generated summary
            - query: Original search query
            - results: Raw search results
            - extracted_data: Any patterns extracted
        """
        # First, search the web with depth crawling
        search_results = self.forward(
            query=query,
            num_results=num_results,
            page_depth=page_depth,
            extract_patterns=extract_patterns,
            verbose=verbose
        )
        
        if not search_results["success"]:
            return search_results
        
        try:
            # Get the model for summarization
            llm = c.module('model.openrouter')()
            
            # Create enhanced prompt for summarization
            context = search_results["context"]
            extracted_info = ""
            if search_results.get("extracted_data"):
                extracted = search_results["extracted_data"]
                if extracted.get("emails"):
                    extracted_info += f"\nEmails found: {', '.join(extracted['emails'][:5])}"
                if extracted.get("phones"):
                    extracted_info += f"\nPhones found: {', '.join(extracted['phones'][:5])}"
                if extracted.get("social_media"):
                    for platform, links in extracted["social_media"].items():
                        if links:
                            extracted_info += f"\n{platform.capitalize()} profiles: {', '.join(links[:3])}"
            
            prompt = f"""
            Based on the following search results for the query "{query}", provide a comprehensive, 
            informative summary of the key information. Focus on factual information and 
            include the most important points from multiple sources. If specific data was extracted,
            incorporate it into your sum.
            
            SEARCH RESULTS:
            {context}
            
            EXTRACTED DATA:
            {extracted_info if extracted_info else "No specific patterns extracted."}
            
            COMPREHENSIVE SUMMARY:
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
                "results": search_results["results"],
                "extracted_data": search_results.get("extracted_data")
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
    
    def extract_structured_data(self,
                               url: str,
                               data_schema: Dict[str, str],
                               verbose: bool = True) -> Dict[str, Any]:
        """
        Extract structured data from a specific URL based on a schema.
        
        Args:
            url: URL to extract data from
            data_schema: Dictionary defining what to extract
                        e.g., {'price': 'CSS selector or description',
                               'title': 'h1.product-title',
                               'reviews': 'div.review-count'}
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with extracted structured data
        """
        if verbose:
            c.print(f"Extracting structured data from: {url}", color="cyan")
        
        try:
            # Crawl the page
            page_data = self._crawl_page(url, depth=0)
            
            if 'error' in page_data:
                return {
                    "success": False,
                    "error": page_data['error'],
                    "url": url
                }
            
            # Extract based on schema
            extracted = {}
            soup = None
            
            # Get the page content for extraction
            if self.use_selenium and self.driver:
                self.driver.get(url)
                time.sleep(2)
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            else:
                headers = {
                    'User-Agent': self.engine_configs[self.search_engine]['user_agent']
                }
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract based on schema
            for field, selector in data_schema.items():
                try:
                    # If selector looks like a CSS selector
                    if '.' in selector or '#' in selector or ' ' in selector:
                        elements = soup.select(selector)
                        if elements:
                            if len(elements) == 1:
                                extracted[field] = elements[0].get_text().strip()
                            else:
                                extracted[field] = [elem.get_text().strip() for elem in elements]
                    else:
                        # Try to find by text content or tag name
                        element = soup.find(text=re.compile(selector, re.I))
                        if element:
                            extracted[field] = element.parent.get_text().strip()
                except Exception as e:
                    if verbose:
                        c.print(f"Failed to extract {field}: {e}", color="yellow")
                    extracted[field] = None
            
            if verbose:
                c.print(f"Successfully extracted: {list(extracted.keys())}", color="green")
            
            return {
                "success": True,
                "url": url,
                "extracted_data": extracted,
                "content": page_data.get('content', {})
            }
            
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            if verbose:
                c.print(error_msg, color="red")
            
            return {
                "success": False,
                "error": error_msg,
                "url": url
            }

# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = WebScraper(use_selenium=True, headless=True)
    
    # Example 1: Basic search with depth crawling
    print("\n=== Example 1: Search with depth crawling ===")
    results = scraper.forward(
        query="artificial intelligence startups 2024",
        num_results=3,
        page_depth=1,  # Crawl one level deep
        extract_patterns=['email', 'social_media'],
        verbose=True
    )
    print(f"\nExtracted emails: {results.get('extracted_data', {}).get('emails', [])}")
    
    # Example 2: Search and summarize
    print("\n=== Example 2: Search and summarize ===")
    summary_results = scraper.search_and_summarize(
        query="latest developments in quantum computing",
        num_results=5,
        page_depth=1,
        verbose=True
    )
    
    # Example 3: Extract structured data from a specific page
    print("\n=== Example 3: Extract structured data ===")
    structured_data = scraper.extract_structured_data(
        url="https://example.com/product",
        data_schema={
            'title': 'h1',
            'price': '.price',
            'description': '.product-description',
            'reviews': '.review-count'
        },
        verbose=True
    )
