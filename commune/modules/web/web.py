
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import hashlib
from typing import Dict, List, Set
import logging
import commune as c

class Web:
    endpoints = ["search", "crawl"]
    
    def text(self, url: str = '') -> str:
        # 1. Fetch the page
        response = requests.get(url)
        html = response.text  # or response.content

        # 2. Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # 3. Extract just the text (without HTML tags)
        page_text = soup.get_text(separator="\n", strip=True)
        return page_text

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to the same domain"""
        try:
            parsed_base = urlparse(self.url)
            parsed_url = urlparse(url)
            return parsed_base.netloc == parsed_url.netloc
        except Exception:
            return False

    # def search(self, query:str='twitter', source:str='desktop') -> str:

    engine2url = { 
            'brave': 'https://search.brave.com/search?q={query}',
            'google': 'https://www.google.com/search?q={query}',
            'bing': 'https://www.bing.com/search?q={query}',
            'yahoo': 'https://search.yahoo.com/search?p={query}',
            'duckduckgo': 'https://duckduckgo.com/?q{query}',
            "sybil": "https://sybil.com/search?q={query}"
        }
    
    engines = list(engine2url.keys())

    def ask(self, *args, **kwargs):
        text = ' '.join(list(map(str, args)))
        context =  self.search(query=text, **kwargs)
        prompt = f"""
        QUERY
        {text}
        CONTEXT
        {context}
        """
        return c.generate(prompt, stream=1)
        

    def search(self, 
               query:str='twitter',  
               engine="all") -> str:
        '''
        Searches the query on the source
        '''

        if engine in self.engine2url:
            url = self.engine2url[engine].format(query=query)
        elif engine == 'all':
            results = {}
            future2engine = {}
            for engine in self.engine2url:
                url = self.engine2url[engine].format(query=query)
                future = c.submit(self.content, [url], timeout=10)
                future2engine[future] = engine

            for f in c.as_completed(future2engine):
                engine = future2engine[f]
                url = f.result()
                print(f'{engine} --> {url}')
                results[engine] = url
            return results
        else:
            raise ValueError(f'Engine {engine} not supported')
        
        return self.content(url)

    def content(self, url: str="https://search.brave.com/search?q=twitter") -> Dict:
        """
        Fetch and extract content from a webpage
        Returns a dictionary containing text content and image URLs
        """
        url = url or self.url
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            result_text = []
            for text in soup.stripped_strings:
                if len(text.strip()) > 0:
                    result_text.append(text.strip())

            # Extract images
            images = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    absolute_url = urljoin(url, src)
                    alt_text = img.get('alt', '')
                    images.append({
                        'url': absolute_url,
                        'alt_text': alt_text
                    })
    
            # Extract links for further crawling
            links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    if self.is_valid_url(absolute_url):
                        links.append(absolute_url)

            return {
                'url': url,
                'text': result_text,
                'images': images,
                'links': links
            }

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            return None

    def save_content(self, content: Dict, output_dir: str = 'crawled_data'):
        """Save the crawled content to files"""
        if not content:
            return

        # Create hash of URL for unique filename
        url_hash = hashlib.md5(content['url'].encode()).hexdigest()[:10]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save text content
        text_file = os.path.join(output_dir, f'{url_hash}_content.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"URL: {content['url']}\n\n")
            f.write("TEXT CONTENT:\n")
            for text in content['text']:
                f.write(f"{text}\n")
            
            f.write("\nIMAGES:\n")
            for img in content['images']:
                f.write(f"URL: {img['url']}\nAlt Text: {img['alt_text']}\n\n")

    def crawl(self, save_output: bool = True, output_dir: str = 'crawled_data'):
        """
        Main crawling method that processes pages up to max_pages
        """
        pages_to_visit = [self.url]
        crawled_content = []

        while pages_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = pages_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue

            self.logger.info(f"Crawling: {current_url}")
            content = self.content(current_url)
            
            if content:
                self.visited_urls.add(current_url)
                crawled_content.append(content)
                
                if save_output:
                    self.save_content(content, output_dir)
                
                # Add new links to visit
                pages_to_visit.extend([
                    url for url in content['links']
                    if url not in self.visited_urls and url not in pages_to_visit
                ])

        return crawled_content