import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import re

class WebSurf:
    """
    A web surfing class that searches the web and extracts both text and images.
    The main interface is the forward() method.
    """
    
    def __init__(self, user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def forward(self, query: str = "metamask integration ", num_results: int = 10, include_images: bool = True, include_text: bool = True) -> Dict[str, Any]:
        """
        Main search interface. Searches the web and returns structured results.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            include_images: Whether to extract images from results
            include_text: Whether to extract text content from results
            
        Returns:
            Dictionary containing:
            - success: bool
            - query: str
            - text_results: List[Dict] with url and content
            - image_results: List[Dict] with url and content (image URL)
            - total_results: int
        """
        try:
            text_results = []
            image_results = []
            
            # Perform search using DuckDuckGo HTML
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text results
            if include_text:
                result_divs = soup.find_all('div', class_='result', limit=num_results)
                for div in result_divs:
                    try:
                        link_tag = div.find('a', class_='result__a')
                        snippet_tag = div.find('a', class_='result__snippet')
                        
                        if link_tag:
                            url = link_tag.get('href', '')
                            title = link_tag.get_text(strip=True)
                            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                            
                            text_results.append({
                                'url': url,
                                'content': f"{title}\n{snippet}",
                                'title': title,
                                'snippet': snippet
                            })
                    except Exception as e:
                        continue
            
            # Extract image results
            if include_images:
                image_search_url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&iax=images&ia=images"
                try:
                    img_response = self.session.get(image_search_url, timeout=10)
                    img_soup = BeautifulSoup(img_response.text, 'html.parser')
                    
                    # Extract image URLs from various possible sources
                    img_tags = img_soup.find_all('img', limit=num_results)
                    for img in img_tags:
                        img_url = img.get('src') or img.get('data-src')
                        if img_url and img_url.startswith('http'):
                            image_results.append({
                                'url': img_url,
                                'content': img_url,
                                'alt': img.get('alt', '')
                            })
                except Exception as e:
                    pass
            
            return {
                'success': True,
                'query': query,
                'text_results': text_results,
                'image_results': image_results,
                'total_results': len(text_results) + len(image_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'text_results': [],
                'image_results': [],
                'total_results': 0,
                'error': str(e)
            }
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Alias for forward method"""
        return self.forward(query, **kwargs)
