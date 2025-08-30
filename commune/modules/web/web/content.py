import commune as c
from typing import Dict, Any, Optional

class WebContent:
    def __init__(self, content: str = ""):
        self.content = content
        self.web = c.mod('web')()

    def forward(self, 
                url: str,
                expand: bool = False,
                extract_patterns: Optional[list] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Get the content of a web page with various options.
        
        Args:
            url: URL to fetch content from
            expand: Whether to fetch full content from linked pages
            extract_patterns: List of patterns to extract ('email', 'phone', 'social_media')
            **kwargs: Additional arguments passed to web.content()
            
        Returns:
            Dictionary containing:
            - url: The URL that was fetched
            - text: List of text content from the page
            - images: List of images with URLs and alt text
            - links: List of links found on the page
            - expanded_content: Full content from linked pages (if expand=True)
            - extracted_patterns: Any patterns extracted (if extract_patterns provided)
        """
        # Use the web module's content method to fetch the page
        result = self.web.content(url=url, expand=expand)
        
        # If extract_patterns is provided, extract the patterns from the content
        if extract_patterns and result:
            # Combine all text for pattern extraction
            all_text = ' '.join(result.get('text', []))
            
            # If expanded content exists, add it to the text
            if expand and 'expanded_content' in result:
                for link, page_data in result['expanded_content'].items():
                    if page_data and 'full_text' in page_data:
                        all_text += ' ' + page_data['full_text']
            
            # Extract patterns using the web module's method
            extracted = self.web._extract_patterns(all_text, extract_patterns)
            result['extracted_patterns'] = extracted
        
        # Format the result as a string representation if needed
        if self.content:
            # If content was provided in init, use it as a template
            return f"WebContent(url={url}, content={self.content}, fetched_content={result})"
        
        return result
    
    def search(self, 
               query: str,
               engine: str = "google",
               expand: bool = False,
               **kwargs) -> Dict[str, Any]:
        """
        Search the web using the specified search engine.
        
        Args:
            query: Search query
            engine: Search engine to use (google, bing, yahoo, duckduckgo, brave, all)
            expand: Whether to expand content from search results
            **kwargs: Additional arguments passed to web.search()
            
        Returns:
            Search results dictionary
        """
        return self.web.search(query=query, engine=engine, expand=expand, **kwargs)
    
    def ask(self, *args, **kwargs) -> str:
        """
        Ask a question and get an AI-generated answer based on web search results.
        
        Args:
            *args: Question components to be joined
            **kwargs: Additional arguments passed to web.ask()
            
        Returns:
            AI-generated answer based on web search context
        """
        return self.web.ask(*args, **kwargs)
    
    def crawl(self, 
              url: str,
              max_pages: int = 10,
              save_output: bool = False,
              output_dir: str = 'crawled_data',
              expand: bool = False,
              **kwargs) -> list:
        """
        Crawl a website starting from the given URL.
        
        Args:
            url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            save_output: Whether to save crawled content to files
            output_dir: Directory to save crawled content
            expand: Whether to fetch full content from linked pages
            **kwargs: Additional arguments
            
        Returns:
            List of crawled content dictionaries
        """
        # Set the URL for the web instance
        self.web.url = url
        self.web.max_pages = max_pages
        
        # Perform the crawl
        return self.web.crawl(save_output=save_output, output_dir=output_dir, expand=expand)
    
    def text(self, url: str) -> str:
        """
        Get just the text content from a webpage.
        
        Args:
            url: URL to fetch text from
            
        Returns:
            Plain text content of the webpage
        """
        return self.web.text(url)
    
    def __repr__(self) -> str:
        return f"WebContent(content={self.content[:50]}...)" if self.content else "WebContent()"
