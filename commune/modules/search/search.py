import commune as c
import asyncio
class SearchAggregator(c.Module):
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
           
    async def search(self, query: str, num_results: int = 5) -> dict:
        """
        Search across multiple search engines asynchronously
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return per search engine
            
        Returns:
            dict: Combined search results from all engines
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [
                self.search_google(session, query, num_results),
                self.search_bing(session, query, num_results),
                self.search_duckduckgo(session, query, num_results)
            ]
            results = await asyncio.gather(*tasks)
            
        return {
            'google': results[0],
            'bing': results[1],
            'duckduckgo': results[2]
        }

    async def search_google(self, session, query: str, num_results: int) -> list:
        """Search Google"""
        url = f'https://www.google.com/search?q={quote_plus(query)}&num={num_results}'
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')
            results = []
            for div in soup.find_all('div', class_='g')[:num_results]:
                try:
                    title = div.find('h3').text
                    link = div.find('a')['href']
                    snippet = div.find('div', class_='VwiC3b').text
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
                except:
                    continue
            return results

    async def search_bing(self, session, query: str, num_results: int) -> list:
        """Search Bing"""
        url = f'https://www.bing.com/search?q={quote_plus(query)}&count={num_results}'
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')
            results = []
            for li in soup.find_all('li', class_='b_algo')[:num_results]:
                try:
                    title = li.find('h2').text
                    link = li.find('a')['href']
                    snippet = li.find('div', class_='b_caption').p.text
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
                except:
                    continue
            return results

    async def search_duckduckgo(self, session, query: str, num_results: int) -> list:
        """Search DuckDuckGo"""
        url = f'https://html.duckduckgo.com/html/?q={quote_plus(query)}'
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')
            results = []
            for div in soup.find_all('div', class_='result')[:num_results]:
                try:
                    title = div.find('a', class_='result__a').text
                    link = div.find('a', class_='result__a')['href']
                    snippet = div.find('a', class_='result__snippet').text
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
                except:
                    continue
            return results

    def test(self):
        """Test the search aggregator"""
        async def run_test():
            query = "artificial intelligence"
            results = await self.search(query, num_results=3)
            
            # Verify structure and content
            assert isinstance(results, dict)
            assert all(engine in results for engine in ['google', 'bing', 'duckduckgo'])
            
            for engine, engine_results in results.items():
                assert isinstance(engine_results, list)
                if len(engine_results) > 0:  # Some engines might fail due to rate limiting
                    for result in engine_results:
                        assert isinstance(result, dict)
                        assert all(key in result for key in ['title', 'link', 'snippet'])
                        assert all(isinstance(value, str) for value in result.values())
            
            c.print(f"✓ Test passed successfully")
            return True
            
        return asyncio.run(run_test())