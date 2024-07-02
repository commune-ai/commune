import unittest
from agent.agent import WebSurfingAgent

class TestWebSurfingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = WebSurfingAgent()

    def test_fetch_web_page(self):
        url = "https://www.medium.com"
        content = self.agent.fetch_web_page(url)
        self.assertIsInstance(content, str)

    def test_search_the_web(self):
        query = "test query"
        results = self.agent.search_the_web(query)
        self.assertIsInstance(results, dict)

if __name__ == '__main__':
    unittest.main()