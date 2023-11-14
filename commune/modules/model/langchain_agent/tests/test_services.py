
import unittest
from unittest.mock import patch
from services.commune_service import CommuneBlockchainService

class TestCommuneBlockchainService(unittest.TestCase):
    def setUp(self):
        self.service = CommuneBlockchainService(api_key='test_key', base_url='http://medium.com')

    @patch('services.commune_service.requests.post')
    def test_post_to_blockchain(self, mock_post):
        mock_post.return_value.status_code = 200
        data = {"test": "data"}
        response = self.service.post_to_blockchain(data)
        self.assertEqual(response.status_code, 200)

    @patch('services.commune_service.requests.get')
    def test_get_from_blockchain(self, mock_get):
        mock_get.return_value.status_code = 200
        query = "test query"
        response = self.service.get_from_blockchain(query)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()