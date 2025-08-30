import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the directory containing bt.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bt import Bittensor

class TestBittensor(unittest.TestCase):
    
    @patch('bt.bt.subtensor')
    def test_initialization(self, mock_subtensor):
        # Setup mock
        mock_instance = MagicMock()
        mock_subtensor.return_value = mock_instance
        
        # Create Bittensor instance
        bt_instance = Bittensor(network="finney")
        
        # Assert subtensor was called with correct network
        mock_subtensor.assert_called_once_with(network="finney")
        
        # Assert network property is set correctly
        self.assertEqual(bt_instance.network, "finney")
        
        # Assert subtensor instance is stored
        self.assertEqual(bt_instance.subtensor, mock_instance)
    
    @patch('bt.bt.subtensor')
    def test_list_wallets(self, mock_subtensor):
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.list_wallets.return_value = ["wallet1", "wallet2"]
        mock_subtensor.return_value = mock_instance
        
        # Create Bittensor instance and call list_wallets
        bt_instance = Bittensor()
        wallets = bt_instance.list_wallets()
        
        # Assert list_wallets was called
        mock_instance.list_wallets.assert_called_once()
        
        # Assert correct wallets were returned
        self.assertEqual(wallets, ["wallet1", "wallet2"])
    
    @patch('bt.bt.subtensor')
    def test_neurons(self, mock_subtensor):
        # Setup mock
        mock_instance = MagicMock()
        mock_neurons = [{"uid": 0}, {"uid": 1}]
        mock_instance.neurons.return_value = mock_neurons
        mock_subtensor.return_value = mock_instance
        
        # Create Bittensor instance and call neurons
        bt_instance = Bittensor()
        neurons = bt_instance.neurons(netuid=2)
        
        # Assert neurons was called with correct netuid
        mock_instance.neurons.assert_called_once_with(netuid=2)
        
        # Assert correct neurons were returned
        self.assertEqual(neurons, mock_neurons)
    
    @patch('bt.bt.subtensor')
    @patch('bt.bt.wallet')
    def test_transfer(self, mock_wallet, mock_subtensor):
        # Setup mocks
        mock_subtensor_instance = MagicMock()
        mock_subtensor_instance.transfer.return_value = True
        mock_subtensor.return_value = mock_subtensor_instance
        
        mock_wallet_instance = MagicMock()
        mock_wallet.return_value = mock_wallet_instance
        
        # Create Bittensor instance and call transfer
        bt_instance = Bittensor()
        result = bt_instance.transfer(
            wallet_name="test_wallet",
            dest_address="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            amount=10.0
        )
        
        # Assert wallet was created with correct name
        mock_wallet.assert_called_once_with("test_wallet")
        
        # Assert transfer was called with correct parameters
        mock_subtensor_instance.transfer.assert_called_once()
        
        # Assert result is correct
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()