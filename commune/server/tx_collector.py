
import os
import json
import time
import asyncio
import commune as c
from typing import Dict, List, Optional, Any, Union

class TxCollector:
    """
    Transaction Collector for tracking API requests and responses.
    This separates transaction tracking from the Store module to maintain separation of concerns.
    """
    
    def __init__(self, 
                 dirpath: str = '~/.commune/server/transactions',
                 retention_days: int = 30,
                 batch_size: int = 100):
        """
        Initialize the transaction collector
        
        Args:
            dirpath: Directory to store transaction logs
            retention_days: How many days to keep transaction logs
            batch_size: How many transactions to batch before writing to disk
        """
        self.dirpath = os.path.abspath(os.path.expanduser(dirpath))
        self.retention_days = retention_days
        self.batch_size = batch_size
        self.pending_transactions = []
        self.lock = asyncio.Lock()
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath, exist_ok=True)
            
        # Start background tasks
        asyncio.create_task(self.periodic_flush())
        asyncio.create_task(self.cleanup_old_transactions())
    
    async def record_transaction(self, transaction_data: Dict) -> str:
        """
        Record a transaction asynchronously
        
        Args:
            transaction_data: Dictionary containing transaction information
            
        Returns:
            Transaction ID
        """
        # Generate transaction ID
        tx_id = self.generate_tx_id(transaction_data)
        
        # Add metadata
        transaction_data['tx_id'] = tx_id
        transaction_data['timestamp'] = transaction_data.get('timestamp', time.time())
        
        # Add to pending transactions
        async with self.lock:
            self.pending_transactions.append(transaction_data)
            
            # If we've reached batch size, flush to disk
            if len(self.pending_transactions) >= self.batch_size:
                await self.flush_transactions()
                
        return tx_id
    
    def generate_tx_id(self, data: Dict) -> str:
        """Generate a unique transaction ID based on the data"""
        # Create a string representation of the data
        data_str = json.dumps(data, sort_keys=True)
        # Hash it with a timestamp to ensure uniqueness
        tx_hash = c.hash(f"{data_str}_{time.time()}")
        return tx_hash
    
    async def flush_transactions(self) -> None:
        """Flush pending transactions to disk"""
        async with self.lock:
            if not self.pending_transactions:
                return
                
            # Group transactions by date for easier querying
            date_str = time.strftime("%Y-%m-%d", time.localtime())
            hour_str = time.strftime("%H", time.localtime())
            
            # Create directory structure
            date_dir = os.path.join(self.dirpath, date_str)
            os.makedirs(date_dir, exist_ok=True)
            
            # Create filename with timestamp to avoid collisions
            filename = f"{hour_str}_{int(time.time())}.json"
            filepath = os.path.join(date_dir, filename)
            
            # Write transactions to file
            with open(filepath, 'w') as f:
                json.dump(self.pending_transactions, f)
                
            # Clear pending transactions
            self.pending_transactions = []
    
    async def periodic_flush(self) -> None:
        """Periodically flush transactions to disk"""
        while True:
            await asyncio.sleep(60)  # Flush every minute
            await self.flush_transactions()
    
    async def cleanup_old_transactions(self) -> None:
        """Clean up old transaction logs"""
        while True:
            await asyncio.sleep(86400)  # Run once a day
            
            # Get current time
            current_time = time.time()
            
            # Walk through directory and remove old files
            for root, dirs, files in os.walk(self.dirpath):
                for dir_name in dirs:
                    try:
                        # Parse directory name as date
                        dir_path = os.path.join(root, dir_name)
                        dir_time = time.mktime(time.strptime(dir_name, "%Y-%m-%d"))
                        
                        # If directory is older than retention period, remove it
                        if current_time - dir_time > self.retention_days * 86400:
                            import shutil
                            shutil.rmtree(dir_path)
                    except ValueError:
                        # Skip directories that don't match our date format
                        pass
    
    def query_transactions(self, 
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          client_key: Optional[str] = None,
                          path: Optional[str] = None,
                          limit: int = 100) -> List[Dict]:
        """
        Query transactions based on various filters
        
        Args:
            start_time: Start timestamp for query range
            end_time: End timestamp for query range
            client_key: Filter by client key
            path: Filter by request path
            limit: Maximum number of results to return
            
        Returns:
            List of matching transactions
        """
        results = []
        
        # Default time range if not specified
        if not end_time:
            end_time = time.time()
        if not start_time:
            start_time = end_time - 86400  # Default to last 24 hours
            
        # Convert timestamps to dates for directory traversal
        start_date = time.strftime("%Y-%m-%d", time.localtime(start_time))
        end_date = time.strftime("%Y-%m-%d", time.localtime(end_time))
        
        # Walk through date directories
        for root, dirs, files in os.walk(self.dirpath):
            dir_name = os.path.basename(root)
            
            # Skip if directory is outside our date range
            if dir_name < start_date or dir_name > end_date:
                continue
                
            # Process files in this directory
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        transactions = json.load(f)
                        
                    # Filter transactions
                    for tx in transactions:
                        # Skip if outside time range
                        tx_time = tx.get('timestamp', 0)
                        if tx_time < start_time or tx_time > end_time:
                            continue
                            
                        # Skip if client key doesn't match
                        if client_key and tx.get('client', {}).get('key') != client_key:
                            continue
                            
                        # Skip if path doesn't match
                        if path and tx.get('path') != path:
                            continue
                            
                        # Add to results
                        results.append(tx)
                        
                        # Check if we've reached the limit
                        if len(results) >= limit:
                            return results
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
                    
        return results
    
    def get_transaction(self, tx_id: str) -> Optional[Dict]:
        """
        Retrieve a specific transaction by ID
        
        Args:
            tx_id: Transaction ID to retrieve
            
        Returns:
            Transaction data or None if not found
        """
        # Walk through all transaction files
        for root, dirs, files in os.walk(self.dirpath):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        transactions = json.load(f)
                        
                    # Look for matching transaction
                    for tx in transactions:
                        if tx.get('tx_id') == tx_id:
                            return tx
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
                    
        return None
    
    def get_stats(self, days: int = 7) -> Dict:
        """
        Get transaction statistics
        
        Args:
            days: Number of days to include in statistics
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_transactions': 0,
            'transactions_by_day': {},
            'transactions_by_path': {},
            'transactions_by_client': {},
            'average_response_time': 0
        }
        
        # Calculate start time
        start_time = time.time() - (days * 86400)
        
        # Query transactions
        transactions = self.query_transactions(
            start_time=start_time,
            limit=10000  # Set a high limit for stats
        )
        
        # Calculate statistics
        total_response_time = 0
        for tx in transactions:
            # Increment total
            stats['total_transactions'] += 1
            
            # Group by day
            day = time.strftime("%Y-%m-%d", time.localtime(tx.get('timestamp', 0)))
            stats['transactions_by_day'][day] = stats['transactions_by_day'].get(day, 0) + 1
            
            # Group by path
            path = tx.get('path', 'unknown')
            stats['transactions_by_path'][path] = stats['transactions_by_path'].get(path, 0) + 1
            
            # Group by client
            client = tx.get('client', {}).get('key', 'unknown')
            stats['transactions_by_client'][client] = stats['transactions_by_client'].get(client, 0) + 1
            
            # Add response time if available
            if 'duration' in tx:
                total_response_time += tx['duration']
                
        # Calculate average response time
        if stats['total_transactions'] > 0:
            stats['average_response_time'] = total_response_time / stats['total_transactions']
            
        return stats
    
    async def test(self) -> Dict:
        """Run tests on the transaction collector"""
        # Generate test transaction
        test_tx = {
            'client': {'key': 'test_key'},
            'path': '/test',
            'method': 'GET',
            'ip': '127.0.0.1'
        }
        
        # Record transaction
        tx_id = await self.record_transaction(test_tx)
        
        # Force flush
        await self.flush_transactions()
        
        # Query transaction
        result = self.get_transaction(tx_id)
        
        # Verify result
        success = result is not None and result['tx_id'] == tx_id
        
        return {
            'success': success,
            'tx_id': tx_id,
            'result': result
        }
