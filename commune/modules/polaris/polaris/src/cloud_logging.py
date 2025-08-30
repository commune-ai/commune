import logging
import threading
import time

logger = logging.getLogger(__name__)

class CloudLogger:
    """A simple logger for sending logs to the cloud server."""
    
    def __init__(self, miner_id):
        """Initialize the cloud logger with the miner ID."""
        self.miner_id = miner_id
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the cloud logging thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._logging_loop, daemon=True)
        self.thread.start()
        logger.info(f"Cloud logging started for miner {self.miner_id}")
    
    def stop(self):
        """Stop the cloud logging thread."""
        if not self.running:
            return
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info(f"Cloud logging stopped for miner {self.miner_id}")
    
    def _logging_loop(self):
        """Main logging loop that runs in a separate thread."""
        while self.running:
            try:
                # In a real implementation, we would collect logs and send them to the server
                # For now, this is just a placeholder
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in cloud logging: {str(e)}")
                time.sleep(30)  # Wait longer after an error 