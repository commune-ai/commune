#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import platform
import signal
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import psutil

from src.user_manager import UserManager
from src.utils import configure_logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('heartbeat.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

server_url_ = os.getenv('SERVER_URL')

class HeartbeatService:
    def __init__(self, 
                 server_url: str = server_url_,
                 heartbeat_interval: int = 30):
        """Initialize HeartbeatService with configuration"""
        self.server_url = server_url.rstrip('/')
        self.heartbeat_interval = heartbeat_interval
        self.session = None
        self.is_running = False
        self.user_manager = UserManager()
        self.miner_id = None
        self.last_heartbeat = None
        self.user_info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_info.json')
        self.file_check_interval = 1  # Check every second for file creation
        
        logger.info("HeartbeatService initialized with:")
        logger.info(f"  Server URL: {self.server_url}")
        logger.info(f"  Heartbeat interval: {self.heartbeat_interval} seconds")
        logger.info(f"  User info path: {self.user_info_path}")

    def _is_user_registered(self) -> bool:
        """Check if a user is registered by verifying user_info.json exists and has required data"""
        try:
            if not os.path.exists(self.user_info_path):
                return False
                
            with open(self.user_info_path, 'r') as f:
                user_info = json.load(f)
                return bool(user_info.get('miner_id'))
                
        except Exception as e:
            logger.error(f"Error checking user registration: {str(e)}", exc_info=True)
            return False

    def _get_miner_id(self) -> str:
        """Get miner ID from user_info.json"""
        try:
            if os.path.exists(self.user_info_path):
                with open(self.user_info_path, 'r') as f:
                    config = json.load(f)
                    return config.get('miner_id')
        except Exception as e:
            logger.error(f"Error reading miner_id: {str(e)}", exc_info=True)
        return None

    def _get_system_metrics(self):
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # System info
            boot_time_timestamp = psutil.boot_time()
            boot_time = datetime.fromtimestamp(boot_time_timestamp).isoformat()
            uptime = time.time() - boot_time_timestamp

            # Network interfaces
            net_info = psutil.net_if_addrs()
            ip_address = "0.0.0.0"
            
            for interface, addresses in net_info.items():
                for addr in addresses:
                    if hasattr(addr, 'family') and addr.family == socket.AF_INET and not interface.startswith(('lo', 'docker', 'veth')):
                        ip_address = addr.address
                        break

            metrics = {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "network_latency": 0,
                "temperature": None
            }

            system_info = {
                "hostname": platform.node(),
                "ip_address": ip_address,
                "os_version": f"{platform.system()} {platform.release()}",
                "uptime": uptime,
                "last_boot": boot_time
            }

            return metrics, system_info
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}", exc_info=True)
            return {}, {}

    async def initialize(self) -> bool:
        """Initialize the heartbeat service"""
        try:
            self.miner_id = self._get_miner_id()
            if not self.miner_id:
                return False
            
            if self.session:
                await self.session.close()
                
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}", exc_info=True)
            return False

    async def send_heartbeat(self) -> bool:
        """Send a heartbeat signal to the server"""
        if not self.session:
            logger.error("Cannot send heartbeat: session not initialized")
            return False

        try:
            metrics, system_info = self._get_system_metrics()
            current_time = datetime.utcnow()
            
            heartbeat_data = {
                "timestamp": current_time.isoformat(),
                "status": "online",
                "version": "1.0.0",
                "metrics": {
                    "miner_id": self.miner_id,
                    "system_info": system_info,
                    "metrics": metrics,
                    "resource_usage": {},
                    "active_jobs": []
                }
            }
            
            endpoint = f"{self.server_url}/heart_beat"
            
            async with self.session.post(
                endpoint,
                json=heartbeat_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"PolarisHeartbeat/{platform.python_version()}"
                }
            ) as response:
                if response.status == 200:
                    self.last_heartbeat = time.time()
                    logger.info("Heartbeat sent successfully")
                    return True
                else:
                    logger.warning(f"Heartbeat failed with status {response.status}")
                    return False

        except aiohttp.ClientError as e:
            logger.error(f"Network error sending heartbeat: {str(e)}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending heartbeat: {str(e)}", exc_info=True)
            return False

    async def run(self):
        """Main service loop with active file monitoring"""
        logger.info("Starting heartbeat service")
        
        while True:
            try:
                # Check for registration
                if not self._is_user_registered():
                    logger.info("Waiting for user registration...")
                    # Quick check interval for file creation
                    await asyncio.sleep(self.file_check_interval)
                    continue

                # Initialize service when registration is found
                if not await self.initialize():
                    logger.error("Failed to initialize, retrying...")
                    await asyncio.sleep(self.file_check_interval)
                    continue

                logger.info("User registration detected - starting heartbeat signals")
                self.is_running = True

                # Enter main heartbeat loop
                while self.is_running and self._is_user_registered():
                    try:
                        success = await self.send_heartbeat()
                        
                        if success:
                            logger.info("Heartbeat sent successfully")
                            await asyncio.sleep(self.heartbeat_interval)
                        else:
                            logger.warning("Heartbeat failed, retrying in 10 seconds")
                            await asyncio.sleep(10)
                            
                    except Exception as e:
                        logger.error(f"Error in heartbeat cycle: {str(e)}")
                        await asyncio.sleep(10)

                # If we exit the loop, reset session
                if self.session:
                    await self.session.close()
                    self.session = None
                self.is_running = False

            except Exception as e:
                logger.error(f"Error in main service loop: {str(e)}")
                await asyncio.sleep(self.file_check_interval)

    async def stop(self):
        """Stop the heartbeat service"""
        logger.info("Stopping heartbeat service")
        self.is_running = False
        
        if self.session:
            await self.session.close()
            
        logger.info("Heartbeat service stopped")

async def main():
    """Main entry point for the heartbeat service"""
    try:
        logger.info("=" * 60)
        logger.info("Starting Polaris Heartbeat Service")
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Working Directory: {os.getcwd()}")
        logger.info("=" * 60)

        service = HeartbeatService()

        def signal_handler(sig, frame):
            sig_name = signal.Signals(sig).name
            logger.info(f"Received signal {sig_name} ({sig})")
            asyncio.create_task(service.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        await service.run()
        
    except Exception as e:
        logger.critical(f"Fatal error in heartbeat service: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())