# Polaris CLI Tool - Heartbeat Service

## Table of Contents

1. [Overview](#overview)
2. [Module: `heartbeat_service.py`](#module-heartbeat_servicepy)
    - [Imports and Dependencies](#imports-and-dependencies)
    - [Logging Configuration](#logging-configuration)
    - [Class: `HeartbeatService`](#class-heartbeatservice)
        - [Initialization (`__init__`)](#initialization-init)
        - [_get_miner_id Method](#_get_miner_id-method)
        - [_get_system_metrics Method](#_get_system_metrics-method)
        - [initialize Method](#initialize-method)
        - [send_heartbeat Method](#send_heartbeat-method)
        - [run Method](#run-method)
        - [stop Method](#stop-method)
    - [Function: `main`](#function-main)
3. [Workflow](#workflow)
4. [Error Handling](#error-handling)
5. [Integration with Other Modules](#integration-with-other-modules)
6. [Conclusion](#conclusion)

---

## Overview

The **Heartbeat Service** is a critical component of the Polaris CLI tool, responsible for regularly sending system metrics and status updates (heartbeats) to a remote orchestrator server. This ensures that the orchestrator is aware of the compute resources' health and availability, facilitating effective management and monitoring.

---

## Module: `heartbeat_service.py`

This module encapsulates the functionality required to collect system metrics and send periodic heartbeat signals to a predefined server endpoint. It leverages asynchronous programming for efficient operation and integrates seamlessly with other components like user management and utility functions.

### Imports and Dependencies

```python
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
```

- **Standard Libraries**:
    - `asyncio`: For asynchronous programming.
    - `json`: To handle JSON data.
    - `logging`: For logging events and errors.
    - `os`, `sys`, `platform`: To interact with the operating system.
    - `signal`: To handle termination signals.
    - `socket`: For network-related operations.
    - `time`, `datetime`: For time tracking and formatting.
    - `pathlib.Path`: For filesystem path manipulations.

- **Third-Party Libraries**:
    - `aiohttp`: For making asynchronous HTTP requests.
    - `psutil`: For retrieving system and process information.

- **Internal Modules**:
    - `UserManager`: Manages user-related data.
    - `configure_logging`: Sets up logging configurations.

### Logging Configuration

```python
logger = logging.getLogger(__name__)
```

- Initializes a logger specific to this module using the module's `__name__`.
- Logging is further configured in the `main` function to include both file and console handlers with a detailed format.

### Class: `HeartbeatService`

The `HeartbeatService` class encapsulates all functionalities related to sending heartbeats to the server.

#### Initialization (`__init__`)

```python
def __init__(self, 
             server_url: str = "https://polaris-test-server.onrender.com/api/v1",
             heartbeat_interval: int = 30):
    """Initialize HeartbeatService with configuration"""
    self.server_url = server_url.rstrip('/')
    self.heartbeat_interval = heartbeat_interval
    self.session = None
    self.is_running = False
    self.user_manager = UserManager()
    self.miner_id = None
    self.last_heartbeat = None
    
    logger.info("HeartbeatService initialized with:")
    logger.info(f"  Server URL: {self.server_url}")
    logger.info(f"  Heartbeat interval: {self.heartbeat_interval} seconds")
```

- **Parameters**:
    - `server_url`: The base URL of the orchestrator server to which heartbeats are sent.
    - `heartbeat_interval`: Time interval (in seconds) between consecutive heartbeats.

- **Attributes**:
    - `self.session`: An `aiohttp` session for making HTTP requests.
    - `self.is_running`: A flag indicating whether the service is active.
    - `self.user_manager`: An instance of `UserManager` to access user data.
    - `self.miner_id`: The unique identifier for the miner, retrieved from user data.
    - `self.last_heartbeat`: Timestamp of the last successful heartbeat.

- **Logging**:
    - Logs initialization parameters for traceability.

#### `_get_miner_id` Method

```python
def _get_miner_id(self) -> str:
    """Get miner ID from config file"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_info.json')
        logger.debug(f"Looking for config file at: {config_path}")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                miner_id = config.get('miner_id')
                if miner_id:
                    logger.info(f"Loaded miner_id from config: {miner_id}")
                    return miner_id
                else:
                    logger.error("No miner_id found in config file")
        else:
            logger.error(f"Config file not found at {config_path}")
    except Exception as e:
        logger.error(f"Error reading miner_id from config: {str(e)}", exc_info=True)
    return None
```

- **Purpose**: Retrieves the `miner_id` from the `user_info.json` configuration file.
- **Process**:
    1. Constructs the path to `user_info.json` located two directories up from the current file.
    2. Checks if the file exists:
        - If it exists, loads the JSON content and extracts `miner_id`.
        - Logs an error if `miner_id` is missing.
    3. Logs an error if the config file doesn't exist.
    4. Catches and logs any exceptions that occur during the process.
- **Returns**: The `miner_id` as a string if found; otherwise, `None`.

#### `_get_system_metrics` Method

```python
def _get_system_metrics(self):
    """Get current system metrics"""
    try:
        logger.debug("Collecting system metrics...")
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        logger.debug(f"CPU Usage: {cpu_usage}%")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        logger.debug(f"Memory Usage: {memory_usage}%")
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        logger.debug(f"Disk Usage: {disk_usage}%")
        
        # System info
        boot_time_timestamp = psutil.boot_time()
        boot_time = datetime.fromtimestamp(boot_time_timestamp).isoformat()
        uptime = time.time() - boot_time_timestamp
        logger.debug(f"System Uptime: {uptime:.2f} seconds")

        # Network interfaces
        net_info = psutil.net_if_addrs()
        ip_address = "0.0.0.0"
        
        logger.debug("Scanning network interfaces...")
        for interface, addresses in net_info.items():
            logger.debug(f"Checking interface: {interface}")
            for addr in addresses:
                if hasattr(addr, 'family') and addr.family == socket.AF_INET and not interface.startswith(('lo', 'docker', 'veth')):
                    ip_address = addr.address
                    logger.debug(f"Found usable IP address: {ip_address} on interface {interface}")
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

        logger.debug("System metrics collected successfully")
        logger.debug(f"System Info: {json.dumps(system_info, indent=2)}")
        logger.debug(f"Metrics: {json.dumps(metrics, indent=2)}")

        return metrics, system_info
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}", exc_info=True)
        return {}, {}
```

- **Purpose**: Gathers real-time system metrics to include in the heartbeat data.
- **Metrics Collected**:
    - **CPU Usage**: Percentage of CPU utilization over 1 second.
    - **Memory Usage**: Percentage of used virtual memory.
    - **Disk Usage**: Percentage of used disk space on the root partition.
    - **Network Latency**: Placeholder (`0`) as actual latency measurement is not implemented.
    - **Temperature**: Placeholder (`None`) as temperature measurement is not implemented.
    
- **System Information**:
    - **Hostname**: The network name of the machine.
    - **IP Address**: The first non-loopback IPv4 address found.
    - **OS Version**: The operating system and its version.
    - **Uptime**: Time since the last system boot in seconds.
    - **Last Boot**: Timestamp of the last system boot in ISO format.

- **Process**:
    1. Uses `psutil` to retrieve CPU, memory, and disk usage.
    2. Calculates system uptime and boot time.
    3. Iterates over network interfaces to find a usable IPv4 address, excluding loopback and virtual interfaces.
    4. Logs detailed debug information for each metric collected.
    5. Returns a tuple containing `metrics` and `system_info`.

- **Error Handling**:
    - Catches and logs any exceptions that occur during metric collection.
    - Returns empty dictionaries if an error is encountered.

#### `initialize` Method

```python
async def initialize(self) -> bool:
    """Initialize the heartbeat service"""
    try:
        logger.info("Starting heartbeat service initialization")
        
        # Get miner ID
        self.miner_id = self._get_miner_id()
        if not self.miner_id:
            logger.error("No miner_id available. Please configure miner_id in config.json")
            return False
        
        # Initialize HTTP session
        logger.info("Creating aiohttp ClientSession with 10s timeout")
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        logger.info("Heartbeat service initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize heartbeat service: {str(e)}", exc_info=True)
        return False
```

- **Purpose**: Prepares the `HeartbeatService` for operation by:
    1. Retrieving the `miner_id`.
    2. Establishing an asynchronous HTTP session with a timeout.
    
- **Process**:
    1. Logs the start of initialization.
    2. Calls `_get_miner_id` to fetch the miner's unique identifier.
    3. Validates the presence of `miner_id`; logs an error and aborts if missing.
    4. Creates an `aiohttp` session with a 10-second total timeout for requests.
    5. Logs successful initialization and returns `True`.
    
- **Error Handling**:
    - Catches and logs any exceptions during initialization.
    - Returns `False` if initialization fails.

#### `send_heartbeat` Method

```python
async def send_heartbeat(self) -> bool:
    """Send a heartbeat signal to the server"""
    if not self.session:
        logger.error("Cannot send heartbeat: session not initialized")
        return False

    try:
        logger.debug("Preparing heartbeat data...")
        metrics, system_info = self._get_system_metrics()
        current_time = datetime.utcnow()
        
        heartbeat_data = {
            "timestamp": current_time.isoformat(),
            "status": "online",  # Changed from "ONLINE" to "online"
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
        logger.info(f"Sending heartbeat to: {endpoint}")
        logger.debug(f"Request payload: {json.dumps(heartbeat_data, indent=2)}")

        start_time = time.time()
        logger.debug("Making POST request...")
        
        async with self.session.post(
            endpoint,
            json=heartbeat_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"PolarisHeartbeat/{platform.python_version()}"
            }
        ) as response:
            response_time = time.time() - start_time
            logger.info(f"Response received in {response_time:.3f} seconds with status {response.status}")
            
            response_body = await response.text()
            logger.debug(f"Response headers: {dict(response.headers)}")
            logger.debug(f"Response body: {response_body}")
            
            if response.status == 200:
                self.last_heartbeat = time.time()
                logger.info("Heartbeat sent successfully")
                return True
            elif response.status == 422:
                logger.error(f"Data validation error. Response: {response_body}")
                return False
            else:
                logger.warning(f"Heartbeat failed with status {response.status}. Response: {response_body}")
                return False

    except aiohttp.ClientError as e:
        logger.error(f"Network error sending heartbeat: {str(e)}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending heartbeat: {str(e)}", exc_info=True)
        return False
```

- **Purpose**: Constructs and sends a heartbeat JSON payload to the orchestrator server, handling responses appropriately.

- **Process**:
    1. Validates that the HTTP session (`self.session`) is initialized.
    2. Collects system metrics and system information using `_get_system_metrics`.
    3. Constructs the `heartbeat_data` dictionary with:
        - `timestamp`: Current UTC time in ISO format.
        - `status`: `"online"`.
        - `version`: Hardcoded as `"1.0.0"`.
        - `metrics`: Nested dictionary containing miner ID, system info, collected metrics, and placeholders for `resource_usage` and `active_jobs`.
    4. Defines the endpoint URL by appending `/heart_beat` to `server_url`.
    5. Logs the endpoint and the payload for debugging.
    6. Sends an asynchronous POST request with the heartbeat data and appropriate headers.
    7. Measures and logs the response time and status.
    8. Processes the response:
        - **200 OK**: Marks the heartbeat as successful.
        - **422 Unprocessable Entity**: Logs a data validation error.
        - **Other Status Codes**: Logs a warning with the response body.
    9. Catches and logs any network-related or unexpected errors.

- **Returns**: `True` if the heartbeat was sent successfully (`200 OK`); otherwise, `False`.

#### `run` Method

```python
async def run(self):
    """Main service loop"""
    if not await self.initialize():
        logger.error("Failed to initialize heartbeat service, exiting")
        return

    self.is_running = True
    logger.info(f"Heartbeat service started with interval of {self.heartbeat_interval} seconds")

    retry_interval = self.heartbeat_interval
    cycle_count = 0
    
    while self.is_running:
        try:
            cycle_count += 1
            logger.info(f"Starting heartbeat cycle #{cycle_count}")
            
            success = await self.send_heartbeat()
            
            if success:
                logger.info(f"Heartbeat cycle #{cycle_count} completed successfully")
                retry_interval = self.heartbeat_interval
            else:
                logger.warning(f"Heartbeat cycle #{cycle_count} failed")
                retry_interval = min(retry_interval * 2, 300)
                logger.info(f"Increasing retry interval to {retry_interval} seconds")
            
            logger.debug(f"Sleeping for {retry_interval} seconds until next cycle")
            await asyncio.sleep(retry_interval)
            
        except Exception as e:
            logger.error(f"Error in heartbeat cycle #{cycle_count}: {str(e)}", exc_info=True)
            await asyncio.sleep(retry_interval)
```

- **Purpose**: Manages the continuous operation of the heartbeat service, handling periodic heartbeats and implementing retry logic on failures.

- **Process**:
    1. Calls `initialize` to set up the service; exits if initialization fails.
    2. Sets `self.is_running` to `True` to enter the main loop.
    3. Initializes `retry_interval` to the configured `heartbeat_interval`.
    4. Enters a `while` loop that continues as long as `self.is_running` is `True`.
    5. For each cycle:
        - Increments the `cycle_count`.
        - Logs the start of the heartbeat cycle.
        - Attempts to send a heartbeat via `send_heartbeat`.
        - Based on the success of the heartbeat:
            - **Success**:
                - Logs successful completion.
                - Resets `retry_interval` to the standard interval.
            - **Failure**:
                - Logs the failure.
                - Doubles the `retry_interval`, capping it at 300 seconds (5 minutes).
                - Logs the updated retry interval.
        - Sleeps for `retry_interval` seconds before the next cycle.
    6. Catches and logs any exceptions within the loop, ensuring the service continues running by sleeping before the next attempt.

- **Retry Logic**:
    - On failure, the interval between heartbeats increases exponentially (doubling each time) up to a maximum of 5 minutes. This prevents overwhelming the server or network during prolonged outages.

#### `stop` Method

```python
async def stop(self):
    """Stop the heartbeat service"""
    logger.info("Stopping heartbeat service...")
    self.is_running = False
    
    if self.session:
        logger.debug("Closing aiohttp session")
        await self.session.close()
        
    logger.info("Heartbeat service stopped successfully")
```

- **Purpose**: Gracefully terminates the heartbeat service by:
    1. Setting `self.is_running` to `False` to exit the main loop.
    2. Closing the `aiohttp` session to free up resources.
    3. Logging the termination.

---

### Function: `main`

```python
async def main():
    """Main entry point for the heartbeat service"""
    try:
        # Configure logging with more detailed format
        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
            format='%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler('heartbeat.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

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
```

- **Purpose**: Serves as the entry point for the heartbeat service when the script is executed directly.

- **Process**:
    1. **Logging Configuration**:
        - Sets the logging level to `DEBUG` for comprehensive logging.
        - Defines the log format to include timestamps with milliseconds, log levels, module names, and messages.
        - Configures two handlers:
            - `FileHandler`: Writes logs to `heartbeat.log`.
            - `StreamHandler`: Outputs logs to `stdout` (console).
    2. **Startup Logs**:
        - Logs a separator line.
        - Logs the start of the Polaris Heartbeat Service.
        - Logs Python version, platform details, and the current working directory.
        - Logs another separator line.
    3. **Service Initialization**:
        - Instantiates the `HeartbeatService`.
    4. **Signal Handling**:
        - Defines `signal_handler` to gracefully stop the service upon receiving termination signals (`SIGINT`, `SIGTERM`).
        - Registers the signal handler for `SIGINT` and `SIGTERM`.
    5. **Service Execution**:
        - Calls `service.run()` to start the main loop.
    6. **Exception Handling**:
        - Catches and logs any fatal errors, then exits the program.
    7. **Cleanup**:
        - Ensures that `service.stop()` is called in the `finally` block to terminate the service gracefully, regardless of how the program exits.

- **Execution**:
    - The `main` function is run within an asyncio event loop using `asyncio.run(main())` when the script is executed directly.

---

## Workflow

1. **Initialization**:
    - The service starts and initializes logging.
    - It creates an instance of `HeartbeatService` with default configurations.
    - Retrieves the `miner_id` from the configuration file.
    - Establishes an asynchronous HTTP session for communication with the server.

2. **Heartbeat Cycle**:
    - Enters a loop that runs as long as `self.is_running` is `True`.
    - In each cycle:
        - Collects current system metrics (CPU, memory, disk usage, etc.).
        - Constructs a heartbeat payload with these metrics and system information.
        - Sends the heartbeat to the orchestrator server via an HTTP POST request.
        - Logs the response status and adjusts the retry interval based on success or failure.
        - Sleeps for the specified interval before the next cycle.

3. **Error Handling and Retries**:
    - On successful heartbeat, the interval resets to the default.
    - On failure, the interval doubles (exponentially backoff) up to a maximum of 5 minutes.
    - Ensures resilience against temporary network issues or server downtimes.

4. **Termination**:
    - Upon receiving termination signals (`SIGINT`, `SIGTERM`), the service gracefully stops by closing the HTTP session and exiting the loop.

---

## Error Handling

- **Configuration Errors**:
    - Missing or invalid `miner_id` leads to service termination.
    - Absence of `user_info.json` or missing `miner_id` is logged as an error.

- **Network Errors**:
    - `aiohttp.ClientError` is caught and logged.
    - Unexpected exceptions during heartbeat sending are caught and logged to prevent crashes.

- **System Metrics Collection Errors**:
    - Any issues during metric collection are logged, and empty dictionaries are returned to avoid sending incomplete data.

- **Service Initialization Errors**:
    - Failures during initialization halt the service with appropriate logging.

---

## Integration with Other Modules

- **`UserManager`**:
    - Accesses user-specific data, particularly the `miner_id` required for heartbeat identification.
  
- **`configure_logging`**:
    - Provides a centralized logging configuration that can be reused across different modules.

- **`src/user_manager` and `src/utils`**:
    - Ensure that user data is managed securely and that utility functions support common tasks like logging and configuration handling.

---
