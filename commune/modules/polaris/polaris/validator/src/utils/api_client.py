"""
Standardized API client for interacting with the Polaris API.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from validator.src.config import ApiConfig
from validator.src.utils.logging_utils import log_exception

T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)

class ApiResponse(Generic[T]):
    """Standardized API response wrapper."""
    
    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        error: Optional[str] = None,
        status_code: Optional[int] = None
    ):
        """
        Initialize the API response.
        
        Args:
            success: Whether the request was successful
            data: The response data (if successful)
            error: Error message (if unsuccessful)
            status_code: HTTP status code
        """
        self.success = success
        self.data = data
        self.error = error
        self.status_code = status_code
    
    @classmethod
    def success_response(cls, data: T, status_code: int = 200) -> 'ApiResponse[T]':
        """Create a successful response."""
        return cls(True, data=data, status_code=status_code)
    
    @classmethod
    def error_response(cls, error: str, status_code: Optional[int] = None) -> 'ApiResponse[T]':
        """Create an error response."""
        return cls(False, error=error, status_code=status_code)

class ApiClient:
    """Client for interacting with the Polaris API."""
    
    def __init__(self, config: ApiConfig, timeout: int = 30):
        """
        Initialize the API client.
        
        Args:
            config: API configuration
            timeout: Request timeout in seconds
        """
        self.config = config
        self.timeout = timeout
        self.session = requests.Session()
    
    def _handle_request_exception(self, e: Exception, url: str) -> ApiResponse:
        """
        Handle request exceptions and return appropriate responses.
        
        Args:
            e: The exception that was raised
            url: The URL that was being requested
            
        Returns:
            An API response with error details
        """
        if isinstance(e, Timeout):
            log_exception(logger, f"Request to {url} timed out", e, include_traceback=False)
            return ApiResponse.error_response("Request timed out")
        elif isinstance(e, ConnectionError):
            log_exception(logger, f"Connection error for {url}", e, include_traceback=False)
            return ApiResponse.error_response("Connection error")
        elif isinstance(e, RequestException):
            log_exception(logger, f"Request exception for {url}", e)
            return ApiResponse.error_response(f"Request failed: {str(e)}")
        else:
            log_exception(logger, f"Unexpected error for {url}", e)
            return ApiResponse.error_response(f"Unexpected error: {str(e)}")
    
    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> ApiResponse:
        """
        Make a GET request to the API.
        
        Args:
            url: The URL to request
            params: Query parameters
            
        Returns:
            An API response
        """
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    return ApiResponse.success_response(data, response.status_code)
                except json.JSONDecodeError as e:
                    log_exception(logger, f"Failed to parse JSON response from {url}", e)
                    return ApiResponse.error_response("Invalid JSON response", response.status_code)
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logger.warning(error_msg)
                return ApiResponse.error_response(error_msg, response.status_code)
        
        except Exception as e:
            return self._handle_request_exception(e, url)
    
    def post(self, url: str, data: Any) -> ApiResponse:
        """
        Make a POST request to the API.
        
        Args:
            url: The URL to request
            data: Request body
            
        Returns:
            An API response
        """
        try:
            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code in (200, 201, 202):
                try:
                    data = response.json()
                    return ApiResponse.success_response(data, response.status_code)
                except json.JSONDecodeError:
                    # Some successful responses might not have JSON bodies
                    return ApiResponse.success_response({}, response.status_code)
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logger.warning(error_msg)
                return ApiResponse.error_response(error_msg, response.status_code)
        
        except Exception as e:
            return self._handle_request_exception(e, url)
    
    def put(self, url: str, data: Any) -> ApiResponse:
        """
        Make a PUT request to the API.
        
        Args:
            url: The URL to request
            data: Request body
            
        Returns:
            An API response
        """
        try:
            response = self.session.put(
                url,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code in (200, 201, 202, 204):
                try:
                    data = response.json()
                    return ApiResponse.success_response(data, response.status_code)
                except json.JSONDecodeError:
                    # Some successful responses might not have JSON bodies
                    return ApiResponse.success_response({}, response.status_code)
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logger.warning(error_msg)
                return ApiResponse.error_response(error_msg, response.status_code)
        
        except Exception as e:
            return self._handle_request_exception(e, url)
    
    def patch(self, url: str, data: Any) -> ApiResponse:
        """
        Make a PATCH request to the API.
        
        Args:
            url: The URL to request
            data: Request body
            
        Returns:
            An API response
        """
        try:
            response = self.session.patch(
                url,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code in (200, 201, 202, 204):
                try:
                    data = response.json()
                    return ApiResponse.success_response(data, response.status_code)
                except json.JSONDecodeError:
                    # Some successful responses might not have JSON bodies
                    return ApiResponse.success_response({}, response.status_code)
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logger.warning(error_msg)
                return ApiResponse.error_response(error_msg, response.status_code)
        
        except Exception as e:
            return self._handle_request_exception(e, url)
    
    def delete(self, url: str) -> ApiResponse:
        """
        Make a DELETE request to the API.
        
        Args:
            url: The URL to request
            
        Returns:
            An API response
        """
        try:
            response = self.session.delete(url, timeout=self.timeout)
            
            if response.status_code in (200, 202, 204):
                try:
                    data = response.json()
                    return ApiResponse.success_response(data, response.status_code)
                except json.JSONDecodeError:
                    # Some successful responses might not have JSON bodies
                    return ApiResponse.success_response({}, response.status_code)
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logger.warning(error_msg)
                return ApiResponse.error_response(error_msg, response.status_code)
        
        except Exception as e:
            return self._handle_request_exception(e, url)
    
    # Convenience methods for Polaris API
    
    def get_miners(self) -> ApiResponse[List[Dict[str, Any]]]:
        """Get all miners from the API."""
        url = self.config.get_miners_url()
        logger.debug(f"Fetching miners from {url}")
        return self.get(url)
    
    def get_miner(self, miner_id: str) -> ApiResponse[Dict[str, Any]]:
        """Get a specific miner's details."""
        url = self.config.get_miner_url(miner_id)
        logger.debug(f"Fetching miner {miner_id} from {url}")
        return self.get(url)
    
    def get_containers(self, miner_id: str) -> ApiResponse[List[str]]:
        """Get containers for a specific miner."""
        url = self.config.get_containers_url(miner_id)
        logger.debug(f"Fetching containers for miner {miner_id} from {url}")
        return self.get(url)
    
    def verify_miner(self, miner_id: str) -> ApiResponse[Dict[str, Any]]:
        """Verify a specific miner."""
        url = self.config.get_verify_url(miner_id)
        logger.debug(f"Verifying miner {miner_id} via {url}")
        return self.patch(url, {"status": "verified"}) 