import os
from typing import Dict, List, Any, Optional, Union
import requests
import json
from .table import TableQuery



class SupabaseClient:
    """
    A Python client for interacting with Supabase.
    
    This class provides methods to perform CRUD operations on Supabase tables,
    handle authentication, and manage storage.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the Supabase client.
        
        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase API key
        """
        self.supabase_url = supabase_url.rstrip('/')
        self.supabase_key = supabase_key
        self.headers = {
            "apikey": supabase_key,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        self.auth_token = None
    
    def _build_url(self, path: str) -> str:
        """Build a URL for the Supabase API."""
        return f"{self.supabase_url}/{path}"
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle the API response."""
        if response.status_code >= 400:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        return response.json()
    
    def set_auth_token(self, token: str) -> None:
        """
        Set the authentication token for authenticated requests.
        
        Args:
            token: JWT token from sign_in or sign_up
        """
        self.auth_token = token
        self.headers["Authorization"] = f"Bearer {token}"
    
    # Database operations
    
    def from_table(self, table_name: str) -> 'TableQuery':
        """
        Create a query builder for a specific table.
        
        Args:
            table_name: The name of the table to query
            
        Returns:
            TableQuery object for chaining query operations
        """
        return TableQuery(self, table_name)
    
    # Auth operations
    
    def sign_up(self, email: str, password: str) -> Dict:
        """
        Register a new user with email and password.
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Response containing user data and session
        """
        url = self._build_url("auth/v1/signup")
        data = {"email": email, "password": password}
        response = requests.post(url, headers=self.headers, json=data)
        result = self._handle_response(response)
        
        if "access_token" in result:
            self.set_auth_token(result["access_token"])
        
        return result
    
    def sign_in(self, email: str, password: str) -> Dict:
        """
        Sign in an existing user with email and password.
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Response containing user data and session
        """
        url = self._build_url("auth/v1/token?grant_type=password")
        data = {"email": email, "password": password}
        response = requests.post(url, headers=self.headers, json=data)
        result = self._handle_response(response)
        
        if "access_token" in result:
            self.set_auth_token(result["access_token"])
        
        return result
    
    def sign_out(self) -> Dict:
        """
        Sign out the current user.
        
        Returns:
            Response indicating success
        """
        if not self.auth_token:
            raise Exception("No active session")
            
        url = self._build_url("auth/v1/logout")
        response = requests.post(url, headers=self.headers)
        self.auth_token = None
        self.headers.pop("Authorization", None)
        return self._handle_response(response)
    
    # Storage operations
    
    def upload_file(self, bucket: str, path: str, file_path: str) -> Dict:
        """
        Upload a file to Supabase Storage.
        
        Args:
            bucket: Storage bucket name
            path: Path within the bucket
            file_path: Local file path to upload
            
        Returns:
            Response with file metadata
        """
        url = self._build_url(f"storage/v1/object/{bucket}/{path}")
        
        with open(file_path, "rb") as f:
            file_data = f.read()
            
        # Remove Content-Type for multipart form
        headers = self.headers.copy()
        headers.pop("Content-Type", None)
        
        response = requests.post(url, headers=headers, data=file_data)
        return self._handle_response(response)
    
    def get_file_url(self, bucket: str, path: str) -> str:
        """
        Get a public URL for a file.
        
        Args:
            bucket: Storage bucket name
            path: Path to the file within the bucket
            
        Returns:
            Public URL for the file
        """
        url = self._build_url(f"storage/v1/object/public/{bucket}/{path}")
        return url

