
from typing import Any, Dict, List, Union

class TableQuery:
    """Helper class for building queries against a specific table."""
    
    def __init__(self, client: 'SupabaseClient', table_name: str):
        """
        Initialize a query builder for a table.
        
        Args:
            client: SupabaseClient instance
            table_name: Name of the table to query
        """
        self.client = client
        self.table_name = table_name
        self.query_params = {}
        self.headers = client.headers.copy()
    
    def select(self, columns: str = "*") -> 'TableQuery':
        """
        Specify columns to select.
        
        Args:
            columns: Comma-separated column names or "*" for all
            
        Returns:
            Self for chaining
        """
        self.query_params["select"] = columns
        return self
    
    def eq(self, column: str, value: Any) -> 'TableQuery':
        """
        Add an equality filter.
        
        Args:
            column: Column name
            value: Value to compare
            
        Returns:
            Self for chaining
        """
        self.headers["Prefer"] = "return=representation"
        self.query_params[column] = f"eq.{value}"
        return self
    
    def gt(self, column: str, value: Any) -> 'TableQuery':
        """
        Add a greater than filter.
        
        Args:
            column: Column name
            value: Value to compare
            
        Returns:
            Self for chaining
        """
        self.query_params[column] = f"gt.{value}"
        return self
    
    def lt(self, column: str, value: Any) -> 'TableQuery':
        """
        Add a less than filter.
        
        Args:
            column: Column name
            value: Value to compare
            
        Returns:
            Self for chaining
        """
        self.query_params[column] = f"lt.{value}"
        return self
    
    def order(self, column: str, ascending: bool = True) -> 'TableQuery':
        """
        Add ordering.
        
        Args:
            column: Column to order by
            ascending: If True, order ascending; if False, descending
            
        Returns:
            Self for chaining
        """
        direction = "asc" if ascending else "desc"
        self.query_params["order"] = f"{column}.{direction}"
        return self
    
    def limit(self, count: int) -> 'TableQuery':
        """
        Limit the number of rows returned.
        
        Args:
            count: Maximum number of rows
            
        Returns:
            Self for chaining
        """
        self.query_params["limit"] = count
        return self
    
    def offset(self, count: int) -> 'TableQuery':
        """
        Skip a number of rows.
        
        Args:
            count: Number of rows to skip
            
        Returns:
            Self for chaining
        """
        self.query_params["offset"] = count
        return self
    
    def execute(self) -> List[Dict]:
        """
        Execute the query and return results.
        
        Returns:
            List of records matching the query
        """
        url = self.client._build_url(f"rest/v1/{self.table_name}")
        response = requests.get(url, headers=self.headers, params=self.query_params)
        return self.client._handle_response(response)
    
    def insert(self, data: Union[Dict, List[Dict]]) -> Dict:
        """
        Insert one or more rows.
        
        Args:
            data: Dictionary or list of dictionaries with column values
            
        Returns:
            Inserted records
        """
        url = self.client._build_url(f"rest/v1/{self.table_name}")
        response = requests.post(url, headers=self.headers, json=data)
        return self.client._handle_response(response)
    
    def update(self, data: Dict) -> Dict:
        """
        Update rows that match the query.
        
        Args:
            data: Dictionary with column values to update
            
        Returns:
            Updated records
        """
        url = self.client._build_url(f"rest/v1/{self.table_name}")
        response = requests.patch(url, headers=self.headers, 
                                 params=self.query_params, json=data)
        return self.client._handle_response(response)
    
    def delete(self) -> Dict:
        """
        Delete rows that match the query.
        
        Returns:
            Deleted records
        """
        url = self.client._build_url(f"rest/v1/{self.table_name}")
        self.headers["Prefer"] = "return=representation"
        response = requests.delete(url, headers=self.headers, params=self.query_params)
        return self.client._handle_response(response)

