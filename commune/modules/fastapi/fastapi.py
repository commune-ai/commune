
import requests
class FastAPI:

    def schema(self, base_url="https://api.app.trustedstake.ai/api"):
        """
        Retrieves the OpenAPI schema from a FastAPI application.
        
        Args:
            base_url (str): The base URL of the FastAPI application
            
        Returns:
            dict: The OpenAPI schema as a dictionary
        """
        # FastAPI typically exposes the OpenAPI schema at /openapi.json
        schema_url = f"{base_url.rstrip('/')}/openapi.json"
        
        try:
            response = requests.get(schema_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                # Try alternative location
                alt_schema_url = f"{base_url.rstrip('/')}/docs/openapi.json"
                try:
                    alt_response = requests.get(alt_schema_url)
                    alt_response.raise_for_status()
                    return alt_response.json()
                except:
                    print(f"Could not find schema at {schema_url} or {alt_schema_url}")
                    raise e
            print(f"HTTP Error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            raise
        except json.JSONDecodeError:
            print("Failed to decode JSON response")
            raise

    def test(self):
        
        api_url="https://api.app.trustedstake.ai/api"
        try:
            schema = self.schema(api_url)
            
            # Print the schema overview
            print(f"API Title: {schema.get('info', {}).get('title')}")
            print(f"Version: {schema.get('info', {}).get('version')}")
            print(f"Number of endpoints: {len(schema.get('paths', {}))}")
            
            # Save the full schema to a file
            with open("fastapi_schema.json", "w") as f:
                json.dump(schema, f, indent=2)
            
            print("\nFull schema saved to 'fastapi_schema.json'")
            
            # Print available endpoints
            print("\nAvailable Endpoints:")
            for path, methods in schema.get('paths', {}).items():
                for method in methods:
                    print(f"{method.upper()} {path}")
                    
        except Exception as e:
            print(f"Failed to retrieve schema: {e}")