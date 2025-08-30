import requests
import json

class ASIConnector:
    """
    A class to programmatically interact with ASI:One (Fetch.ai's Agentic AI).
    Handles queries, multi-turn conversations, and structured data exchange.
    """
    url = 'https://asi1.ai/'
    
    def __init__(self, api_key=None, base_url="https://api.asi.fetch.ai/v1"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else None
        }
        self.session_id = None  # For maintaining conversation context
        
    def query(self, prompt, agentic_mode=False, **kwargs):
        """
        Send a query to ASI:One with optional agentic workflow controls.
        
        Args:
            prompt (str): Input text/instruction
            agentic_mode (bool): Enable autonomous task decomposition
            **kwargs: Additional params (tools, temperature, etc.)
        """
        payload = {
            "prompt": prompt,
            "parameters": {
                "agentic": agentic_mode,
                "session_id": self.session_id,
                **kwargs
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/query",
                headers=self.headers,
                data=json.dumps(payload)
            )
            result = response.json()
            
            # Update session if new context was created
            if 'session_id' in result:
                self.session_id = result['session_id']
                
            return result.get('output', "No response generated")
            
        except Exception as e:
            return f"API Error: {str(e)}"
    
    def tool_execution(self, tool_name, params):
        """Trigger specific agentic tools/plugins"""
        return self.query(
            f"@tool:{tool_name}", 
            params=params,
            agentic_mode=True
        )

# Example usage
if __name__ == "__main__":
    asi = ASIConnector(api_key="your_key_here")
    
    # Simple query
    print(asi.query("Explain quantum entanglement to a 5-year-old"))
    
    # Agentic workflow
    print(asi.query(
        "Book a flight to Tokyo next week",
        agentic_mode=True,
        tools=["calendar", "web_search"]
    ))
