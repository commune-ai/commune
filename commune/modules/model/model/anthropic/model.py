import commune as c
import anthropic
            
class Anthropic(c.Module):
    """
    Anthropic module for managing Claude API interactions within the commune framework
    """
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = 'claude-3-sonnet',
                 max_tokens: int = 4096,
                 temperature: float = 0.7):
        """Initialize the Anthropic module
        
        Args:
            api_key (str): Anthropic API key
            model (str): Model to use (default: claude-3-sonnet)
            max_tokens (int): Maximum tokens for completion
            temperature (float): Sampling temperature
        """
        self.set_config(locals())
        self.api_key = api_key or c.get_api_key('anthropic')
        
    def forward(self, 
             prompt: str,
             system: str = None,
             stream: bool = False,
             **kwargs) -> str:
        """Call the Anthropic API
        
        Args:
            prompt (str): Input prompt
            system (str): System message
            stream (bool): Whether to stream response
            **kwargs: Additional arguments passed to API
            
        Returns:
            str: Model response
        """
        try:
            # Import anthropic here to avoid dependency issues

            client = anthropic.Anthropic(api_key=self.api_key)
            
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=stream,
                **kwargs
            )
            
            if stream:
                response = ""
                for chunk in message:
                    if chunk.content:
                        response += chunk.content[0].text
                        if hasattr(self, 'verbose') and self.verbose:
                            c.print(chunk.content[0].text, end='')
                return response
            
            return message.content[0].text
            
        except Exception as e:
            c.print(f"Error forwarding Anthropic API: {str(e)}")
            return str(e)
            
    def test(self):
        """Test the Anthropic module"""
        prompt = "Write a haiku about AI"
        response = self.forward(prompt)
        c.print(f"Prompt: {prompt}")
        c.print(f"Response: {response}")
        return response

