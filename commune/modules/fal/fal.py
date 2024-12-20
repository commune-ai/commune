import os
import json
import subprocess
from typing import Optional
import commune as c

class fal:
    def __init__(self, api: Optional = None):
        """
        Initialize the FalDepthProcessor.
        
        Args:
            api_key (str, optional): FAL API key. If not provided, will try to get from environment.
        """
        self.api_key = c.module('apikey')(self).get_key()
        if not self.api_key:
            raise ValueError("FAL_KEY must be provided either as argument or environment variable")

    def process_depth(self, prompt: str, control_image_url: str) -> str:
        """
        Process depth estimation using FAL API.
        
        Args:
            prompt (str): The text prompt for image generation
            control_image_url (str): URL of the control image
            
        Returns:
            str: The request ID from the API response
        """
        # Prepare the curl command
        curl_command = [
            'curl',
            '--request', 'POST',
            '--url', 'https://queue.fal.run/fal-ai/flux-pro/v1/depth',
            '--header', f'Authorization: Key {self.api_key}',
            '--header', 'Content-Type: application/json',
            '--data', json.dumps({
                "prompt": prompt,
                "control_image_url": control_image_url
            })
        ]

        try:
            # Execute the curl command
            process = subprocess.Popen(
                curl_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Curl command failed: {stderr.decode()}")

            # Parse the response
            response = stdout.decode()
            
            # Extract request ID using string manipulation
            import re
            match = re.search(r'"request_id":\s*"([^"]+)"', response)
            if match:
                return match.group(1)
            else:
                raise ValueError("Could not find request_id in response")

        except Exception as e:
            raise RuntimeError(f"Error processing depth: {str(e)}")

    @staticmethod
    def example_usage():
        """
        Demonstrates how to use the FalDepthProcessor class.
        """
        processor = fal()
        request_id = processor.process_depth(
            prompt="A blackhole in space.",
            control_image_url="https://fal.media/files/penguin/vt-SeIOweN7_oYBsvGO6t.png"
        )
        print(f"Request ID: {request_id}")

