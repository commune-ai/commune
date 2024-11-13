import requests
import base64
import json


url = f'https://github.com/TopEagle36/pineapple-fastapi'

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_github_repo_text(repo_url):
    try:
        # Send a GET request to the repository URL
        response = requests.get(repo_url)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all files and directories
        content = soup.find('div', {'class': 'js-details-container Details'})
        
        if not content:
            return "Unable to find repository content"
            
        # Get all file links
        files = []
        for item in content.find_all('a', {'class': 'js-navigation-open'}):
            file_url = urljoin(repo_url, item['href'])
            if 'blob' in file_url:  # Only process files, not directories
                files.append(file_url)
        
        # Extract text from each file
        all_text = []
        for file_url in files:
            try:
                # Convert URL to raw content URL
                raw_url = file_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                file_content = requests.get(raw_url).text
                all_text.append(f"\n--- File: {file_url} ---\n")
                all_text.append(file_content)
            except Exception as e:
                all_text.append(f"Error reading file {file_url}: {str(e)}")
                
        return '\n'.join(all_text)
        
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    text = get_github_repo_text(url)
    print(text)
   