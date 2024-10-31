import requests
import json

def get_folder_contents_advanced(url='commune-ai/commune.git', 
                                 host_url = 'https://github.com/',
                                 auth_token=None):
    try:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Python Script'
        }
        if not url.startswith(host_url):
            url = host_url + url
        
        if auth_token:
            headers['Authorization'] = f'token {auth_token}'
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse JSON response
        content = response.json()
        
        # If it's a GitHub API response, it will be a list of files/folders
        if isinstance(content, list):
            return json.dumps(content, indent=2)
        return response.text
        
    except Exception as e:
        print(f"Error: {e}")
        return None