import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
import requests

@dataclass
class GitHubAccount:
    """Data class to store GitHub account information"""
    username: str
    email: str
    token: str  # Personal Access Token
    is_active: bool = False

class GitHubAccountManager:
    """Manages multiple GitHub accounts and switching between them"""
    
    def __init__(self, config_file: str = "~/.github_accounts.json"):
        self.config_file = Path(config_file).expanduser()
        self.accounts: Dict = {}
        self.load_accounts()
    
    def load_accounts(self) -> None:
        """Load accounts from configuration file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for username, info in data.items():
                        self.accounts = GitHubAccount(**info)
            except Exception as e:
                print(f"Error loading accounts: {e}")
    
    def save_accounts(self) -> None:
        """Save accounts to configuration file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            username: {
                'username': acc.username,
                'email': acc.email,
                'token': acc.token,
                'is_active': acc.is_active
            }
            for username, acc in self.accounts.items()
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
        # Secure the file (Unix-like systems)
        os.chmod(self.config_file, 0o600)
    
    def add_account(self, username: str, email: str, token: str) -> bool:
        """Add a new GitHub account"""
        try:
            # Verify the token works
            headers = {'Authorization': f'token {token}'}
            response = requests.get('https://api.github.com/user', headers=headers)
            
            if response.status_code == 200:
                self.accounts = GitHubAccount(username, email, token)
                self.save_accounts()
                print(f"Successfully added account: {username}")
                return True
            else:
                print(f"Failed to verify token: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error adding account: {e}")
            return False
    
    def remove_account(self, username: str) -> bool:
        """Remove a GitHub account"""
        if username in self.accounts:
            del self.accounts
            self.save_accounts()
            print(f"Removed account: {username}")
            return True
        print(f"Account not found: {username}")
        return False
    
    def list_accounts(self) -> List:
        """List all stored accounts"""
        accounts_info = []
        for username, account in self.accounts.items():
            status = "ACTIVE" if account.is_active else ""
            accounts_info.append(f"{username} ({account.email}) {status}")
        return accounts_info
    
    def get_active_account(self) -> Optional[GitHubAccount]:
        """Get the currently active account"""
        for account in self.accounts.values():
            if account.is_active:
                return account
        return None
    
    def switch_account(self, username: str) -> bool:
        """Switch to a different GitHub account"""
        if username not in self.accounts:
            print(f"Account not found: {username}")
            return False
        
        try:
            # Deactivate all accounts
            for acc in self.accounts.values():
                acc.is_active = False
            
            # Activate selected account
            account = self.accounts
            account.is_active = True
            
            # Update Git global configuration
            self._configure_git(account)
            
            # Update GitHub CLI if installed
            self._configure_gh_cli(account)
            
            # Save the state
            self.save_accounts()
            
            print(f"Switched to account: {username}")
            return True
            
        except Exception as e:
            print(f"Error switching account: {e}")
            return False
    
    def _configure_git(self, account: GitHubAccount) -> None:
        """Configure Git with account credentials"""
        commands = [
            ['git', 'config', '--global', 'user.name', account.username],
            ['git', 'config', '--global', 'user.email', account.email],
        ]
        
        for cmd in commands:
            subprocess.run(cmd, check=True)
        
        # Store credentials using git credential helper
        self._store_git_credentials(account)
    
    def _store_git_credentials(self, account: GitHubAccount) -> None:
        """Store Git credentials for HTTPS authentication"""
        # Configure credential helper
        subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=True)
        
        # Create credentials string
        credentials = f"https://{account.username}:{account.token}@github.com"
        
        # Store in git credentials
        cred_file = Path.home() / '.git-credentials'
        
        # Read existing credentials
        existing_creds = []
        if cred_file.exists():
            with open(cred_file, 'r') as f:
                existing_creds = 
        
        # Add new credentials
        existing_creds.append(credentials)
        
        # Write back
        with open(cred_file, 'w') as f:
            f.write('\n'.join(existing_creds) + '\n')
        
        # Secure the file
        os.chmod(cred_file, 0o600)
    
    def _configure_gh_cli(self, account: GitHubAccount) -> None:
        """Configure GitHub CLI if installed"""
        try:
            # Check if gh is installed
            result = subprocess.run(['which', 'gh'], capture_output=True, text=True)
            if result.returncode == 0:
                # Login with token
                subprocess.run(
                    ['gh', 'auth', 'login', '--with-token'],
                    input=account.token.encode(),
                    check=True
                )
        except:
            # gh CLI not installed or configuration failed
            pass
    
    def test_current_account(self) -> Dict:
        """Test the current account configuration"""
        account = self.get_active_account()
        if not account:
            return {"error": "No active account"}
        
        try:
            # Test GitHub API
            headers = {'Authorization': f'token {account.token}'}
            response = requests.get('https://api.github.com/user', headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                
                # Get current Git config
                git_user = subprocess.run(
                    ['git', 'config', '--global', 'user.name'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                git_email = subprocess.run(
                    ['git', 'config', '--global', 'user.email'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                return {
                    "github_username": user_data.get('login'),
                    "github_email": user_data.get('email'),
                    "git_config_user": git_user,
                    "git_config_email": git_email,
                    "status": "OK"
                }
            else:
                return {"error": f"API request failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize the manager
    manager = GitHubAccountManager()
    
    # Add accounts (you need to generate Personal Access Tokens from GitHub)
    # https://github.com/settings/tokens
    manager.add_account(
        username="your-username-1",
        email="email1@example.com",
        token="your-personal-access-token-1"
    )
    
    manager.add_account(
        username="your-username-2",
        email="email2@example.com",
        token="your-personal-access-token-2"
    )
    
    # List all accounts
    print("Available accounts:")
    for account in manager.list_accounts():
        print(f"  {account}")
    
    # Switch to an account
    manager.switch_account("your-username-1")
    
    # Test current configuration
    print("\nCurrent configuration:")
    test_result = manager.test_current_account()
    for key, value in test_result.items():
        print(f"  {key}: {value}")
    
    # Switch to another account
    manager.switch_account("your-username-2")