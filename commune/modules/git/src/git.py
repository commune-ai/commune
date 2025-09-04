import commune as c
import subprocess
import requests
import base64
import re
import pandas as pd
import os
import hashlib
import os
import json
from pathlib import Path

class Git:

    def __init__(self, path:str = './'):
        self.path = os.path.abspath(os.path.expanduser(path))
    def is_repo(self, path:str ):
        # has the .git folder
        return c.cmd(f'ls -a {path}').count('.git') > 0

    def push(self, path=None, msg:str='update', safety=False):
        path = self.get_path(path)
        cmds = ['git add .', f'git commit -m "{msg}"', 'git push']
        if safety:
            # check if the commands are safe to run
            if input(f'Do you want to run these cmds {cmds}?') != 'y':
                return {'status': 'cancelled', 'cmds': cmds}
        for cmd in cmds:
            c.cmd(cmd, cwd=cwd)
        return {
            'status': 'success',
            'cmds': cmds,
        }

    def git_repos(self, path='./'):
            import os
            repos = []
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    if d.endswith('.git'):
                        repos +=  [f"{root}"]

            repos = [r for r in repos if not r == path]

            return repos

    def repo_url(self, path:str = None) -> str:
        lpath = self.get_path(path)
        return c.cmd('git remote -v',cwd=path, verbose=False).split('\n')[0].split('\t')[1].split(' ')[0]
    
    def hash(self, path:str = None):
        path = self.get_path(path)
        return c.cmd('git rev-parse HEAD', cwd=path, verbose=False).split('\n')[0].strip()

    def history(self, path:str = None):
        """
        a history ofo the commits and the times
        """
        path = self.get_path(path)
        result = c.cmd('git log', cwd=path, verbose=False)
        return result

    def reset_hard(self, path:str = None):
        path = self.get_path(path)
        return c.cmd('git reset --hard', cwd=path, verbose=False)
    
    def get_path(self, path:str = None):
        if path == None:
            path = self.path
        return path

    def commit_hash(self, lib_path:str = None):
        if lib_path == None:
            lib_path = self.lib_path
        return c.cmd('git rev-parse HEAD', cwd=lib_path, verbose=False).split('\n')[0].strip()
    
    def get_info(self, path:str = None, name:str = None, n=10):
        path = path or c.mods_path
        git_path = path + '/.git'
        git_url = c.cmd('git config --get remote.origin.url', cwd=path).strip().split('\n')[0].strip().split(' ')[0].strip()
        git_branch = c.cmd('git rev-parse --abbrev-ref HEAD', cwd=path).strip().split('\n')[0].strip().split(' ')[0].strip()
        git_commit = c.cmd('git rev-parse HEAD', cwd=path)
        git_commit = git_commit.split('\n')[0].strip().split(' ')[0].strip()
        past_commits = c.cmd('git log --oneline', cwd=path).split('\n')
        # get all of the info of each commit
        past_commits = past_commits[:n]
        commit2comment = {}
        for co in past_commits:
            if len(co) == 0:
                continue
            commit, comment = co.split(' ', 1)
            commit2comment[commit] = comment
        diff = self.diff(path)
        return {
            'url': git_url,
            'branch': git_branch,
            'commit': git_commit,
            'past_commits': commit2comment,
            'diff': diff,
        }
        
    def commits(self, path: str = None, n: int = 10,features=['date', 'additions', 'deletions']) -> pd.DataFrame:
        """
        Get a DataFrame of commits with comment, time, and number of additions/deletions.
        
        Args:
            path: Path to the git repository. Defaults to self.path.
            n: Number of commits to retrieve. Defaults to 10.
            
        Returns:
            pandas DataFrame with commit information.
        """
        import pandas as pd
        
        path = self.get_path(path)
        
        # Get commit hashes, authors, dates, and messages
        log_format = "%H|%an|%ad|%s"
        log_cmd = f'git log --pretty=format:"{log_format}" -n {n}'
        log_result = c.cmd(log_cmd, cwd=path, verbose=False).split('\n')
        
        # Get stats for each commit
        stats_cmd = f'git log --numstat --pretty=format:"%H" -n {n}'
        stats_result = c.cmd(stats_cmd, cwd=path, verbose=False).split('\n')
        
        # Process the results
        commits = []
        current_commit = None
        additions = 0
        deletions = 0
        
        for line in stats_result:
            if line.strip():
                if len(line) == 40:  # This is a commit hash
                    current_commit = line
                    additions = 0
                    deletions = 0
                else:
                    # This is a stats line
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        try:
                            additions += int(parts[0]) if parts[0] != '-' else 0
                            deletions += int(parts[1]) if parts[1] != '-' else 0
                        except ValueError:
                            pass
                        
                        # Store the stats for this commit
                        for log_line in log_result:
                            if log_line.startswith(current_commit):
                                hash_val, author, date, message = log_line.split('|', 3)
                                if hash_val == current_commit:
                                    commits.append({
                                        'hash': hash_val,
                                        'author': author,
                                        'date': date,
                                        'message': message,
                                        'additions': additions,
                                        'deletions': deletions
                                    })
                                    break
        
        # Create DataFrame
        if not commits:
            # Alternative approach if the above didn't work
            commits = []
            for line in log_result:
                if not line.strip():
                    continue
                hash_val, author, date, message = line.split('|', 3)
                # Get stats for this specific commit
                stat_cmd = f'git show --numstat --format="%h" {hash_val}'
                stat_output = c.cmd(stat_cmd, cwd=path, verbose=False).split('\n')
                
                add = 0
                delete = 0
                for stat_line in stat_output[1:]:  # Skip the first line which is the commit hash
                    parts = stat_line.strip().split('\t')
                    if len(parts) == 3:
                        try:
                            add += int(parts[0]) if parts[0] != '-' else 0
                            delete += int(parts[1]) if parts[1] != '-' else 0
                        except ValueError:
                            pass
                
                commits.append({
                    'hash': hash_val,
                    'author': author,
                    'date': date,
                    'message': message,
                    'additions': add,
                    'deletions': delete
                })

        
        df =  pd.DataFrame(commits)
        # remove duplicate hashes and add the additions and deletionsa and group by the hash

        df = df.groupby('hash').agg({
            'author': 'first',
            'date': 'first',
            'message': 'first',
            'additions': 'sum',
            'deletions': 'sum'
        }).reset_index()

        # order by date
        df = df.sort_values('date', ascending=False)
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Calcutta')
        print('Repo(url={url}, branch={branch}, commit={commit})'.format( 
            url=self.repo_url(path),
            branch=self.branch(path),
            commit=self.commit_hash(path)
        ))
        return df[features]
    
    def gitpath(self, path, **kwargs) -> str:
        """
        find th
        e github url of a module
        """
        path = os.path.abspath(os.path.expanduser(path))
        git_path = path + '/.git'
        if os.path.exists(git_path):
            return git_path
        return None

    def branches(self, path:str = None):
        """
        Get a list of branches in the git repository.
        
        """

        path = self.get_path(path)
        # get the branches
        branches = c.cmd('git branch', cwd=path, verbose=False).split('\n')
        # remove the * from the current branch
        branches = [b.replace('*', '').strip() for b in branches]
        # remove empty lines
        branches = [b for b in branches if len(b) > 0]
        return branches

    def init(self, path:str = None, name:str = None):
        """
        Initialize a git repository at the given path.
        
        Args:
        """
        return c.cmd('git init', cwd=path, verbose=False)

    def giturl(self, url:str='commune-ai/commune'):
        gitprefix = 'https://github.com/'
        gitsuffix = '.git'
        if not url.startswith(gitprefix):
            url = gitprefix + url
        if not url.endswith(gitsuffix):
            url = url + gitsuffix
        return url

    def diff(self, path:str = None, relative=False):
        """
        Get the diff of files in a git repository.
        
        Args:
            path: Path to the git repository. Defaults to self.path.
            relative: If True, returns relative paths. If False, returns absolute paths.
            
            
        Returns:
            Dictionary mapping file paths to their diffs.
        """
        if path is None:
            path = self.path
        
        # Run git diff command
        response = c.cmd('git diff', cwd=path, verbose=False)
        return response
    def dff(self, path='./'):
        diff = self.diff(path)
        df  = []
        for k,v in diff.items():
            df.append({
                'path': k,
                'additions': len(re.findall(r'\+', v)),
                'deletions': len(re.findall(r'-', v)),
                # 'hash': c.hash(v),
            })

        df = pd.DataFrame(df)
        return df
            
    def branch(self, path=None):
        path = self.get_path(path)
        return c.cmd('git rev-parse --abbrev-ref HEAD', cwd=path).strip().split('\n')[0].strip().split(' ')[0].strip()

    def init_repo(
        repo_path: str, 
        user_name: str = None, 
        user_email: str = None,
        initial_branch: str = "main",
        verbose: bool = False
    ) -> str:
        """
        Initialize a Git repository with optional configuration.
        
        Args:
            repo_path (str): Path where the Git repository should be initialized
            user_name (str, optional): Git user name for this repository
            user_email (str, optional): Git user email for this repository
            initial_branch (str, optional): Name of the initial branch
            verbose (bool): Whether to print command output
            
        Returns:
            str: Command output
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(repo_path, exist_ok=True)
        
        # Initialize the repository with specified branch name
        result = cmd(f"git init -b {initial_branch}", cwd=repo_path, verbose=verbose)
        
        # Configure user if provided
        if user_name:
            cmd(f"git config user.name '{user_name}'", cwd=repo_path, verbose=verbose)
        
        if user_email:
            cmd(f"git config user.email '{user_email}'", cwd=repo_path, verbose=verbose)
        
        # Create initial README and commit
        with open(os.path.join(repo_path, "README.md"), "w") as f:
            f.write(f"# {os.path.basename(repo_path)}\n\nInitialized with Python.")
        
        cmd("git add README.md", cwd=repo_path, verbose=verbose)
        cmd("git commit -m 'Initial commit'", cwd=repo_path, verbose=verbose)
        
        return result


    def push_folder_as_new_branch(folder_path, repo_url, branch_name, commit_message="Initial commit"):
        """
        Push a folder as a new branch to a Git repository without initializing a new repo.
        
        Args:
            folder_path (str): Path to the folder to push
            repo_url (str): URL of the Git repository
            branch_name (str): Name of the new branch to create
            commit_message (str, optional): Commit message. Defaults to "Initial commit".
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            folder_path = Path(folder_path).resolve()
            
            # Create a temporary directory for git operations
            temp_dir = folder_path.parent / f"temp_git_{branch_name}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Initialize git in the temp directory
            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            
            # Set the remote
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=temp_dir, check=True)
            
            # Create and checkout an orphan branch (no history)
            subprocess.run(["git", "checkout", "--orphan", branch_name], cwd=temp_dir, check=True)
            
            # Copy all files from the folder to the temp directory
            for item in folder_path.glob('**/*'):
                if item.is_file():
                    # Create relative path
                    rel_path = item.relative_to(folder_path)
                    # Create target directory if it doesn't exist
                    target_dir = temp_dir / rel_path.parent
                    os.makedirs(target_dir, exist_ok=True)
                    # Copy the file
                    with open(item, 'rb') as src, open(temp_dir / rel_path, 'wb') as dst:
                        dst.write(src.read())
            
            # Add all files
            subprocess.run(["git", "add", "."], cwd=temp_dir, check=True)
            
            # Commit
            subprocess.run(["git", "commit", "-m", commit_message], cwd=temp_dir, check=True)
            
            # Push to remote
            subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=temp_dir, check=True)
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
            return True
        
        except Exception as e:
            print(f"Error: {e}")
            return False


    def image_id(self, dockerfile_path='~/commune/Dockerfile', context_path=None):
        """
        Create a unique image ID based on Dockerfile content and build context.
        
        Args:
            dockerfile_path: Path to the Dockerfile
            context_path: Path to the build context directory (optional)
        
        Returns:
            str: A unique image ID (SHA256 hash)
        """
        dockerfile_path = os.path.expanduser(dockerfile_path)
        hasher = hashlib.sha256()
        
        # Hash the Dockerfile content
        with open(dockerfile_path, 'rb') as f:
            hasher.update(f.read())
        # Generate the image ID
        return f"sha256:{hasher.hexdigest()}"