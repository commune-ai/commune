
import os
import glob
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple

def abspath(path):
    """
    Convert a path to an absolute path, expanding user directory (~).
    
    Args:
        path: Path to convert
        
    Returns:
        Absolute path
    """
    return os.path.abspath(os.path.expanduser(path))

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def detailed_error(e):
    """
    Extract detailed error information from an exception.
    
    Args:
        e: Exception object
        
    Returns:
        Dictionary with error details
    """
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    file_name = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'file_name': file_name.replace(os.path.expanduser('~'), '~'),
        'line_no': line_no,
        'line_text': line_text
    }   
    return response

def put_text(path, text):
    """
    Write text to a file.
    
    Args:
        path: Path to the file
        text: Text to write
        
    Returns:
        Dictionary with path and text
    """
    path = abspath(path)
    dirpath = os.path.dirname(path)
    makedirs(dirpath)  # Ensure the directory exists
    
    # Ensure the directory exists
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return {'path': path, 'text': text}

def get_text(path):
    """
    Read text from a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Text content of the file
    """
    if os.path.isdir(path):
        file2text = {}
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    path = os.path.abspath(file_path)
                    file2text[file_path] = get_text(file_path)
        return file2text
    else:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return None

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def list_files(directory, 
              pattern="*", 
              recursive=True, 
              ignore_patterns=None,
              max_size=None):
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        ignore_patterns: Patterns to ignore
        max_size: Maximum file size in bytes
        
    Returns:
        List of file paths
    """
    ignore_patterns = ignore_patterns or []
    
    if recursive:
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            # Filter out directories to ignore
            for ignore_pattern in ignore_patterns:
                dirnames[:] = [d for d in dirnames if not glob.fnmatch.fnmatch(d, ignore_pattern)]
            
            for filename in filenames:
                if glob.fnmatch.fnmatch(filename, pattern) and not any(
                    glob.fnmatch.fnmatch(filename, ignore) for ignore in ignore_patterns
                ):
                    file_path = os.path.join(root, filename)
                    if max_size is None or os.path.getsize(file_path) <= max_size:
                        matches.append(file_path)
        return matches
    else:
        return [
            f for f in glob.glob(os.path.join(directory, pattern))
            if os.path.isfile(f) and not any(
                glob.fnmatch.fnmatch(os.path.basename(f), ignore) for ignore in ignore_patterns
            ) and (max_size is None or os.path.getsize(f) <= max_size)
        ]

    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    file_name = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'file_name': file_name.replace(os.path.expanduser('~'), '~'),
        'line_no': line_no,
        'line_text': line_text
    }   
    return response


def detect_project_type(directory):
    """
    Attempt to detect the type of project in a directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Project type string or None if unknown
    """
    files = set(os.path.basename(f) for f in list_files(directory))
    
    # Check for various project types
    if 'package.json' in files:
        return 'node'
    elif 'requirements.txt' in files or 'setup.py' in files or any(f.endswith('.py') for f in files):
        if 'manage.py' in files and any('django' in get_text(os.path.join(directory, f)).lower() 
                                       for f in ['requirements.txt', 'setup.py'] if f in files):
            return 'django'
        elif 'app.py' in files and any('flask' in get_text(os.path.join(directory, f)).lower() 
                                     for f in ['requirements.txt', 'setup.py'] if f in files):
            return 'flask'
        elif 'main.py' in files and any('fastapi' in get_text(os.path.join(directory, f)).lower() 
                                      for f in ['requirements.txt', 'setup.py'] if f in files):
            return 'fastapi'
        return 'python'
    elif 'pom.xml' in files or 'build.gradle' in files:
        return 'java'
    elif 'Cargo.toml' in files:
        return 'rust'
    elif 'go.mod' in files:
        return 'go'
    elif 'Gemfile' in files or 'config/routes.rb' in files:
        return 'ruby_on_rails'
    elif 'composer.json' in files:
        return 'php'
    elif 'CMakeLists.txt' in files:
        return 'cmake'
    elif 'Makefile' in files:
        return 'make'
    elif 'docker-compose.yml' in files or 'Dockerfile' in files:
        return 'docker'
    
    return None

def calculate_file_hash(file_path):
    """
    Calculate the SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex digest of the hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def backup_directory(directory, backup_dir=None):
    """
    Create a backup of a directory.
    
    Args:
        directory: Directory to backup
        backup_dir: Destination directory (default: directory + '.bak')
        
    Returns:
        Path to the backup directory
    """
    directory = abspath(directory)
    if backup_dir is None:
        backup_dir = directory + '.bak'
    
    backup_dir = abspath(backup_dir)
    
    # Ensure the backup directory exists
    ensure_directory_exists(os.path.dirname(backup_dir))
    
    # Remove existing backup if it exists
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    # Create the backup
    shutil.copytree(directory, backup_dir)
    
    return backup_dir

def run_command(command, cwd=None, capture_output=True):
    """
    Run a shell command.
    
    Args:
        command: Command to run
        cwd: Working directory
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Dictionary with returncode, stdout, and stderr
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=capture_output,
            text=True
        )
        
        return {
            'returncode': result.returncode,
            'stdout': result.stdout if capture_output else '',
            'stderr': result.stderr if capture_output else '',
            'success': result.returncode == 0
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }

def find_files_by_content(directory, search_text, file_pattern="*", ignore_patterns=None):
    """
    Find files containing specific text.
    
    Args:
        directory: Directory to search
        search_text: Text to search for
        file_pattern: File pattern to match
        ignore_patterns: Patterns to ignore
        
    Returns:
        List of file paths containing the search text
    """
    matching_files = []
    files = list_files(directory, pattern=file_pattern, ignore_patterns=ignore_patterns)
    
    for file_path in files:
        try:
            content = get_text(file_path)
            if search_text in content:
                matching_files.append(file_path)
        except:
            # Skip files that can't be read as text
            pass
    
    return matching_files

def diff_files(file1, file2):
    """
    Compare two files and return the differences.
    
    Args:
        file1: Path to the first file
        file2: Path to the second file
        
    Returns:
        Differences as a string
    """
    file1_content = get_text(file1).splitlines()
    file2_content = get_text(file2).splitlines()
    
    import difflib
    differ = difflib.Differ()
    diff = list(differ.compare(file1_content, file2_content))
    
    return '\n'.join(diff)

def save_json(data, file_path):
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        file_path: Path to save to
        
    Returns:
        Path to the saved file
    """
    file_path = abspath(file_path)
    ensure_directory_exists(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return file_path

def load_json(file_path):
    """
    Load data from JSON.
    
    Args:
        file_path: Path to load from
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_file_info(file_path):
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    file_path = abspath(file_path)
    stat = os.stat(file_path)
    
    return {
        'path': file_path,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'created': stat.st_ctime,
        'extension': os.path.splitext(file_path)[1],
        'is_binary': is_binary_file(file_path)
    }

def is_binary_file(file_path, sample_size=1024):
    """
    Check if a file is binary.
    
    Args:
        file_path: Path to the file
        sample_size: Number of bytes to check
        
    Returns:
        True if the file is binary, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
        # Check for null bytes and high ratio of non-printable characters
        textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
        return bool(sample.translate(None, textchars))
    except:
        return True

