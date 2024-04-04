# Git Module

This Python module is used for executing Git operations with the assistance of the Commune Python package.

## Features

- Clone: Use the `clone` method to clone repositories from given URLs into specified directories.
- Submodule management: Using the `add_submodule` method, git submodules can be added.
- Git operations: execute common git operations like pull, push, status check, commit and hard reset.
- URL and Hash Fetch: fetch remote git repository URL and commit hash using `repo_url` and `commit_hash` methods respectively.
- Merge remote repositories: `merge_remote_repo` allows you to merge a remote repository with a local one.

## Usage

You can use this module to manage your git operations using Python. Here's an example:

```python
from Git import Git

Git.clone(repo_url='https://github.com/author/my-repo.git')
Git.add_submodule(url='https://github.com/author/another-repo.git', name='my_submodule')
status = Git.gstat()
Git.pull(stash=True)
commit_hash = Git.commit_hash()
Git.push(msg='My Commit Message')
Git.merge_remote_repo(remote_name='origin', remote_url='https://github.com/author/another-repo.git', remote_branch='my_branch', local_branch='my_local_branch')
```

## Key Classes and Methods

- `Git:` The main class, contains static methods for executing various git operations.
- `clone:` Clones a Git repository from a given URL into a target directory.
- `add_submodule:` Adds a git submodule from a given URL.
- `pull`, `push`, `gstat`, `commit`, `reset_hard`: These methods perform the git operations: pull, push, status check, commit, and hard reset respectively.
- `repo_url`, `commit_hash`: Fetch remote git repository URL and commit hash respectively.
- `merge_remote_repo`: Merge a remote branch with a local branch of a git repository.

## Requirements

- Python
- Commune Python package
- Git installed on the system

## Notice

This module allows operation of Git repositories. Ensure proper permissions are available before performing git operations. Be cautious of the potential to overwrite and loose work when manipulating repositories.