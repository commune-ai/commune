# 🚀 Commune Git Module

[![GitHub stars](https://img.shields.io/github/stars/commune-ai/commune?style=social)](https://github.com/commune-ai/commune)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

The Commune Git Module is a powerful, Pythonic interface for Git operations, designed for seamless integration with the Commune framework. This module provides a comprehensive set of tools for managing Git repositories, tracking changes, and automating Git workflows.

## 🔥 Features

- **Repository Management**: Initialize, clone, and manage Git repositories
- **Commit Analysis**: Track changes, analyze commit history, and visualize contributions
- **Branch Operations**: Create, switch, and manage branches with ease
- **Remote Operations**: Push, pull, and synchronize with remote repositories
- **Diff Analysis**: Analyze code changes with detailed diff information
- **Commit Visualization**: Generate insights from commit history with Pandas integration

## 🛠️ Installation

```bash
pip install commune
```

## 🚀 Quick Start

```python
import commune as c

# Initialize Git module
git = c.module('git')

# Get repository information
repo_info = git.get_info('./my_project')
print(f"Repository URL: {repo_info['url']}")
print(f"Current branch: {repo_info['branch']}")
print(f"Current commit: {repo_info['commit']}")

# Analyze commit history
commit_df = git.commits('./my_project', n=10)
print(commit_df)

# Push changes
git.push('./my_project', msg='Update documentation')
```

## 📊 Commit Analysis

The Git module provides powerful tools for analyzing commit history:

```python
# Get detailed commit history with additions and deletions
commit_history = git.commits(
    path='./my_project',
    n=20,  # Number of commits to analyze
    features=['date', 'author', 'message', 'additions', 'deletions']
)

# Visualize commit activity
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(commit_history['date'], commit_history['additions'], label='Additions')
plt.bar(commit_history['date'], -commit_history['deletions'], label='Deletions')
plt.legend()
plt.title('Code Changes Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 🔄 Diff Analysis

```python
# Get file diffs in the repository
diffs = git.diff('./my_project')

# Analyze changes by file
diff_df = git.dff('./my_project')
print(diff_df.sort_values('additions', ascending=False))
```

## 🌟 Advanced Usage

### Initialize a new repository

```python
git.init_repo(
    repo_path='./new_project',
    user_name='Your Name',
    user_email='your.email@example.com',
    initial_branch='main'
)
```

### Push a folder as a new branch

```python
git.push_folder_as_new_branch(
    folder_path='./feature_code',
    repo_url='https://github.com/username/repo.git',
    branch_name='feature-x',
    commit_message='Add feature X implementation'
)
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to help improve the Git module.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ by the [Commune](https://github.com/commune-ai/commune) team.
