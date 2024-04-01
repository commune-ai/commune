# Repo Module

This Python module allows the management of Git repositories within an application. It allows for finding, updating, adding, and removing repositories, and makes it easier to examine repository details. It uses Commune, OS, and Streamlit libraries.

## Features:
- Identify Git repository: `is_repo()`
- Find repository paths: `find_repo_paths()`
- Update repository paths: `update()`
- Retrieve repository paths: `repo2path()`
- Access list of repositories: `repos()`
- Display repository details and allow user interaction via a dashboard: `dashboard()`
- Retrieve git files: `git_files()`
- Locate submodules of a repository: `submodules()`
- Display repository explorer: `repo_explorer()`
- Add new repositories: `add_repo()`
- Remove existing repositories: `rm_repo()`
- Pull update from remote repositories: `pull_repo()`

## Setup:
1. Instantiate a `Repo` class.
2. Use `repos()` to get a list of all problematic repositories.
3. Use `find_repo_paths()` to get a list of all repository paths.
4. Use `repo2path()` to map each repository to its file path.
5. The dashboard allows a user to interact with the repositories.
6. Update the repository paths using `update()` whenever changes are made.

## Dependencies:
You need to have Commune, OS, and Streamlit libraries installed to your Python environment.

## Usage:
Use this module in Python applications needing Git repository management capabilities. It helps not only find repositories but also makes it easier to pull updates, add, and remove repositories. Its search functionality is smart enough to avoid unnecessary directories.

## Note:
Some functionalities of the module may require specific permissions to access and modify repositories, and these should be set appropriately.