import commune as c
import os
import streamlit as st

class Repo(c.Module):
    home_dir = os.path.expanduser("~")
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def is_repo(self, path):
        # is a directory 
        assert os.path.isdir(path)
        # has a .git directory
        assert os.path.isdir(os.path.join(path, '.git'))
        # has a .git/config file
        assert os.path.isfile(os.path.join(path, '.git/config'))
        return True
    
    def find_repo_paths(self, path = None, avoid_strings = ['.cargo']):
        if path == None:
            path = self.home_dir
        repos = []
        c.print(path)
        for root, dirs, files in os.walk(path):
            if any([avoid in root for avoid in avoid_strings]):
                continue
            if '.git' in dirs and os.path.isfile(os.path.join(root, '.git/config')):
                repos.append(root)
        return repos
    
    def update(self):
        self.repo2path(update=True)
    
    def repo2path(self, repo = None, update=False, repo_path='repo2path'):
        repo2path = {} if update else self.get(repo_path, {}) 
        if len(repo2path) > 0:
            return repo2path
        find_repo_paths = self.find_repo_paths()
        for path in find_repo_paths:
            repo_name = path.split('/')[-1]
            repo2path[repo_name] = path
        self.put(repo_path, repo2path)
        if repo != None:
            return repo2path[repo]
        return repo2path
    


    @classmethod
    def dashboard(cls):
        import streamlit as st
        import pandas as pd
        self = cls()
        c.load_style()
        update_button = st.button('Update')
        if update_button:
            c.submit(self.update)
        self.repo2path = self.repo2path()
        repos = list(self.repo2path.keys())
        repo = st.selectbox('Repo', repos)
        repo_path = self.repo2path[repo]
        tabs = ['explorer', 'manager']
        tabs = st.tabs(tabs)
        with tabs[0]:
            self.repo_explorer(repo_path)
        with tabs[1]:
            self.repo_manager()
        st.write(repo_path)
        
    def repo_explorer(self, repo_path):
        
        repo_files = os.listdir(repo_path)
        with st.expander('files'):
            st.write(repo_files)
        with st.expander('readme', True):
            readme_paths = [f for f in repo_files if 'readme' in f.lower() and '.md' in f.lower()]
            if len(readme_paths) == 0:
                c.print('No readme found')
            for readme_path in readme_paths:
                readme_text = c.get_text(os.path.join(repo_path, readme_path))
                st.write(readme_text)

    def add_repo(self, repo_path, path = None):
        path = path or repo_path.split('/')[-1]
        c.cmd(f'git clone {repo_path} {path}')
        self.update()
        return {'success': True, 'path': path, 'repo_path': repo_path}
    
    def repo_manager(self):
        with st.expander('Add Repo'):
            cols = st.columns(2)
            repo_path = cols[0].text_input('Repo Path')
            repo_name = repo_path.split('/')[-1] if '/' in repo_path else None
            repo_name = cols[1].text_input('Repo Name', value=repo_name)
            add_repo_button = st.button('Add Repo')
            if add_repo_button:
                self.add_repo(repo_path, repo_name)

        with st.expander('Remove Repo'):
            cols = st.columns(2)
            repo = cols[0].selectbox('Repo', list(self.repo2path.keys()))
            remove_repo_button = st.button('Remove Repo')
            if remove_repo_button:
                self.remove_repo(repo)


Repo.run(__name__)