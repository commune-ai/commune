import commune as c
import os

class Repo(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(locals())

    def is_repo(self, path):
        # is a directory 
        cond = [
            os.path.isdir(path), 
            os.path.isdir(os.path.join(path, '.git')),
            os.path.isfile(os.path.join(path, '.git/config'))
        ]
        return all(cond)
    
    def find_repo_paths(self, path = None, avoid_strings = ['.cargo']):
        if path == None:
            path = c.home_path
        repos = []
        c.print(path)
        for root, dirs, files in os.walk(path):
            if any([avoid in root for avoid in avoid_strings]):
                continue
            if '.git' in dirs and os.path.isfile(os.path.join(root, '.git/config')):
                repos.append(root)
        return repos
    



    @classmethod
    def dashboard(cls):
        import streamlit as st
        import pandas as pd
        self = cls()
        c.load_style()
        update_button = st.button('Update')
        if update_button:
            c.submit(self.update)
        repo2path = self.repo2path()
        repos = list(repo2path.keys())
        repo = st.selectbox('Repo', repos)
        repo_pull_button = st.button('pull')
        if repo_pull_button:
            st.write('pulling')
            st.write(repo)
            c.submit(self.pull_repo, args=[repo])
        repo_path = repo2path[repo]
        tabs = ['explorer', 'manager']
        tabs = st.tabs(tabs)
        with tabs[0]:
            self.repo_explorer(repo_path)
        with tabs[1]:
            self.repo_manager()
        st.write(repo_path)

    def git_files(self, repo):
        repo_path = self.repo2path()[repo]
        return c.glob(repo_path+'/.git/**/*')

    def submodules(self, repo):
        repo_path = self.repo2path()[repo]
        submodules = c.ls(repo_path+'/.git/modules')
        submodules = [os.path.basename(s) for s in submodules if os.path.basename(s) != 'config']
        return submodules
    
    def repo2submodules(self):
        repo2submodules = {}
        for repo in self.repos():
            repo2submodules[repo] = self.submodules(repo)
        return repo2submodules
    

    
    def repo_explorer(self, repo_path):
        
        repo_files = c.glob(repo_path)
        readme_files = [file for file in repo_files if 'readme' in file.lower()]
        with st.expander('files'):
            selected_files = st.multiselect('files', repo_files)
            file2text = { file: c.get_text(file) for file in selected_files}

            for file, text in file2text.items():
                st.write(file)
                st.markdown(text)

        with st.expander('readme', True):
            if len(readme_files) == 0:
                c.print('No readme found')

            readme_text = c.get_text(readme_files[0])
            st.write(readme_text)




    def rm_repo(self, repo):
        repo_path = self.repo2path()[repo]
        c.rm(repo_path)
        self.update()
        repos = self.repos()
        assert repo not in repos
        return {'success': True, 'path': repo_path, 'repo': repo, 
                'msg': f'Repo {repo} removed successfully'}


    def repo_paths(self):
        return list(self.repo2path().values())
    def add_repo(self, repo_path, path = None, cwd = None, sudo=False):
        repo_name =  os.path.basename(repo_path).replace('.git', '')
        if path == None:
            path = os.path.abspath(os.path.expanduser('~/'+ repo_name))
        assert not os.path.isdir(path), f'{path} already exists'

        c.cmd(f'git clone {repo_path}', verbose=True, cwd=cwd, sudo=sudo)


        repo_paths = self.repo_paths()

        assert path in repo_paths

        return {'success': True, 'path': path, 'repo_path': repo_path}
    
    def repos(self, *args, **kwargs):
        return list(self.repo2path(*args, **kwargs).keys())
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
            repo = cols[0].selectbox('Repo', list(self.repo2path().keys()), key='remove_repo')
            remove_repo_button = st.button('Remove Repo')
            if remove_repo_button:
                self.remove_repo(repo)

        

    def pull_repo(self, repo):
        repo_path = self.repo2path()[repo]
        return c.cmd(f'git pull', cwd=repo_path)
    

        
    def pull_many(self, *repos, timeout=20):
        futures = []
        for repo in repos:
            futures.append(c.submit(self.pull_repo, args=[repo], timeout=timeout))
        return c.wait(futures, timeout=timeout)

