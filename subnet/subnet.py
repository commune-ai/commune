
import commune as c
import os


class Subnet(c.Module):
    


    def names(self):
        return list(self.state().keys())
    
    def subnet2urls(self):
        return list(map(lambda x: x['url'], self.state().values()))
    
    
    

    def subnets(self):
        subnet_paths =  c.ls(self.dirpath())
        subnet_paths = list(filter(lambda x: os.path.isdir(x) and not x.endswith('__'), subnet_paths) )
        info_map = {}
        for path in subnet_paths:
            # get the git url from each one .git
            subnet_name = path.split('/')[-1] 
            response = c.cmd('git remote -v', cwd=path ).split('\n')[0]
            branch = response.split('\t')[0]
            url = response.split('\t')[1].split(' ')[0]
            c.print(f'Branch={branch} Url={url} Name={subnet_name}')
            subnet_info =  {
                'name': subnet_name,
                'url': url,
                'branch': branch,
                'path': path
            }
            info_map[subnet_name] = subnet_info
        return info_map
    
    def add_subnet(self, url, name=None, 
                   url_prefix='https://github.com/', 
                   url_suffix='.git'):

        #
        # Add the prefix to the url if it does not have it
        if not url.startswith(url_prefix):
            url = url_prefix + url
        if not url.endswith(url_suffix):
            url = url + url_suffix

        
        if name is None:
            name = url.split('/')[-1].split('.')[0]
        return c.cmd(f'git clone {url} {name}', cwd=self.dirpath())
    

    def pull_subnet(self, name):
        return c.cmd(f'git pull', cwd=self.dirpath(name))
    

    def state_path(self):
        return self.dirpath() + '/subnets.json'
    

    def subnet2files(self):
        subnet2files = {}
        for name, info in self.state().items():
            path = info['path']
            files = c.ls(path)
            files = list(filter(lambda x: os.path.isfile(x), files))
            subnet2files[name] = files
        return subnet2files
    

    def subnet2readmefile(self):
        subnet2readmefile = {}
        for name, info in self.state().items():
            for f in c.ls(info['path']):
                if f.lower().endswith('.md'):
                    subnet2readmefile[name] = f
                    break
        return subnet2readmefile
    
    def save_state(self):
        return c.save_json(self.state_path(), self.state())

    
    def load_state(self, max_age=60, update=False, **kwargs):
        state = self.get_json(self.state_path(), None, max_age=max_age, update=update, **kwargs)
        if state == None: 
            state = self.state()
            self.put_json(state)

        


    

