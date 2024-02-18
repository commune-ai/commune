import commune as c
import os
class ModuleManager(c.Module):



    @classmethod
    def new_module( cls,
                   module : str = None,
                   repo : str = None,
                   base : str = 'base',
                   code : str = None,
                   include_config : bool = False,
                   overwrite : bool  = False,
                   module_type : str ='dir'):
        """ Makes directories for path.
        """
        if module == None: 
            assert repo != None, 'repo must be specified if module is not specified'
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        module_path = 'path'
        module = module.replace('.','/')
        if c.has_module(module) and overwrite==False:
            return {'success': False, 'msg': f' module {module} already exists, set overwrite=True to overwrite'}
        
        # add it to the root
        module_path = os.path.join(c.modules_path, module)
        
        if overwrite and c.module_exists(module_path): 
            c.rm(module_path)

        
        if repo != None:
            # Clone the repository
            c.cmd(f'git clone {repo} {module_path}')
            # Remove the .git directory
            c.cmd(f'rm -rf {module_path}/.git')

        # Create the module name if it doesn't exist, infer it from the repo name 
        if module == None:
            assert repo != None, 'repo must be specified if module is not specified'
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        
        # currently we are using the directory name as the module name
        if module_type == 'dir':
            c.mkdir(module_path, exist_ok=True)
        else:
            raise ValueError(f'Invalid module_type: {module_type}, options are dir, file')
        

        if code == None:
            base_module = c.module(base)
            code = base_module.code()

            
        module = module.replace('/','_') # replace / with _ for the class name
        
        # define the module code and config paths
        
        module_code_path =f'{module_path}/{module}.py'
        module_code_lines = []
        class_name = module[0].upper() + module[1:] # capitalize first letter
        class_name = ''.join([m.capitalize() for m in module.split('_')])
        
        # rename the class to the correct name 
        for code_ln in code.split('\n'):
            if all([ k in code_ln for k in ['class','c.Module', ')', '(']]) or all([ k in code_ln for k in ['class','commune.Module', ')', '(']]):
                indent = code_ln.split('class')[0]
                code_ln = f'{indent}class {class_name}(c.Module):'
            module_code_lines.append(code_ln)
            
        c.put_text(module_code_path, code)
        if include_config:
            module_config_path = module_code_path.replace('.py', '.yaml')
            c.save_yaml(module_config_path, {'class_name': class_name})
        
        c.module_tree(update=True)

        return {'success': True, 'msg': f' created a new repo called {module}'}
      