
from typing import *
import os

def str2python(input)-> dict:
    import json
    assert isinstance(input, str), 'input must be a string, got {}'.format(input)
    try:
        output_dict = json.loads(input)
    except json.JSONDecodeError as e:
        return input

    return output_dict



def get_folder_contents_advanced(url='commune-ai/commune.git', 
                                 host_url = 'https://github.com/',
                                 auth_token=None):
    try:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Python Script'
        }
        if not url.startswith(host_url):
            url = host_url + url
        
        if auth_token:
            headers['Authorization'] = f'token {auth_token}'
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse JSON response
        content = response.json()
        
        # If it's a GitHub API response, it will be a list of files/folders
        if isinstance(content, list):
            return json.dumps(content, indent=2)
        return response.text
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def file2hash(path='./'):
    import commune as c
    file2hash = {}
    import commune as c
    for k,v in c.file2text(path).items():
        file2hash[k] = c.hash(v)
    return file2hash



def locals2kwargs(locals_dict:dict, kwargs_keys=['kwargs'], remove_arguments=['cls','self']) -> dict:
    locals_dict = locals_dict or {}
    kwargs = locals_dict or {}
    for k in remove_arguments:
        kwargs.pop(k, None)
    assert isinstance(kwargs, dict), f'kwargs must be a dict, got {type(kwargs)}'
    # These lines are needed to remove the self and cls from the locals_dict
    for k in kwargs_keys:
        kwargs.update( locals_dict.pop(k, {}) or {})
    return kwargs



def walk(path='./', depth=2):
    import commune as c
    results = []
    if depth == 0:
        return results
    path = os.path.abspath(path)
    # break when it gets past 3 depths from the path file

    for subpath in c.ls(path):
        try:
            if os.path.isdir(subpath):
                results += walk(subpath, depth=depth-1)
            else:
                results += [subpath]
        except Exception as e:
            pass
    return results



def get_num_files( directory):
    num_files = 0
    for root, _, files in os.walk(directory):
        num_files += len(files)
    return num_files
    
def sizeof( obj):
    import sys
    result = 0
    if isinstance(obj, dict):
        for k,v in obj.items():
            result +=  sizeof(k) + sizeof(v)
    elif isinstance(obj, list):
        for v in obj:
            result += sizeof(v)
    else:
        result += sys.getsizeof(obj)
            
    return result



def mv(path1, path2):
    assert os.path.exists(path1), path1
    if not os.path.isdir(path2):
        path2_dirpath = os.path.dirname(path2)
        if not os.path.isdir(path2_dirpath):
            os.makedirs(path2_dirpath, exist_ok=True)
    shutil.move(path1, path2)
    assert os.path.exists(path2), path2
    assert not os.path.exists(path1), path1
    return {'success': True, 'msg': f'Moved {path1} to {path2}'}


def file2chars( path='./', fmt='b') -> int:
    import commune as c
    files = c.glob(path)
    file2size = {}
    file2size = dict(sorted(file2size.items(), key=lambda item: item[1]))
    return file2size

def find_largest_folder(directory: str = '~/'):
    directory = resolve_path(directory)
    """Find the largest folder in the given directory."""
    largest_size = 0
    largest_folder = ""

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            folder_size = get_folder_size(folder_path)
            if folder_size > largest_size:
                largest_size = folder_size
                largest_folder = folder_path

    return largest_folder, largest_size


def get_folder_size(folder_path:str='/'):
    folder_path = resolve_path(folder_path)
    """Calculate the total size of all files in the folder."""
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def file2size( path='./', fmt='b') -> int:
    import commune as c
    files = c.glob(path)
    file2size = {}
    for file in files:
        file2size[file] = format_data_size(filesize(file), fmt)
    file2size = dict(sorted(file2size.items(), key=lambda item: item[1]))
    return file2size

def cp(path1:str, path2:str, refresh:bool = False):
    import shutil
    # what if its a folder?
    assert os.path.exists(path1), path1
    if refresh == False:
        assert not os.path.exists(path2), path2
    
    path2_dirpath = os.path.dirname(path2)
    if not os.path.isdir(path2_dirpath):
        os.makedirs(path2_dirpath, exist_ok=True)
        assert os.path.isdir(path2_dirpath), f'Failed to create directory {path2_dirpath}'

    if os.path.isdir(path1):
        shutil.copytree(path1, path2)

    elif os.path.isfile(path1):
        
        shutil.copy(path1, path2)
    else:
        raise ValueError(f'path1 is not a file or a folder: {path1}')
    return path2

def get_folder_size( folder_path:str='/'):
    folder_path = os.path.abspath(folder_path)
    """Calculate the total size of all files in the folder."""
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def mv( path1, path2):
    import shutil
    assert os.path.exists(path1), path1
    if not os.path.isdir(path2):
        path2_dirpath = os.path.dirname(path2)
        if not os.path.isdir(path2_dirpath):
            os.makedirs(path2_dirpath, exist_ok=True)
    shutil.move(path1, path2)
    assert os.path.exists(path2), path2
    assert not os.path.exists(path1), path1
    return {'success': True, 'msg': f'Moved {path1} to {path2}'}

def rmtree( path):
    import shutil
    assert os.path.isdir(path), f'{path} is not a directory'
    return shutil.rmtree(path)
rmdir = rmtree 

def isdir( path):
    path = os.path.abspath(path=path)
    return os.path.isdir(path)
    
def isfile( path):
    path = os.path.abspath(path=path)
    return os.path.isfile(path)
    
def makedirs( *args, **kwargs):
    return os.makedirs(*args, **kwargs)



async def async_write(path, data,  mode ='w'):
    import aiofiles
    async with aiofiles.open(path, mode=mode) as f:
        await f.write(data)




def get_line(module, idx):
    import commune as c
    code = c.code(module)
    lines = code.split('\n')
    assert idx < len(lines), f'idx {idx} is out of range for {len(lines)}'  
    line =  lines[max(idx, 0)]
    return line

def find_lines(text:str, search:str) -> List[str]:
    """
    Finds the lines in text with search
    """
    found_lines = []
    lines = text.split('\n')
    for line in lines:
        if search in line:
            found_lines += [line]
    
    return found_lines
def file2lines(path:str='./')-> List[str]:
    result = file2text(path)
    return {f: text.split('\n') for f, text in result.items()}

def file2n(path:str='./')-> List[str]:
    result = file2text(path)
    return {f: len(text.split('\n')) for f, text in result.items()}

def munch( x:dict, recursive:bool=True)-> 'Munch':
    from munch import Munch
    '''
    Turn dictionary into Munch
    '''
    if isinstance(x, dict):
        for k,v in x.items():
            if isinstance(v, dict) and recursive:
                x[k] = dict2munch(v)
        x = Munch(x)
    return x 

dict2munch = munch

def put_yaml( path:str,  data: dict) -> Dict:
    import yaml
    from munch import Munch
    from copy import deepcopy
    import pandas as pd
    '''
    Loads a yaml file
    '''
    # Directly from dictionary
    data_type = type(data)
    if data_type in [pd.DataFrame]:
        data = data.to_dict()
    if data_type in [Munch]:
        data = munch2dict(deepcopy(data))
    if data_type in [dict, list, tuple, set, float, str, int]:
        yaml_str = yaml.dump(data)
    else:
        raise NotImplementedError(f"{data_type}, is not supported")
    with open(path, 'w') as file:
        file.write(yaml_str)
    return {'success': True, 'msg': f'Wrote yaml to {path}'}


def get_yaml( path:str=None, default={}, **kwargs) -> Dict:
    '''f
    Loads a yaml file
    '''
    import yaml
    path = os.path.abspath(path)
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data
load_yaml = get_yaml
save_yaml = put_yaml 


def search_files(path:str='./', search:str='__pycache__') -> List[str]:
    import commune as c
    path = os.path.abspath(path)
    files = c.glob(path)
    return list(filter(lambda x: search in x, files))


def lsdir( path:str) -> List[str]:
    path = os.path.abspath(path)
    return os.listdir(path)


def tilde_path():
    return os.path.expanduser('~')


def hidden_files(path:str='./')-> List[str]:
    import commune as c
    path = os.path.abspath(path)
    files = [f[len(path)+1:] for f in  c.glob(path)]
    hidden_files = [f for f in files if f.startswith('.')]
    return hidden_files


def wordinfolder( word:str, path:str='./')-> bool:
    import commune as c
    path = c.os.path.abspath(path)
    files = c.glob(path)
    progress = c.tqdm(len(files))
    for f in files:
        try:
            text = c.get_text(f)
        except Exception as e:
            continue
        if word in text:
            return True
        progress.update(1)
    return False

def resolve_path(path):
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path

def get_file_size( path:str):
    path = os.path.abspath(path)
    return os.path.getsize(path)
    

def get_files( path ='./', files_only:bool = True, recursive:bool=True):
    import glob
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path) and not path.endswith('**'):
        path = os.path.join(path, '**')
    paths = glob.glob(path, recursive=recursive)
    if files_only:
        paths =  list(filter(lambda f:os.path.isfile(f), paths))
    return sorted(paths)


def type2files( path:str='./', **kwargs):
    files = get_files(path, **kwargs)
    type2files = {}
    for f in files:
        if '.' in f:
            f_type = f.split('.')[-1]
            if f_type not in type2files:
                type2files[f_type] = []
            type2files[f_type].append(f)
    return type2files

def type2filecount( path:str='./', **kwargs):
    return {k: len(v) for k,v in type2files(path, **kwargs).items()}

def abspath(path:str):
    return os.path.abspath(os.path.expanduser(path))

def file2text(path = './', avoid_terms = ['__pycache__', 
                                '.git', 
                                '.ipynb_checkpoints', 
                                'package.lock',
                                'egg-info',
                                'Cargo.lock',
                                'artifacts',
                                'yarn.lock',
                                'cache/',
                                'target/debug',
                                'node_modules'],
                avoid_paths = ['~', '/tmp', '/var', '/proc', '/sys', '/dev'],
                relative=True,  **kwargs):
    
    path = os.path.abspath(os.path.expanduser(path))
    assert all([not os.path.abspath(k) in path for k in avoid_paths]), f'path {path} is in avoid_paths'
    file2text = {}
    for file in get_files(path, recursive=True, avoid_terms=avoid_terms , **kwargs):
        if os.path.isdir(file):
            continue
        try:
            with open(file, 'r') as f:
                content = f.read()
                file2text[file] = content
        except Exception as e:
            continue
    if relative:
        return {k[len(path)+1:]:v for k,v in file2text.items()}
    return file2text


def cp( path1:str, path2:str, refresh:bool = False):
    import shutil
    # what if its a folder?
    assert os.path.exists(path1), path1
    if refresh == False:
        assert not os.path.exists(path2), path2
    
    path2_dirpath = os.path.dirname(path2)
    if not os.path.isdir(path2_dirpath):
        os.makedirs(path2_dirpath, exist_ok=True)
        assert os.path.isdir(path2_dirpath), f'Failed to create directory {path2_dirpath}'

    if os.path.isdir(path1):
        shutil.copytree(path1, path2)

    elif os.path.isfile(path1):
        
        shutil.copy(path1, path2)
    else:
        raise ValueError(f'path1 is not a file or a folder: {path1}')
    return path2


def path2text( path:str, relative=False):
    import glob
    path = os.path.abspath(path)
    assert os.path.exists(path), f'path {path} does not exist'
    if os.path.isdir(path):
        filepath_list = glob.glob(path + '/**')
    else:
        assert os.path.exists(path), f'path {path} does not exist'
        filepath_list = [path] 
    path2text = {}
    for filepath in filepath_list:
        try:
            path2text[filepath] = get_text(filepath)
        except Exception as e:
            pass
    if relative:
        pwd = pwd()
        path2text = {os.path.relpath(k, pwd):v for k,v in path2text.items()}
    return path2text


def textsize( path: str = './', **kwargs ) -> str:
    return len(str(cls.text(path)))