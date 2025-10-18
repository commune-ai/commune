
import json
import commune as c
import os


def expanduser(path):
    return os.path.expanduser(path)

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_json(file_path, data):
    dir_path = os.path.dirname(file_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def logs(name):
    return c.logs(name)
    