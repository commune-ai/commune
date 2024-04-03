import commune as c
from flask import Flask, jsonify, request
import random
from datetime import datetime, timedelta
import pandas as pd

from flask_cors import CORS

app = Flask(__name__)

CORS(app) 

# Function to generate dummy data for ID, email, and date
def generate_dummy_data():
    unique_id = '#' + ''.join(random.choices('0123456789', k=7))

    email = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10)) + '@example.com'

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    random_date = start_date + (end_date - start_date) * random.random()
    formatted_date = random_date.strftime('%d %b, %Y')

    return unique_id, email, formatted_date

# Endpoint to get the list of modules
@app.route('/modules', methods=['GET'])
def get_modules():
    modules = c.modules()

    # Generate dummy data for each module
    formatted_modules = []
    for module in modules:
        unique_id, email, formatted_date = generate_dummy_data()
        formatted_module = {
            "checkbox": "",
            "id": unique_id,
            "Name": module,
            "Email": email,
            "Date": formatted_date,
            "status": "completed",  
            "trash": ""
        }
        formatted_modules.append(formatted_module)

    return jsonify(formatted_modules)

# Endpoint to get metadata for a particular module
@app.route('/modules/metadata', methods=['GET'])
def get_module_metadata():
    # Retrieve module name from query parameter
    module_name = request.args.get('module_name')

    if not module_name:
        return jsonify({'error': 'Module name not provided in query parameter'}), 400


    module_metadata = {}

    try:
        module = c.module(module_name)
        # module_metadata['code'] = module.code()
        module_metadata['config'] = module.config()
        module_metadata['functions'] = module.fns()
        module_metadata['schema'] = module.schema()
    except Exception as e:
        return jsonify({'error': str(e)}), 404

    return jsonify(module_metadata)

@app.route('/modules/trees', methods=['GET'])
def list_modules():
    module_name = request.args.get('module_name', None)
    module = c.module()

    try:
        if module_name:
            modules_tree = module.tree(module_name)
        else:
            modules_tree = module.tree()
        return jsonify(modules_tree)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/modules/keys', methods=['GET'])
def module_keys():
    module_name = request.args.get('module_name', None)  

    try:
        if module_name:
            keys = c.module(module_name).keys()
        else:
            keys = c.keys()
        return jsonify(keys)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/modules/active_thread_count', methods=['GET'])
def module_active_thread_count():
    module_name = request.args.get('module_name')
    if not module_name:
        return jsonify({'error': 'Module name not provided in query parameter'}), 400

    try:
        active_thread_count = c.module(module_name).active_thread_count()
        return jsonify({"active_thread_count": active_thread_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/modules/users', methods=['GET'])
def module_users(): # may need to be checked 
    module_name = request.args.get('module_name')
    if not module_name:
        return jsonify({'error': 'Module name not provided in query parameter'}), 400

    try:
        users = c.module(module_name).users()
        return jsonify(users)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/modules/namespaces', methods=['GET'])
def namespaces():
    module_name = request.args.get('module_name', None)
    module = c.module()
    try:
        if module_name:
            namespaces_list = module.namespace(module_name)
        else:
            namespaces_list = module.namespace()
        return jsonify(namespaces_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# returns list of modules being ACTIVELY served 
@app.route('/modules/servers', methods=['GET'])
def list_servers():
    module_name = request.args.get('module_name')  
    
    try:
        if module_name:
            served_modules = c.servers(module_name)
        else:
            served_modules = c.servers()
        return jsonify(served_modules)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# @app.route('/modules/info', methods=['GET'])
# def get_module_info():
#     module_name = request.args.get('module_name')

#     if not module_name:
#         return jsonify({'error': 'Module name not provided in query parameter'}), 400

#     try:
#         module = c.module(module_name)  # Adjusted this line
#         module_info = module.info() 
#         return jsonify(module_info)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 404


# @app.route('/modules/stats', methods=['GET'])
# def module_stats():
#     module_name = request.args.get('module_name')
#     module = c.module()
#     if not module_name:
#         return jsonify({'error': 'Module name not provided in query parameter'}), 400

#     try:
#         if module_name:
#             modules_tree = module.stats(module_name)
#         else:
#             modules_tree = module.stats()
#         return jsonify(modules_tree)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500



# TODO: add routes to consume PM2 data 


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# c tree # list of modules with their location on the filesystem 

# c modules.fintest info # info abt a module 

# c modules.fintest keys # keys associated with a module 

# c modules.fintest stats # stats abt a module 
 
# c modules.fintest users

# c modules.fintest active_thread_count

# c namespace model.openai.free or c namespace 
    
# c servers & TODO: c server history (c servers gives list of modules being actively served)