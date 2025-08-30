import requests
import commune as c
import os

class Lighthouse:
    def __init__(self, api_key='api/lighthouse'):
        self.set_api_key(api_key)
        self.base_url = 'https://node.lighthouse.storage/api/v0'
        self.auth_url = 'https://api.lighthouse.storage/api/auth'

    def set_api_key(self, api_key):
        api_key = c.get(api_key, 'LIGHTHOUSE_API_KEY')
        api_key = os.environ.get(api_key, api_key)
        print('api_key', api_key)
        self.api_key = api_key
        return api_key

    def add_file(self, file_path):
        url = f'{self.base_url}/add'
        file = os.path.abspath(file_path)
        headers = {'Authorization': f'Bearer {self.api_key}'}
        files = {'file': open(file, 'rb')}
        response = requests.post(url, headers=headers, files=files)
        if response.status_code == 200:
            print('File uploaded successfully:', response.json())
        else:
            print('Error uploading file:', response.status_code, response.text)
        return responsep

    def list_files(self):
        url = f'https://api.lighthouse.storage/api/user/files_uploaded?lastKey=null'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print('Error:', response.status_code, response.text)
            return None
       

    def upload_encrypted_file(self, file_path):
        url = f'{self.base_url}/add_encrypted'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, headers=headers, files=files)
        return response.json()

    def get_auth_message(self, public_key):
        url = f'{self.auth_url}/get_message'
        response = requests.get(url, params={'publicKey': public_key})
        return response.json()

    def file_info(self, cid):
        url = f'{self.base_url}/file_info/{cid}'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(url, headers=headers)
        return response.json()

    def test_lighthouse():
        test_api_key = 'your_test_api_key'
        lh = Lighthouse(api_key=test_api_key)

        # Test file upload
        upload_response = lh.upload_file('/path/to/testfile.jpeg')
        print('Upload Response:', upload_response)

        # Test encrypted file upload
        encrypted_upload_response = lh.upload_encrypted_file('/path/to/testfile.jpeg')
        print('Encrypted Upload Response:', encrypted_upload_response)

        # Test list files
        files_list = lh.list_files()
        print('Files List:', files_list)

        # Test file info (replace with a valid CID)
        if files_list and 'data' in files_list and len(files_list['data']) > 0:
            file_cid = files_list['data'][0]['cid']
            file_details = lh.file_info(cid=file_cid)
            print('File Details:', file_details)


    def create_api_key(self, key=None):
        key = c.get_key(key, type='eth')
        public_key =  key.address
        data = requests.get(
            f'https://api.lighthouse.storage/api/auth/get_message?publicKey={public_key}'
        )
        signature = key.sign(data).hex()
        print('signature', signature)
        response = requests.post(
            'https://api.lighthouse.storage/api/auth/create_api_key',
            json={
                'publicKey': public_key,
                'signedMessage': signature,
                'keyName': 'test'
            }
        )
        return response.json()