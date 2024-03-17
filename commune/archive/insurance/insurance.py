
import commune
from typing import Dict , Any, List
import streamlit as st
import json
class Insurance(commune.Module):
    
    def __init__(self,
                 user: str = 'Alice',
                 password: str = '1234567',
                storage: Dict = None,):
    

        self.signin(username=user, password=password)
        self.set_storage(storage)
        
    
    def set_storage(self, storage: Dict = None):
        storage = {} if storage == None else storage
        assert isinstance(storage, dict), 'storage must be a dictionary'
        self.storage = storage
            
    
    def save_claim(self,
            policy_id:str, 
            claim_data: Dict,
            key: str = None,
            encrypt: bool = False) -> str:
        claim_data = self.munch2dict(claim_data)
        claim_data['last_time_saved'] = int(commune.time())
        key = self.resolve_key(key)
        if encrypt:
            claim_data = key.encrypt(claim_data)
        # start with signature, data, public_address
        storage_item = key.sign(claim_data, return_dict=True)
        storage_item['encrypt'] = encrypt
        address = key.ss58_address
        if address not in self.storage:
            self.storage[address] = {}
        
        path = f'claims/{address}/{policy_id}'
        self.put_json(path, storage_item)
        
        return policy_id
    @property
    def claim_paths(self, key: str = None) -> List:
        key = self.resolve_key(key)
        return self.glob(f'claims/{key.address}/*')


    
    def resolve_key(self, key: str = None) -> commune.key:
        if key == None:
            key = self.key
        return key
    
    
    def get_claim(self,
            policy_id: str = None, 
            path: str = None,
            key:str = None,
            item: Dict = None,
            max_staleness: int = 1000) -> Any:
        key = self.resolve_key(key)
        
        if item == None:
            if path != None and policy_id == None: 
                item = self.get_json(path)
            else:
                assert policy_id != None, 'must provide policy_id or path'
            
                item = self.get_json(f'claims/{key.address}/{policy_id}')

        verified = key.verify(item)
        
        
        # decrypt if necessary
        if self.is_encrypted(item):
            
            item['data'] = key.decrypt(item['data'])
        if isinstance(item['data'], str):
            item['data'] = self.str2python(item['data'])
        assert verified, 'could not verify signature'
        
        # check staleness
        staleness = commune.time() - item['data']['last_time_saved']
        # assert staleness < max_staleness
        
        


        return item['data']


    @property
    def key2address(self) -> Dict:
        key2address = {}
        for address, storage_map in self.storage.items():
            for k, v in storage_map.items():
                key2address[k] = address
        return key2address
        

    def is_encrypted(self, item: Dict) -> bool:
        return item.get('encrypt', False)
 
    @classmethod
    def test(cls):
        self = cls()
        
        object_list = [0, {'fam': 1}, 'whadup']
        for obj in object_list:
            self.put('test', obj)
            assert self.get('test') == obj
            
    def get_user_claims(self):
        claims = []
        st.write(self.claim_paths)
        for path in self.claim_paths:
            claims.append(self.get_claim(path=path))
        return claims
        
    @classmethod
    def sandbox(cls):
        
        self = cls()
        data = {'fam': 1, 'bro': 'fam', 'chris': {'sup': [1,'dawg']}}
        key = commune.get_key('Alice')
        st.write(self.put('bro',
                          data,
                          encrypt=False,
                          key=key
                          ))
        st.write(self.put('fam', data, key=commune.get_key('Bob'), encrypt=False))

        # st.write(self.get('bro'))
        st.write(self.key2address)
        st.write(self.state_dict())

    def streamlit_signin(self):
        '''
        sign in to the insurance module
        '''
        username = st.text_input('Username', value=self.username)
        password = st.text_input('Password', value=self.password, type='password')
        # expand the button to the full width of the sidebar
        cols = st.columns(2)


        signin_button = st.button('Sign In')
    
        if signin_button:
            self.signin(username=username, password=password)
        
  
        
    def signin(self, username, password):
        self.username = username
        self.password = password
        self.key = commune.get_key(username+password)
        self.address = self.key.ss58_address
        # st.write('Signed in as', self.username)
        return True
    def streamlit_sidebar(self):
        with st.sidebar:
            st.write('# Commune AI Insurance')
            self.streamlit_signin()

    @property
    def default_claim_data(self):
        import datetime
        '''
        Default claim data
        '''
        return self.dict2munch({
            'policy_number': '123456',
            'claimant_name': self.username,
            'claimant_contact': 'Not Provided',
            'incident_date': datetime.datetime.now(),
            'incident_location': 'Toronto, Ontario, Canada',
            'incident_description': 'Fender bender',
            'claim_type': 'Auto',
            'claim_status': 'Open',
            'claim_amount': 1000,
            'claim_document': None
            
        })
    @classmethod
    def glob(cls,  path ='**', resolve_path:bool = True, files_only:bool = True):
        import os
        from glob import glob
        path = cls.resolve_path(path, extension=None) if resolve_path else path
        
        if os.path.isdir(path):
            path = os.path.join(path, '**')
            
        paths = glob(path, recursive=True)
        if len(paths) == 0:
            paths = glob(os.path.join(path, '**'), recursive=True)
        if len(paths) == 0:
            paths = glob(os.path.join(path, '*'), recursive=True)

        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        return paths
         
    def streamlit_save_claim(self):
        '''
        Save a claim
        '''
        
        claim_data = commune.copy(self.default_claim_data)
        
        # have these in boxes
        
        with st.expander('', True):
            st.write('### Claim Details')
            claim_data.claim_document = st.file_uploader('Upload Claim Document (optional)', type=['pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png'])

            cols = st.columns(2)

            with cols[0]:
                
                
                claim_data.policy_number = st.text_input('Policy Number', value=claim_data.policy_number)
                claim_data.claimant_name = st.text_input('Claimant Name', value=self.username)
                claim_data.claimant_contact = st.text_input('Claimant Contact (Phone/Email)', value=claim_data.claimant_contact)
                claim_data.incident_date = st.date_input('Incident Date', value=claim_data.incident_date)
                claim_data.claim_status = st.text_input('Claim Status', value=claim_data.claim_status)

            with cols[1]:
                claim_data.incident_location = st.text_input('Incident Location', value=claim_data.incident_location)
                claim_data.incident_description = st.text_area('Incident Description', value=claim_data.incident_description)
                claim_data.claim_type = st.text_xinput('Claim Type', value=claim_data.claim_type)
                claim_data.claim_amount = st.number_input('Claim Amount (in USD)', value=claim_data.claim_amount)
        # get contents of the file
        if claim_data.claim_document is not None:
            claim_data.claim_document = claim_data.claim_document.read()
        
        claim_data.incident_date = claim_data.incident_date.strftime('%Y-%m-%d')
        

        self.button['save_claim'] = cols[0].button('Save Claim')
        

        if self.button['save_claim'] :
            # Save the claim data to your preferred storage (e.g., database, file, API)
            self.save_claim(claim_data['policy_number'], claim_data, encrypt=True)
            

            st.write('Claim saved')
        
        with st.expander('', True):
            for claim in self.my_claims:
                title = f"Claim Type: {claim['claim_type']} | Incident Data: {claim['incident_date']}"
                st.write("### "+title)
                st.write(claim)

    @property
    def my_claims(self) -> List[Dict]:
        return self.get_user_claims()
    @classmethod
    def streamlit(cls):
        cls.new_event_loop()
        self = cls()
        self.button = {}
        self.streamlit_sidebar()
        # doctor emoji 
        # rocket emoji 
        st.write(f'# Hello {self.username}, here are your claims ')
        self.streamlit_save_claim()
    
if __name__ == "__main__":
    Insurance.streamlit()
    
    