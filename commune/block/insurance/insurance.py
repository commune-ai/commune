
import commune
from typing import Dict , Any, List
import streamlit as st
import json
class Insurance(commune.Module):
    
    def __init__(self, store: Dict = None, key: 'Key' = None):
    
        self.set_storage(store)
        self.set_key(key)
        
    
    def set_storage(self, storage: Dict = None):
        storage = {} if storage == None else storage
        assert isinstance(storage, dict), 'storage must be a dictionary'
        self.storage = storage
            
    
    def put(self,
            k:str, 
            v: Any,
            key: str = None,
            encrypt: bool = False) -> str:
        data = None
        key = self.resolve_key(key)
        if isinstance(v, dict):
            if 'data' in v and 'time' in v:
                data  = v
        
        if data == None:
            data = {'data': v, 'time': int(commune.time())}
        if encrypt:
            data = key.encrypt(data)
        # start with signature, data, public_address
        storage_item = key.sign(data, return_dict=True)
        storage_item['encrypt'] = encrypt
        address = key.ss58_address
        if address not in self.storage:
            self.storage[address] = {}
        self.storage[address][k] = storage_item
        
        
        
        return k
    
    def state_dict(self):
        import json
        state_dict = {}
        for address, storage_map in self.storage.items():
   
            state_dict[address] = json.dumps(storage_map)
            
        return state_dict
    def from_state_dict(self, state_dict: Dict) -> None:
        import json
        for k, v in state_dict.items():
            self.storage[k] = json.loads(v)
            
    def save(self, path: str):
        
        state_dict = self.state_dict()
        
        return self.put_json( path=path, data=state_dict)
    
    
    def resolve_key(self, key: str = None) -> commune.key:
        if key == None:
            key = self.key
        return key
    
    def get(self,
            k, 
            key:str = None,
            max_staleness: int = 1000) -> Any:
        key = self.resolve_key(key)

        item = self.storage[key.ss58_address][k]
        verified = key.verify(item)
        
        # decrypt if necessary
        if self.is_encrypted(item):
            
            item['data'] = key.decrypt(item['data'])
        item['data'] = self.str2python(item['data'])
        assert verified, 'could not verify signature'
        
        # check staleness
        staleness = commune.time() - item['data']['time']
        assert staleness < max_staleness

        return item['data']['data']


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
            
        
    @classmethod
    def sandbox(cls):
        
        self = cls()
        data = {'fam': 1, 'bro': 'fam', 'chris': {'sup': [1,'dawg']}}
        key = commune.key('Alice')
        st.write(self.put('bro',
                          data,
                          encrypt=False,
                          key=key
                          ))
        st.write(self.put('fam', data, key=commune.key('Bob'), encrypt=False))

        # st.write(self.get('bro'))
        st.write(self.key2address)
        st.write(self.state_dict())

    def streamlit_signin(self):
        '''
        sign in to the insurance module
        '''
        username = st.text_input('Username', value='Alice')
        password = st.text_input('Password', value='Bob', type='password')
        # expand the button to the full width of the sidebar
        cols = st.columns(2)

        self.username = username
        self.password = password
        seed = username+":"+ password
        self.key = commune.key(seed)
        
            
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

    def streamlit_save_claim(self):
        '''
        Save a claim
        '''
        st.write('## Save Claim')

        
        # ... (all input fields as in the previous example)

        self.button['save_claim'] = st.button('Save Claim', self.streamlit_save_claim)
        if self.button['save_claim']:
            # Create a dictionary with the claim data
            claim_data = self.default_claim_data
            
            # Convert the claim data dictionary to a JSON object
            claim_data_json = json.dumps(claim_data)

            # Save the JSON object to your preferred storage (e.g., database, file, API)
            # For example, you can save the JSON object to a file:
            with open('claim_data.json', 'w') as f:
                f.write(claim_data_json)

            st.write('Claim saved')

    def streamlit_save_claim(self):
        '''
        Save a claim
        '''
        st.write('## Save Claim')
        claim_data = commune.copy(self.default_claim_data)
        
        # have these in boxes
        
        with st.expander('Claim Data', True):
            cols = st.columns(2)
            with cols[0]:
                claim_data.policy_number = st.text_input('Policy Number', value=claim_data.policy_number)
                claim_data.claimant_name = st.text_input('Claimant Name', value=claim_data.claimant_name)
                claim_data.claimant_contact = st.text_input('Claimant Contact (Phone/Email)', value=claim_data.claimant_contact)
                claim_data.incident_date = st.date_input('Incident Date', value=claim_data.incident_date)
                claim_data.claim_status = st.text_input('Claim Status', value=claim_data.claim_status)

            with cols[1]:
                claim_data.incident_location = st.text_input('Incident Location', value=claim_data.incident_location)
                claim_data.incident_description = st.text_area('Incident Description', value=claim_data.incident_description)
                claim_data.claim_type = st.text_input('Claim Type', value=claim_data.claim_type)
                claim_data.claim_amount = st.number_input('Claim Amount (in USD)', value=claim_data.claim_amount)
        claim_data.claim_document = st.file_uploader('Upload Claim Document (optional)', type=['pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png'])
        # get contents of the file
        if claim_data.claim_document is not None:
            claim_data.claim_document = claim_data.claim_document.read()
        
        claim_data.incident_date = claim_data.incident_date.strftime('%Y-%m-%d')
        

        self.button['save_claim'] = st.button('Save Claim', self.streamlit_save_claim)
        if self.button['save_claim']:
            # Save the claim data to your preferred storage (e.g., database, file, API)
            st.write('Claim saved')

    @classmethod
    def streamlit(cls):
        self = cls()
        self.button = {}
        self.streamlit_sidebar()
        # rocket emoji 
        st.write(f'# Hello {self.username} 🚀')
        self.streamlit_save_claim()
    
if __name__ == "__main__":
    Insurance.streamlit()
    
    