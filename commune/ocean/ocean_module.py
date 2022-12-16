
# Create Ocean instance
import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
from commune import Module

from commune.utils import  dict_put, dict_has, Timer
import datetime
from ocean_lib.ocean.ocean_assets import OceanAssets 
from ocean_lib.example_config import ExampleConfig
from ocean_lib.web3_internal.contract_base import ContractBase
from ocean_lib.models.datatoken import Datatoken
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.structures.file_objects import IpfsFile, UrlFile
from ocean_lib.services.service import Service
from ocean_lib.structures.file_objects import FilesTypeFactory
from ocean_lib.exceptions import AquariusError
from typing import *
# Create Alice's wallet
from ocean_lib.config import Config
from ocean_lib.models.data_nft import DataNFT
from ocean_lib.web3_internal.wallet import Wallet
from ocean_lib.web3_internal.constants import ZERO_ADDRESS
# from web3._utils.datatypes import Contract
import fsspec

from ocean_lib.structures.file_objects import UrlFile

# from algocean import BaseModule


class OceanModule(commune.Module):
    default_wallet_key = 'default'
    wallets = {}
    def __init__(self, ocean=None, network=None, config=None, **kwargs):
        BaseModule.__init__(self, config=config, **kwargs)
        self.set_ocean(ocean= ocean)
        self.set_network(network=network)


    def load_wallets(self, wallets=None):
        '''
        Load Private Key variable into your
         wallet when pointing to the private 
         key env.

         wallet:
            alice: PRIVATE_KEY1
            bob: PRIVATE_KEY2


        or you can put in the keys manually
         but thats too deep
        '''
        if wallets == None:
            wallets = self.config.get('wallet', self.config.get('wallets'))
        assert isinstance(wallets, dict), f'{wallets} should be a dictionary'

        for k,v in wallets.items():
            self.add_wallet(wallet_key=k, private_key=v)
    

    generate_wallets = load_wallets




    @property
    def network(self):
    
        if not hasattr(self, '_network'):
            self._network = self.config['network']
        return self._network

    @network.setter
    def network(self, network:str):
        return self.set_network(network=network)

    def set_network(self, network:str=None):
        '''
        set the network
        defa
        ults to local fork
        '''
        if network == None:
            network = self.config.get('network')
        self._network = network
        self.set_ocean(ocean_config=f'{self.ocean_configs_folder}/{network}.in')

    def set_ocean(self, ocean):
        if 'OCEAN_NETWORK_URL' in os.environ:
            os.environ.pop('OCEAN_NETWORK_URL')
        self.config['ocean'] = self.get_ocean(ocean, return_ocean=False)
        self.ocean = Ocean(self.config['ocean'])
        self.web3 = self.ocean.web3
        self.aquarius = self.ocean.assets._aquarius
        self.load_wallets()
        
    @staticmethod
    def get_ocean( ocean_config=None, return_ocean=True):
        if ocean_config == None:
            ocean_config =  ExampleConfig.get_config()
        elif isinstance(ocean_config, str):
            if ocean_config.startswith('./'):
            
                ocean_config = os.path.dirname(__file__) + ocean_config[1:]
            ocean_config = Config(filename=ocean_config)
        
        elif isinstance(ocean_config, Config):
            ocean_config = ocean_config
        else:
            raise NotImplementedError  
        
        assert isinstance(ocean_config, Config), 'ocean_config must be type Config'
        if return_ocean:
            return Ocean(ocean_config)
        return ocean_config



    def get_existing_wallet_key(self, private_key:str=None, address:str=None):
        for w_k, w in self.wallets.items():
            if private_key==w.private_key or address == w.address:
                return w_k

        return None

    def add_wallet(self, wallet_key:str='default', private_key:str='TEST_PRIVATE_KEY1', ):
        '''
        wallet_key: what is the key you want to store the wallet in
        private_key: the key itself or an env variable name pointing to that key
        '''
        if isinstance(wallet,Wallet):
            self.wallets[wallet_key] = wallet
            return wallet
        # fetch the name or the key
        private_key = os.getenv(private_key, private_key)
        self.wallets[wallet_key] = self.generate_wallet(private_key=private_key)
        return self.wallets[wallet_key]

    def regenerate_wallets(self):
        for k, wallet in self.wallets.items():
            self.wallets[k] = self.generate_wallet(private_key=wallet.private_key)

    def generate_wallet(self, private_key:str):
        private_key = os.getenv(private_key, private_key)
        return Wallet(web3=self.web3, 
                      private_key=private_key, 
                      block_confirmations=self.config['ocean'].block_confirmations, 
                      transaction_timeout=self.config['ocean'].transaction_timeout)  
    
    def rm_wallet(self, key):
        '''
        remove wallet and all data relating to it
        '''
        del self.wallets[key]
    remove_wallet = rm_wallet

    def list_wallets(self, return_keys=True):
        '''
        list wallets
        '''
        if return_keys:
            return list(self.wallets.keys())
        else:
            return  [(k,v) for k,v in self.wallets.items()]
    ls_wallets = list_wallets

    @property
    def wallet(self):
        # gets the default wallet
        return self.wallets[self.default_wallet_key]

    def set_default_wallet(self, key:str):
        self.default_wallet_key = key
        return self.wallets[self.default_wallet_key]


    @property
    def default_wallet_key(self):
        wallet_keys= self.wallet_keys

        if not hasattr(self, '_default_wallet_key'):
            self._default_wallet_key = None

        if self._default_wallet_key == None:
            if len(wallet_keys)>0:
                self._default_wallet_key  =  wallet_keys[0]
            else:
                self._default_wallet_key  = None
        elif isinstance(self._default_wallet_key, str):
            self._default_wallet_key = self._default_wallet_key
        
        return self._default_wallet_key 
                


    @property
    def wallet_keys(self):
        return list(self.wallets.keys())

    @default_wallet_key.setter
    def default_wallet_key(self, key):
        assert key in self.wallet_keys, f'{key} is not in {wallet_keys}'
        self._default_wallet_key = key
        return self._default_wallet_key
        
    
    def get_wallet(self, wallet, return_address=False):
        if wallet == None:
            wallet = self.wallet
        elif isinstance(wallet, str):
            if self.web3.isAddress(wallet):
                assert return_address
                return wallet
            
            wallet = self.wallets[wallet]
        elif isinstance(wallet,Wallet):
            wallet = wallet
        else:
            raise Exception(f'Bro, the wallet {wallet} does not exist or is not supported')

        assert isinstance(wallet, Wallet), f'wallet is not of type Wallet but  is {Wallet}'
    
        if return_address:
            return wallet.address
        else: 
            return wallet


    def create_datanft(self, name:str , symbol:str=None, wallet:Union[str, Wallet]=None):
        wallet = self.get_wallet(wallet=wallet)

        if symbol == None:
            symbol = name
        nft_key =  symbol
        datanft = self.get_datanft(datanft=nft_key, handle_error=True)
        if datanft == None:
            datanft = self.ocean.create_data_nft(name=name, symbol=symbol, from_wallet=wallet)

        return datanft

    def list_datanfts(self):
        return list(self.datanfts.keys())


    def generate_datatoken_name(self, datanft:Union[str, DataNFT]=None):
        datanft = self.get_datanft(datanft)
        index =  0 
        nft_token_map =  self.nft_token_map(datanft)
        while True:
            token_name = f'DT{index}'
            if token_name not in nft_token_map:
                return token_name
    def create_datatoken(self, name:str, symbol:str=None, datanft:Union[str, DataNFT]=None, wallet:Union[str, Wallet]=None):
        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft)
        datatokens_map = self.get_datatokens(datanft=datanft, return_type='map')
        datatoken = datatokens_map.get(name)
        
        symbol = symbol if symbol != None else name
        if datatoken == None:
            datatoken = datanft.create_datatoken(name=name, symbol=symbol, from_wallet=wallet)

        assert isinstance(datatoken, Datatoken), f'{datatoken}'
        return datatoken



    def get_contract(self, address:str, contract_class=ContractBase):
        return contract_class(web3=self.web3, address=address)
    
    def get_address(self, contract):
        return contract.address


    # @staticmethod
    # def get_asset_did(asset:Asset):
    #     return asset.did

    def get_assets(self, wallet:Union[str, Wallet]=None, return_type='dict'):
        '''
        get assets from wallet
        '''
        
        wallet = self.get_wallet(wallet)
        text_query = f'metadata.author:{wallet.address}' 
        # current_time_iso = datetime.datetime.now().isoformat()
        assets = self.search(text=text_query, return_type='asset')
        return assets

    def get_asset(self, datanft=None, did=None, handle_error=False, timeout=10):
        '''
        get asset from datanft using aquarius
        '''
        if isinstance(did, str):
            return self.aquarius.get_asset_ddo(did)
        if isinstance(datanft, str):
            if datanft.startswith('did:op'):
                did = datanft
        if isinstance(datanft, Asset):
            return datanft

        datanft =self.get_datanft(datanft)
        query_text = f'nft.address:{datanft.address}'

        with Timer() as timer:
            while timer.elapsed < timeout:
                try:
                    time.sleep(0.1)
                    assets = self.search(text=query_text, return_type='asset')
                    assert len(assets)==1, f'This asset from datanft: {datanft.address} does not exist'
                    assert isinstance(assets[0], Asset), f'The asset is suppose to be an Asset My guy'
                    return assets[0]
                except Exception as e:
                    if handle_error:
                        return None
                    else:
                        raise(e)
        



    def get_wallet_datanfts(self, wallet:Union[str, Wallet]=None):
        wallet_address = self.get_wallet(wallet, return_address=True)
        return self.search(text='metadata.address:{wallet_address}', return_type='asset')
        


    # def assets(self, datanft, wallet):


    
    @staticmethod
    def fill_default_kwargs(default_kwargs, kwargs):
        return {**default_kwargs, **kwargs}


    def nft_token_map(self, datanft=None, return_type='map'):
        '''
        params:
            return_type: the type of retun, options
                options:
                    - map: full map (dict)
                    - key: list of keys (list[str])
                    - value: list of values (list[Datatoken])
        
        '''
        supported_return_types = ['map', 'key', 'value']
        assert return_type in supported_return_types, \
              f'Please specify a return_type as one of the following {supported_return_types}'

        
        datanft =self.get_datanft(datanft)
        datanft_symbol = datanft.symbol()
        output_token_map = {}

        for k,v in self.datatokens.items():
            k_nft,k_token = k.split('.')
            if datanft_symbol == k_nft:
                output_token_map[k_token] = v


        if return_type in ['key']:
            return list(output_token_map.keys())
        elif return_type in ['value']:
            return list(output_token_map.values())
        elif return_type in ['map']:
            return output_token_map

        raise Exception('This should not run fam')


    def dispense_tokens(self, 
                        datatoken:Union[str, Datatoken]=None, 
                        datanft:Union[str, DataNFT]=None,
                        amount:int=100,
                        destination:str=None,
                        wallet:Union[str,Wallet]=None):
        wallet = self.get_wallet(wallet)
        amount = self.ocean.to_wei(amount)
        datatoken = self.get_datatoken(datatoken=datatoken, datanft=datanft)
        if destination == None:
            destination = wallet.address 
        else:
            destination = self.get_wallet(destination, return_address=True)


        # ContractBase.to_checksum_address(destination)


        self.ocean.dispenser.dispense(datatoken=datatoken.address,
                                     amount=amount, 
                                    destination= destination,
                                     from_wallet=wallet)

    def create_dispenser(self,
                        datatoken:Union[str, Datatoken]=None, 
                        datanft:Union[str, DataNFT]=None,
                        max_tokens:int=100, 
                        max_balance:int=None, 
                        with_mint=True, 
                        wallet=None,
                        **kwargs):

        datatoken=  self.get_datatoken(datatoken=datatoken, datanft=datanft)
        wallet = self.get_wallet(wallet)

        dispenser = self.ocean.dispenser

        max_tokens = self.ocean.to_wei(max_tokens)
        if max_balance == None:
            max_balance = max_tokens
        else:
            max_balance = self.ocean.to_wei(max_tokens)
        # Create dispenser

        datatoken.create_dispenser(
            dispenser_address=dispenser.address,
            max_balance=max_balance,
            max_tokens=max_tokens,
            with_mint=with_mint,
            allowed_swapper=ZERO_ADDRESS,
            from_wallet=wallet,
        )

    

    def get_datanft(self, datanft:Union[str, DataNFT]=None, handle_error:bool= False):
        '''
        dataNFT can be address, key in self.datanfts or a DataNFT
        '''

        try:

            if isinstance(datanft, DataNFT):
                datanft =  datanft

            elif isinstance(datanft, str):
                if self.web3.isAddress(datanft):
                    datanft = DataNFT(web3=self.web3, address=datanft)
                elif datanft in self.datanfts :
                    datanft = self.datanfts[datanft]
                else:
                    raise NotImplementedError(f'{datanft} is not found')
            
            assert isinstance(datanft, DataNFT), f'datanft should be in the formate of DataNFT, not {datanft}'
            return datanft
        except Exception as e:
            if handle_error:
                return None
            else:
                raise(e)



    def create_asset(self,datanft, services:list, metadata:dict=None, wallet=None, **kwargs ):
        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft=datanft)
        asset = self.get_asset(datanft, handle_error=True)
        if asset != None:
            assert isinstance(asset, Asset)
            return asset
               
        if metadata == None:
            metadata = self.create_metadata(datanft=datanft,wallet=wallet, **kwargs.get('metadata', {}))
        
        default_kwargs= dict(
        data_nft_address = datanft.address,
        deployed_datatokens = [self.get_datatoken(s.datatoken) for s in services],
        publisher_wallet= wallet,
        metadata= metadata,
        services=services
        )

        kwargs = {**kwargs, **default_kwargs}

        
        if asset == None:
            asset = self.ocean.assets.create(**kwargs)
        
        return asset

            
    

    def dummy_files(self, mode='ipfs'):
        cid = self.client.ipfs.put_json(data={'dummy':True})
        return self.create_files([{'hash':f'{cid}', 'type':'ipfs'}]*1)

    @staticmethod
    def create_files(file_objects:Union[list, dict]=None, handle_null=False):
        if isinstance(file_objects, dict):
            file_objects = [file_objects]
        assert isinstance(file_objects, list) 
        assert isinstance(file_objects[0], dict)

        output_files = []
        for file_object in file_objects:
            output_files.append(FilesTypeFactory(file_object))

        return output_files



    def mint(self, to:Union[str,Wallet], value:int=1,datanft:str=None, datatoken:str=None, wallet:Wallet=None , encode_value=True):
        wallet = self.get_wallet(wallet=wallet)
        to_address = self.get_wallet(wallet=to, return_address=True)
        datatoken = self.get_datatoken(datanft=datanft,datatoken=datatoken)
        
        if encode_value:
            value = self.ocean.to_wei(str(value))
        
        assert datatoken != None, f'datatoken is None my guy, args: {dict(datanft=datanft, datatoken=datatoken)}'
        datatoken.mint(account_address=to_address, 
                        value=value, from_wallet=wallet )


    def get_datatoken(self, datatoken:str=None, datanft:str=None) -> Datatoken:

        if isinstance(datatoken, Datatoken):
            return datatoken

        if isinstance(datatoken, str):
            if self.web3.isAddress(datatoken): 
                return Datatoken(web3=self.web3,address=datatoken)


            datatokens_map = self.get_datatokens(datanft=datanft, return_type='map')
            if datatoken in datatokens_map:
                return datatokens_map[datatoken]
        else:
            raise Exception(f'BRO {datanft} is not define')

    
    # @property
    # def datatokens(self):
    #     return {}

    def get_balance(self,wallet:Union[Wallet,str]=None, datanft:str=None, datatoken:str=None):
        
        wallet_address = self.get_wallet(wallet=wallet, return_address=True)
        datatoken = self.get_datatoken(datanft=datanft, datatoken=datatoken )
        if datatoken == None:
            value =  self.web3.eth.get_balance(wallet_address)
        else:
            value =  datatoken.balanceOf(wallet_address)
        
        return value
   
    @property
    def assets(self):
        return self.get_assets()

    @property
    def datatokens(self):
        dt_list = []
        for asset in self.assets:
            dt_list += self.get_datatokens(asset=asset)
        return [dt for dt in dt_list]

    @property
    def services(self):
        services = []
        for asset in self.get_assets():
            services += asset.services

        return services
    
    @property
    def datanfts(self):
        datanfts = []
        for asset in self.get_assets():
            datanfts += [DataNFT(web3=self.web3, address=asset.nft['address'])]

        return datanfts

    def get_services(self, asset, return_type = 'service'):

        asset = self.get_asset(asset)

        supported_types = ['service', 'dict']
        assert return_type in supported_types
        
        if return_type== 'service':
            return asset.services
        elif return_type == 'dict':
            return [s for s in asset.services.__dict__]


    def create_service(self,
                        name: str,
                        service_type:str = 'access',
                        files:list = None,
                        datanft:Optional[str]=None,
                        datatoken: Optional[str]=None,
                        additional_information: dict = {},
                        wallet=None,**kwargs):
        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft=datanft)
        datatoken = datatoken if datatoken != None else name
        datatoken = self.get_datatoken(datanft=datanft, datatoken=datatoken)

        if files == None:
            files = self.dummy_files()

        service_dict = dict(
            name=name,
            service_type=service_type,
            service_id= kwargs.get('id', name),
            files=files,
            service_endpoint=kwargs.get('service_endpoint', self.ocean.config.provider_url),
            datatoken=datatoken.address,
            timeout=kwargs.get('timeout', 3600),
            description = 'Insert Description here',
            additional_information= additional_information,
        )

        service_dict = {**service_dict, **kwargs}

        return Service(**service_dict)




    def get_services(self, asset=None, datanft=None, return_type='service'):
        if asset != None:
            asset = self.get_asset(asset)
        elif datanft != None:
            asset = self.get_asset(datanft)
        else:
            raise NotImplementedError
        

        services = asset.services
        if return_type == 'dict':
            services = [ s.__dict__ for s in services]
        elif return_type == 'service':
            services = services
        else:
            raise NotImplementedError
        
        return services





    def get_service(self, asset=None, service=None):
        if isinstance(service, Service):
            return service
        else:
            asset = self.get_asset(asset)
            if service == None:
                assert len(asset.services)>0, 'There are no services for the asset'
                return asset.services[0]
            elif isinstance(service, int):
                return asset.services[service]
            else:
                raise NotImplementedError(f"asset:{asset} service: {service}")
        
            
    def pay_for_access_service(self,
                              asset:Union[str,Asset],
                              service:Union[str,Service]=None,
                              wallet:Union[str, Wallet]=None, **kwargs):
        
        
        asset = self.get_asset(asset)
        service= self.get_service(asset=asset, service=service)
        wallet = self.get_wallet(wallet=wallet) 


        default_kwargs = dict(
            asset=asset,
            service=service,
            consume_market_order_fee_address=service.datatoken,
            consume_market_order_fee_token=wallet.address,
            consume_market_order_fee_amount=0,
            wallet=wallet,
        )

        kwargs = {**default_kwargs, **kwargs}

        order_tx_id = self.ocean.assets.pay_for_access_service( **kwargs )     

        return order_tx_id   
        

    def download_asset(self, asset, service=None, destination='./', order_tx_id=None,index=None, wallet=None):
        asset = self.get_asset(asset)
        service= self.get_service(asset=asset, service=service)
        wallet = self.get_wallet(wallet=wallet) 

        if order_tx_id == None:
            order_tx_id = self.pay_for_access_service(asset=asset, service=service, wallet=wallet)

        file_path = self.ocean.assets.download_asset(
                                        asset=asset,
                                        service=service,
                                        consumer_wallet=wallet,
                                        destination=destination,
                                        index=index,
                                        order_tx_id=order_tx_id
                                    )
        return file_path

    def create_metadata(self, datanft=None, wallet=None, **kwargs ):

        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft)
        metadata ={}

        metadata['name'] = datanft.name
        metadata['description'] = kwargs.get('description', 'Insert Description')
        metadata['author'] = kwargs.get('author', wallet.address)
        metadata['license'] = kwargs.get('license', "CC0: PublicDomain")
        metadata['categories'] = kwargs.get('categories', [])
        metadata['tags'] = kwargs.get('tags', [])
        metadata['additionalInformation'] = kwargs.get('additionalInformation', {})
        metadata['type'] = kwargs.get('type', 'dataset')

        current_datetime = datetime.datetime.now().isoformat()
        metadata["created"]=  current_datetime
        metadata["updated"] = current_datetime

        return metadata



    def search(self, text: str, return_type:str='asset') -> list:
        """
        Search an asset in oceanDB using aquarius.
        :param text: String with the value that you are searching
        :return: List of assets that match with the query
        """
        # logger.info(f"Searching asset containing: {text}")

        ddo_list = [ddo_dict['_source'] for ddo_dict in self.aquarius.query_search({"query": {"query_string": {"query": text}}}) 
                        if "_source" in ddo_dict]
        
        if return_type == 'asset':
            ddo_list = [Asset.from_dict(ddo) for ddo in ddo_list]
        elif return_type == 'dict':
            pass
        else:
            raise NotImplementedError

        return ddo_list

    def hash(self, data:str, algo='keccak'):
        if algo == 'keccak':
            return self.web3.toHex((self.web3.keccak(text=data)))
        else:
            raise NotImplementedError

    def sign(self, data:str, wallet=None):
        raise NotImplemented

    @staticmethod
    def describe(instance):
        supported_return_types = [ContractBase]
        
        if isinstance(instance, ContractBase):
            return instance.contract.functions._functions
        else:
            raise NotImplementedError(f'Can only describe {ContractBase}')

    def get_datatokens(self, datanft:Union[ContractBase]=None, asset=None, return_type:str='value'):


        datatokens = []
        if asset != None:
            asset =  self.get_asset(asset)
            datatoken_obj_list =  asset.datatokens
            for datatoken_obj in datatoken_obj_list:
                dt_address = datatoken_obj.get('address')
                datatokens += [Datatoken(web3=self.web3, address=dt_address)]
            
        elif datanft != None:
            datanft = self.get_datanft(datanft)   
            dt_address_list = datanft.contract.caller.getTokensList()
            datatokens = [Datatoken(web3=self.web3,address=dt_addr) for dt_addr in dt_address_list]
        
        else:
            raise NotImplementedError(f'datanft: {datanft}')

        supporrted_return_types = ['map', 'key', 'value']

        assert return_type in supporrted_return_types, f'{return_type} not in {supporrted_return_types}'
        if return_type in ['map']:
            return {t.symbol():t for t in datatokens }
        elif return_type in ['key']:
            return [t.symbol() for t in datatokens]
        elif return_type in ['value']:
            return datatokens
        else:
            raise NotImplementedError


if __name__ == '__main__':
    import os
    # OceanModule.st_test()
    module = OceanModule()
    # st.write(module.ocean.config.__dict__)
    module.set_network('mumbai')
    st.write(module.wallet.web3.provider)

    module.create_datanft('bro', wallet='bob')
    # module.st_test()
