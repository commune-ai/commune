from substrateinterface import SubstrateInterface
import commune

from commune.web3.substrate.account import SubstrateAccount

class SubstrateNetwork(SubstrateInterface, commune.Module):
    def __init__(self, 
                url:str="127.0.0.1:9944", 
                websocket:str=None, 
                ss58_format:int=42, 
                type_registry:dict=None, 
                type_registry_preset=None, 
                cache_region=None, 
                runtime_config=None, 
                use_remote_preset=False,
                 ws_options=None, 
                 auto_discover=True, 
                 auto_reconnect=True, 
                 *args, 
                 **kwargs):

        '''
        A specialized class in interfacing with a Substrate node.

        Parameters
       A specialized class in interfacing with a Substrate node.

        Parameters
        url : the URL to the substrate node, either in format <https://127.0.0.1:9933> or wss://127.0.0.1:9944
        
        ss58_format : The address type which account IDs will be SS58-encoded to Substrate addresses. Defaults to 42, for Kusama the address type is 2
        
        type_registry : A dict containing the custom type registry in format: {'types': {'customType': 'u32'},..}
        
        type_registry_preset : The name of the predefined type registry shipped with the SCALE-codec, e.g. kusama
        
        cache_region : a Dogpile cache region as a central store for the metadata cache
        
        use_remote_preset : When True preset is downloaded from Github master, otherwise use files from local installed scalecodec package
        
        ws_options : dict of options to pass to the websocket-client create_connection function
        : dict of options to pass to the websocket-client create_connection function
                
        '''
        if not url.startswith('ws://'):
            url = f'ws://{url}'
        SubstrateInterface.__init__(self,
                                    url=url, 
                                    websocket=websocket, 
                                    ss58_format=ss58_format, 
                                    type_registry=type_registry, 
                                    type_registry_preset=type_registry_preset, 
                                    cache_region=cache_region, 
                                    runtime_config=runtime_config, 
                                    use_remote_preset=use_remote_preset,
                                    ws_options=ws_options, 
                                    auto_discover=auto_discover, 
                                    auto_reconnect=auto_reconnect, *args, **kwargs)

    
if __name__ == '__main__':
    module = SubstrateNetwork()
    st.write(module)
