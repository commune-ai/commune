import os
import urllib
import commune as c
import requests
import logging
import netaddr
from typing import *


class Network(c.Module):

    @staticmethod
    def int_to_ip(int_val: int) -> str:
        r""" Maps an integer to a unique ip-string 
            Args:
                int_val  (:type:`int128`, `required`):
                    The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

            Returns:
                str_val (:tyep:`str`, `required):
                    The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

            Raises:
                netaddr.core.AddrFormatError (Exception):
                    Raised when the passed int_vals is not a valid ip int value.
        """
        import netaddr
        return str(netaddr.IPAddress(int_val))
    
    @staticmethod
    def ip_to_int(str_val: str) -> int:
        r""" Maps an ip-string to a unique integer.
            arg:
                str_val (:tyep:`str`, `required):
                    The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

            Returns:
                int_val  (:type:`int128`, `required`):
                    The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

            Raises:
                netaddr.core.AddrFormatError (Exception):
                    Raised when the passed str_val is not a valid ip string value.
        """
        return int(netaddr.IPAddress(str_val))

    @staticmethod
    def ip_version(str_val: str) -> int:
        r""" Returns the ip version (IPV4 or IPV6).
            arg:
                str_val (:tyep:`str`, `required):
                    The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

            Returns:
                int_val  (:type:`int128`, `required`):
                    The ip version (Either 4 or 6 for IPv4/IPv6)

            Raises:
                netaddr.core.AddrFormatError (Exception):
                    Raised when the passed str_val is not a valid ip string value.
        """
        return int(netaddr.IPAddress(str_val).version)

    @staticmethod
    def ip__str__(ip_type:int, ip_str:str, port:int):
        """ Return a formatted ip string
        """
        return "/ipv%i/%s:%i" % (ip_type, ip_str, port)

    @classmethod
    def is_valid_ip(cls, ip:str) -> bool:
        r""" Checks if an ip is valid.
            Args:
                ip  (:obj:`str` `required`):
                    The ip to check.

            Returns:
                valid  (:obj:`bool` `required`):
                    True if the ip is valid, False otherwise.
        """
        try:
            netaddr.IPAddress(ip)
            return True
        except Exception as e:
            return False

    @classmethod
    def external_ip(cls, default_ip='0.0.0.0') -> str:
        r""" Checks CURL/URLLIB/IPIFY/AWS for your external ip.
            Returns:
                external_ip  (:obj:`str` `required`):
                    Your routers external facing ip as a string.

            Raises:
                Exception(Exception):
                    Raised if all external ip attempts fail.
        """
        # --- Try curl.



        ip = None
        try:
            ip = c.cmd('curl -s ifconfig.me')
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            print(e)

        if cls.is_valid_ip(ip):
            return ip
        try:
            ip = requests.get('https://api.ipify.org').text
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            print(e)

        if cls.is_valid_ip(ip):
            return ip
        # --- Try AWS
        try:
            ip = requests.get('https://checkip.amazonaws.com').text.strip()
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            print(e)

        if cls.is_valid_ip(ip):
            return ip
        # --- Try myip.dnsomatic 
        try:
            process = os.popen('curl -s myip.dnsomatic.com')
            ip  = process.readline()
            assert isinstance(cls.ip_to_int(ip), int)
            process.close()
        except Exception as e:
            print(e)  

        if cls.is_valid_ip(ip):
            return ip
        # --- Try urllib ipv6 
        try:
            ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            print(e)

        if cls.is_valid_ip(ip):
            return ip
        # --- Try Wikipedia 
        try:
            ip = requests.get('https://www.wikipedia.org').headers['X-Client-IP']
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            print(e)

        if cls.is_valid_ip(ip):
            return ip

        return default_ip

    @staticmethod
    def upnpc_create_port_map(port: int):
        r""" Creates a upnpc port map on your router from passed external_port to local port.

            Args: 
                port (int, `required`):
                    The local machine port to map from your external port.

            Return:
                external_port (int, `required`):
                    The external port mappeclass to the local port on your machine.

            Raises:
                Exception (Exception):
                    Raised if UPNPC port mapping fails, for instance, if upnpc is not enabled on your router.
        """

        try:
            import miniupnpc
            upnp = miniupnpc.UPnP()
            upnp.discoverdelay = 200
            logger.debug('UPNPC: Using UPnP to open a port on your router ...')
            logger.debug('UPNPC: Discovering... delay={}ms', upnp.discoverdelay)
            ndevices = upnp.discover()
            upnp.selectigd()
            logger.debug('UPNPC: ' + str(ndevices) + ' device(s) detected')

            ip = upnp.lanaddr
            external_ip = upnp.externalipaddress()

            logger.debug('UPNPC: your local ip address: ' + str(ip))
            logger.debug('UPNPC: your external ip address: ' + str(external_ip))
            logger.debug('UPNPC: status = ' + str(upnp.statusinfo()) + " connection type = " + str(upnp.connectiontype()))

            # find a free port for the redirection
            external_port = port
            rc = upnp.getspecificportmapping(external_port, 'TCP')
            while rc != None and external_port < 65536:
                external_port += 1
                rc = upnp.getspecificportmapping(external_port, 'TCP')
            if rc != None:
                raise Exception("UPNPC: No available external ports for port mapping.")

            logger.info('UPNPC: trying to redirect remote: {}:{} => local: {}:{} over TCP', external_ip, external_port, ip, port)
            upnp.addportmapping(external_port, 'TCP', ip, port, 'Bittensor: %u' % external_port, '')
            logger.info('UPNPC: Create Success')

            return external_port

        except Exception as e:
            raise Exception(e) from e

    @classmethod
    def unreserve_port(cls,port:int, 
                       var_path='reserved_ports'):
        reserved_ports =  cls.get(var_path, {}, root=True)
        
        port_info = reserved_ports.pop(port,None)
        if port_info == None:
            port_info = reserved_ports.pop(str(port),None)
        
        output = {}
        if port_info != None:
            c.put(var_path, reserved_ports, root=True)
            output['msg'] = 'port removed'
        else:
            output['msg'] =  f'port {port} doesnt exist, so your good'

        output['reserved'] =  cls.reserved_ports()
        return output
    
    

    
    @classmethod
    def unreserve_ports(cls,*ports, 
                       var_path='reserved_ports' ):
        reserved_ports =  cls.get(var_path, {})
        if len(ports) == 0:
            # if zero then do all fam, tehe
            ports = list(reserved_ports.keys())
        elif len(ports) == 1 and isinstance(ports[0],list):
            ports = ports[0]
        ports = list(map(str, ports))
        reserved_ports = {rp:v for rp,v in reserved_ports.items() if not any([p in ports for p in [str(rp), int(rp)]] )}
        c.put(var_path, reserved_ports)
        return cls.reserved_ports()
    
    
    @classmethod
    def check_used_ports(cls, start_port = 8501, end_port = 8600, timeout=5):
        port_range = [start_port, end_port]
        used_ports = {}
        for port in range(*port_range):
            used_ports[port] = cls.port_used(port)
        return used_ports


    @classmethod
    def free_ports(cls, n=10, reserve:bool = False, random_selection:bool = False, **kwargs ) -> List[int]:
        free_ports = []
        avoid_ports = kwargs.pop('avoid_ports', [])
        for i in range(n):
            try:
                free_ports += [cls.free_port(reserve=reserve, 
                                            random_selection=random_selection, 
                                            avoid_ports=avoid_ports, **kwargs)]
            except Exception as e:
                c.print(f'Error: {e}', color='red')
                break
            avoid_ports += [free_ports[-1]]
        
              
        return free_ports
    
    @classmethod
    def random_port(cls, *args, **kwargs):
        return cls.choice(cls.free_ports(*args, **kwargs))
    

    

    @classmethod
    def free_port(cls, 
                  ports = None,
                  port_range: List[int] = None , 
                  ip:str =None, 
                  avoid_ports = None,
                  random_selection:bool = True) -> int:
        
        '''
        
        Get an availabldefe port within the {port_range} [start_port, end_poort] and {ip}
        '''
        avoid_ports = avoid_ports if avoid_ports else []
        
        if ports == None:
            port_range = cls.get_port_range(port_range)
            ports = list(range(*port_range))
            
        ip = ip if ip else c.default_ip

        if random_selection:
            ports = c.shuffle(ports)
        port = None
        for port in ports: 
            if port in avoid_ports:
                continue
            
            if cls.port_available(port=port, ip=ip):
                return port
            
        raise Exception(f'ports {port_range[0]} to {port_range[1]} are occupied, change the port_range to encompase more ports')

    get_available_port = free_port



    def check_used_ports(self, start_port = 8501, end_port = 8600, timeout=5):
        port_range = [start_port, end_port]
        used_ports = {}
        for port in range(*port_range):
            used_ports[port] = self.port_used(port)
        return used_ports
    


    @classmethod
    def resolve_port(cls, port:int=None, **kwargs):
        
        '''
        
        Resolves the port and finds one that is available
        '''
        if port == None or port == 0:
            port = cls.free_port(port, **kwargs)
            
        if c.port_used(port):
            port = cls.free_port(port, **kwargs)
            
        return int(port)

   

    @classmethod
    def port_available(cls, port:int, ip:str ='0.0.0.0'):
        return not cls.port_used(port=port, ip=ip)
        
   

    @classmethod
    def get_port_range(cls, port_range: list = None) -> list:
        if port_range == None:
            port_range = c.get('port_range', default=cls.default_port_range)
            
        if len(port_range) == 0:
            port_range = cls.default_port_range
        port_range = list(port_range)
        assert isinstance(port_range, list), 'Port range must be a list'
        assert isinstance(port_range[0], int), 'Port range must be a list of integers'
        assert isinstance(port_range[1], int), 'Port range must be a list of integers'
        return port_range
    



    @classmethod
    def port_used(cls, port: int, ip: str = '0.0.0.0', timeout: int = 1):
        import socket
        if not isinstance(port, int):
            return False
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set the socket timeout
            sock.settimeout(timeout)

            # Try to connect to the specified IP and port
            try:
                port=int(port)
                sock.connect((ip, port))
                return True
            except socket.error:
                return False
    
    @classmethod
    def port_free(cls, *args, **kwargs) -> bool:
        return not cls.port_used(*args, **kwargs)

    @classmethod
    def port_available(cls, port:int, ip:str ='0.0.0.0'):
        return not cls.port_used(port=port, ip=ip)
    
        

    @classmethod
    def used_ports(cls, ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
        '''
        Get availabel ports out of port range
        
        Args:
            ports: list of ports
            ip: ip address
        
        '''
        port_range = cls.resolve_port_range(port_range=port_range)
        if ports == None:
            ports = list(range(*port_range))
        
        async def check_port(port, ip):
            return cls.port_used(port=port, ip=ip)
        
        used_ports = []
        jobs = []
        for port in ports: 
            jobs += [check_port(port=port, ip=ip)]
                
        results = cls.gather(jobs)
        for port, result in zip(ports, results):
            if isinstance(result, bool) and result:
                used_ports += [port]
            
        return used_ports
    


    
    @classmethod
    def scan_ports(cls,host=None, start_port=1, end_port=50000):
        if host == None:
            host = cls.external_ip()
        import socket
        open_ports = []
        for port in range(start_port, end_port + 1):  # ports from start_port to end_port
            if cls.port_used(port=port, ip=host):
                print(f'Port {port} is open')
                open_ports.append(port)
            else:
                cls.print(f'Port {port} is closed', color='red')
        return open_ports

    @classmethod
    def resolve_port(cls, port:int=None, **kwargs):
        
        '''
        
        Resolves the port and finds one that is available
        '''
        if port == None or port == 0:
            port = cls.free_port(port, **kwargs)
            
        if cls.port_used(port):
            port = cls.free_port(port, **kwargs)
            
        return int(port)

    @classmethod
    def has_free_ports(self, n:int = 1, **kwargs):
        return len(self.free_ports(n=n, **kwargs)) > 0
    

    @classmethod
    def get_port_range(cls, port_range: list = None) -> list:
        if port_range == None:
            port_range = c.get('port_range', default=cls.default_port_range)
            
        if len(port_range) == 0:
            port_range = cls.default_port_range
        port_range = list(port_range)
        assert isinstance(port_range, list), 'Port range must be a list'
        assert isinstance(port_range[0], int), 'Port range must be a list of integers'
        assert isinstance(port_range[1], int), 'Port range must be a list of integers'
        return port_range
    
    @classmethod
    def port_range(cls):
        return cls.get_port_range()
    
    @classmethod
    def resolve_port_range(cls, port_range: list = None) -> list:
        return cls.get_port_range(port_range)

    

    @classmethod
    def set_port_range(cls, *port_range: list):
        if len(port_range) ==0 :
            port_range = cls.default_port_range
        elif len(port_range) == 1:
            if port_range[0] == None:
                port_range = cls.default_port_range

        assert len(port_range) == 2, 'Port range must be a list of two integers'        
        for port in port_range:
            assert isinstance(port, int), f'Port {port} range must be a list of integers'
        assert port_range[0] < port_range[1], 'Port range must be a list of integers'
                
        c.put('port_range', port_range)
        return port_range
    
    @classmethod
    def get_port(cls, port:int = None)->int:
        port = port if port is not None and port != 0 else cls.free_port()
        while cls.port_used(port):
            port += 1   
        return port 
    

    
    @classmethod
    def port_free(cls, *args, **kwargs) -> bool:
        return not cls.port_used(*args, **kwargs)


    @classmethod
    def port_available(cls, port:int, ip:str ='0.0.0.0'):
        return not cls.port_used(port=port, ip=ip)
        
    @classmethod
    def used_ports(cls, ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
        '''
        Get availabel ports out of port range
        
        Args:
            ports: list of ports
            ip: ip address
        
        '''
        port_range = cls.resolve_port_range(port_range=port_range)
        if ports == None:
            ports = list(range(*port_range))
        
        async def check_port(port, ip):
            return c.port_used(port=port, ip=ip)
        
        used_ports = []
        jobs = []
        for port in ports: 
            jobs += [check_port(port=port, ip=ip)]
                
        results = cls.gather(jobs)
        for port, result in zip(ports, results):
            if isinstance(result, bool) and result:
                used_ports += [port]
            
        return used_ports
    

    get_used_ports = used_ports
    

    

    @classmethod
    def get_available_ports(cls, port_range: List[int] = None , ip:str =None) -> int:
        port_range = cls.resolve_port_range(port_range)
        ip = ip if ip else c.default_ip
        
        available_ports = []
        # return only when the port is available
        for port in range(*port_range): 
            if not cls.port_used(port=port, ip=ip):
                available_ports.append(port)
                  
        return available_ports
    available_ports = get_available_ports
    