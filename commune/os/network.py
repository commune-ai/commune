# hey/thanks bittensor
import os
import urllib
import commune as c
import requests
import logging


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
        import netaddr
        return int(netaddr.IPAddress(str_val).version)

    @staticmethod
    def ip__str__(ip_type:int, ip_str:str, port:int):
        """ Return a formatted ip string
        """
        return "/ipv%i/%s:%i" % (ip_type, ip_str, port)


    @classmethod
    def external_ip(cls,verbose: bool = False, default_ip='0.0.0.0') -> str:
        r""" Checks CURL/URLLIB/IPIFY/AWS for your external ip.
            Returns:
                external_ip  (:obj:`str` `required`):
                    Your routers external facing ip as a string.

            Raises:
                Exception(Exception):
                    Raised if all external ip attempts fail.
        """
        # --- Try curl.
        ip = '0.0.0.0'
        try:
            ip = c.cmd('curl -s ifconfig.me')
            assert isinstance(cls.ip_to_int(ip), int)
            c.print(ip, 'ifconfig.me', verbose=verbose)
        except Exception as e:
            c.print(e, verbose=verbose)

        try:
            ip = requests.get('https://api.ipify.org').text
            assert isinstance(cls.ip_to_int(ip), int)
            c.print(ip, 'ipify.org', verbose=verbose)
        except Exception as e:
            c.print(e, verbose=verbose)

        # --- Try AWS
        try:
            ip = requests.get('https://checkip.amazonaws.com').text.strip()
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            c.print(e, verbose=verbose)

        # --- Try myip.dnsomatic 
        try:
            process = os.popen('curl -s myip.dnsomatic.com')
            ip  = process.readline()
            assert isinstance(cls.ip_to_int(ip), int)
            process.close()
        except Exception as e:
            c.print(e, verbose=verbose)    

        # --- Try urllib ipv6 
        try:
            ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            c.print(e, verbose=verbose)

        # --- Try Wikipedia 
        try:
            ip = requests.get('https://www.wikipedia.org').headers['X-Client-IP']
            assert isinstance(cls.ip_to_int(ip), int)
        except Exception as e:
            c.print(e, verbose=verbose)

        if len(ip) == 0 or ip == None:
            ip = default_ip
        
        return ip

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
