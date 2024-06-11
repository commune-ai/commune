# hey/thanks bittensor
import os
import urllib

from loguru import logger

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
    import netaddr
    return int(netaddr.IPAddress(str_val))

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

def ip__str__(ip_type:int, ip_str:str, port:int):
    """ Return a formatted ip string
    """
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)

class ExternalIPNotFound(Exception):
    """ Raised if we cannot attain your external ip from CURL/URLLIB/IPIFY/AWS """

def external_ip() -> str:
    r""" Checks CURL/URLLIB/IPIFY/AWS for your external ip.
        Returns:
            external_ip  (:obj:`str` `required`):
                Your routers external facing ip as a string.

        Raises:
            ExternalIPNotFound (Exception):
                Raised if all external ip attempts fail.
    """
    # --- Try curl.
    ip = None
    try:
        ip = c.cmd('curl -s ifconfig.me')
    except Exception as e:
        c.print(e)

    # --- Try ipify
    try:
        ip = requests.get('https://api.ipify.org').text
        assert isinstance(ip_to_int(ip), int)
    except Exception:
        c.print(e)

    # --- Try AWS
    try:
        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        assert isinstance(ip_to_int(ip), int)
    except Exception:
        c.print(e)

    # --- Try myip.dnsomatic 
    try:
        process = os.popen('curl -s myip.dnsomatic.com')
        ip  = process.readline()
        process.close()
        assert isinstance(ip_to_int(ip), int)
    except Exception:
        pass    

    # --- Try urllib ipv6 
    try:
        external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try Wikipedia 
    try:
        external_ip = requests.get('https://www.wikipedia.org').headers['X-Client-IP']
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    raise ExternalIPNotFound


class UPNPCException(Exception):
    """ Raised when trying to perform a port mapping on your router. """


def upnpc_create_port_map(port: int):
    r""" Creates a upnpc port map on your router from passed external_port to local port.

        Args: 
            port (int, `required`):
                The local machine port to map from your external port.

        Return:
            external_port (int, `required`):
                The external port mapped to the local port on your machine.

        Raises:
            UPNPCException (Exception):
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
            raise UPNPCException("UPNPC: No available external ports for port mapping.")

        logger.info('UPNPC: trying to redirect remote: {}:{} => local: {}:{} over TCP', external_ip, external_port, ip, port)
        upnp.addportmapping(external_port, 'TCP', ip, port, 'Bittensor: %u' % external_port, '')
        logger.info('UPNPC: Create Success')

        return external_port

    except Exception as e:
        raise UPNPCException(e) from e
