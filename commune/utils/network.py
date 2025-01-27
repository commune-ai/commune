
from typing import *

def port_free( *args, **kwargs) -> bool:
    return not port_used(*args, **kwargs)

def port_available(port:int, ip:str ='0.0.0.0'):
    return not port_used(port=port, ip=ip)

def used_ports(ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
    '''
    Get availabel ports out of port range
    
    Args:
        ports: list of ports
        ip: ip address
    
    '''
    import commune as c
    port_range = resolve_port_range(port_range=port_range)
    if ports == None:
        ports = list(range(*port_range))
    
    async def check_port(port, ip):
        return port_used(port=port, ip=ip)
    
    used_ports = []
    jobs = []
    for port in ports: 
        jobs += [check_port(port=port, ip=ip)]
            
    results = c.wait(jobs)
    for port, result in zip(ports, results):
        if isinstance(result, bool) and result:
            used_ports += [port]
        
    return used_ports


def resolve_ip(ip=None, external:bool=True) -> str:
    if ip == None:
        if external:
            ip = external_ip()
        else:
            ip = '0.0.0.0'
    assert isinstance(ip, str)
    return ip



def resolve_port(port:int=None, **kwargs):
    '''
    Resolves the port and finds one that is available
    '''
    if port == None or port == 0:
        port = free_port(port, **kwargs)
    if port_used(port):
        port = free_port(port, **kwargs)
    return int(port)



def get_available_ports(port_range: List[int] = None , ip:str =None) -> int:
    import commune as c
    port_range = c.resolve_port_range(port_range)
    ip = ip if ip else '0.0.0.0'
    available_ports = []
    # return only when the port is available
    for port in range(*port_range): 
        if not c.port_used(port=port, ip=ip):
            available_ports.append(port)         
    return available_ports
available_ports = get_available_ports


def resolve_port(port:int=None, **kwargs):
    '''
    Resolves the port and finds one that is available
    '''
    if port == None or port == 0:
        port = free_port(port, **kwargs)
        
    if port_used(port):
        port = free_port(port, **kwargs)
        
    return int(port)



def ip(max_age=None, update:bool = False, **kwargs) -> str:
    
    try:
        import commune as c
        path = 'ip'
        ip = c.get(path, None, max_age=max_age, update=update)
        if ip == None:
            ip = external_ip()
            c.put(path, ip)
    except Exception as e:
        print('Error while getting IP')
        return '0.0.0.0'
    return ip


def has_free_ports(n:int = 1, **kwargs):
    return len(free_ports(n=n, **kwargs)) > 0


def ip_version(str_val: str) -> int:
    import netaddr
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

def ip__str__(ip_type:int, ip_str:str, port:int):
    """ Return a formatted ip string
    """
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)

def external_ip( default_ip='0.0.0.0') -> str:
    import commune as c
    
    r""" Checks CURL/URLLIB/IPIFY/AWS for your external ip.
        Returns:
            external_ip  (:obj:`str` `required`):
                Your routers external facing ip as a string.

        Raises:
            Exception(Exception):
                Raised if all external ip attempts fail.
    """
    ip = None
    try:
        ip = c.cmd('curl -s ifconfig.me')
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    try:
        ip = requests.get('https://api.ipify.org').text
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    # --- Try AWS
    try:
        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    # --- Try myip.dnsomatic 
    try:
        process = os.popen('curl -s myip.dnsomatic.com')
        ip  = process.readline()
        assert isinstance(c.ip_to_int(ip), int)
        process.close()
    except Exception as e:
        print(e)  

    if is_valid_ip(ip):
        return ip
    # --- Try urllib ipv6 
    try:
        ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    # --- Try Wikipedia 
    try:
        ip = requests.get('https://www.wikipedia.org').headers['X-Client-IP']
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip

    return default_ip


def unreserve_port(port:int, 
                    var_path='reserved_ports'):
    import commune as c
    reserved_ports =  c.get(var_path, {}, root=True)
    
    port_info = reserved_ports.pop(port,None)
    if port_info == None:
        port_info = reserved_ports.pop(str(port),None)
    
    output = {}
    if port_info != None:
        c.put(var_path, reserved_ports, root=True)
        output['msg'] = 'port removed'
    else:
        output['msg'] =  f'port {port} doesnt exist, so your good'

    output['reserved'] =  c.reserved_ports()
    return output

def unreserve_ports(*ports, var_path='reserved_ports' ):
    import commune as c
    reserved_ports =  c.get(var_path, {})
    if len(ports) == 0:
        # if zero then do all fam, tehe
        ports = list(reserved_ports.keys())
    elif len(ports) == 1 and isinstance(ports[0],list):
        ports = ports[0]
    ports = list(map(str, ports))
    reserved_ports = {rp:v for rp,v in reserved_ports.items() if not any([p in ports for p in [str(rp), int(rp)]] )}
    c.put(var_path, reserved_ports)
    return c.reserved_ports()

def kill_port(port:int):
    r""" Kills a process running on the passed port.
        Args:
            port  (:obj:`int` `required`):
                The port to kill the process on.
    """
    try:
        os.system(f'kill -9 $(lsof -t -i:{port})')
    except Exception as e:
        print(e)
        return False
    return True

def kill_ports(ports = None, *more_ports):
    ports = ports or used_ports()
    if isinstance(ports, int):
        ports = [ports]
    if '-' in ports:
        ports = list(range([int(p) for p in ports.split('-')]))
    ports = list(ports) + list(more_ports)
    for port in ports:
        kill_port(port)
    return check_used_ports()

def is_port_public(port:int, ip:str=None, timeout=0.5):
    import socket
    ip = ip or ip()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((ip, port)) == 0
        
def public_ports(timeout=1.0):
    import commune as c
    futures = []
    for port in free_ports():
        c.print(f'Checking port {port}')
        futures += [c.submit(is_port_public, {'port':port}, timeout=timeout)]
    results =  c.wait(futures, timeout=timeout)
    results = list(map(bool, results))
    return results

def free_ports(n=10, random_selection:bool = False, **kwargs ) -> List[int]:
    free_ports = []
    avoid_ports = kwargs.pop('avoid_ports', [])
    for i in range(n):
        try:
            free_ports += [free_port(  random_selection=random_selection, 
                                        avoid_ports=avoid_ports, **kwargs)]
        except Exception as e:
            print(f'Error: {e}')
            break
        avoid_ports += [free_ports[-1]]
 
    return free_ports

def random_port(*args, **kwargs):
    import commune as c
    return c.choice(c.free_ports(*args, **kwargs))


def free_port(ports = None,
                port_range: List[int] = None , 
                ip:str =None, 
                avoid_ports = None,
                random_selection:bool = True) -> int:
    import commune as c
    
    '''
    
    Get an availabldefe port within the {port_range} [start_port, end_poort] and {ip}
    '''
    avoid_ports = avoid_ports if avoid_ports else []
    
    if ports == None:
        port_range = c.get_port_range(port_range)
        ports = list(range(*port_range))
        
    ip = ip if ip else '0.0.0.0'

    if random_selection:
        ports = c.shuffle(ports)
    port = None
    for port in ports: 
        if port in avoid_ports:
            continue
        if c.port_available(port=port, ip=ip):
            return port
    raise Exception(f'ports {port_range[0]} to {port_range[1]} are occupied, change the port_range to encompase more ports')

get_available_port = free_port


get_used_ports = used_ports



def used_ports(ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
    import commune as c
    '''
    Get availabel ports out of port range
    
    Args:
        ports: list of ports
        ip: ip address
    
    '''
    port_range = resolve_port_range(port_range=port_range)
    if ports == None:
        ports = list(range(*port_range))
    
    async def check_port(port, ip):
        return port_used(port=port, ip=ip)
    
    used_ports = []
    jobs = []
    for port in ports: 
        jobs += [check_port(port=port, ip=ip)]
            
    results = c.gather(jobs)
    for port, result in zip(ports, results):
        if isinstance(result, bool) and result:
            used_ports += [port]
        
    return used_ports


def port_free(*args, **kwargs) -> bool:
    return not port_used(*args, **kwargs)


def get_port(port:int = None)->int:
    port = port if port is not None and port != 0 else free_port()
    while port_used(port):
        port += 1   
    return port 

def port_range():
    return get_port_range()

def ports() -> List[int]:
    
    return list(range(*get_port_range()))

def resolve_port_range(port_range: list = None) -> list:
    return get_port_range(port_range)

def set_port_range(*port_range: list):
    import commune as c
    if '-' in port_range[0]:
        port_range = list(map(int, port_range[0].split('-')))
    if len(port_range) ==0 :
        port_range = c.default_port_range
    elif len(port_range) == 1:
        if port_range[0] == None:
            port_range = c.default_port_range
    assert len(port_range) == 2, 'Port range must be a list of two integers'        
    for port in port_range:
        assert isinstance(port, int), f'Port {port} range must be a list of integers'
    assert port_range[0] < port_range[1], 'Port range must be a list of integers'
    c.put('port_range', port_range)
    return port_range


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

def get_port_range(port_range: list = None) -> list:
    import commune as c
    port_range = c.get('port_range', c.default_port_range)
    if isinstance(port_range, str):
        port_range = list(map(int, port_range.split('-')))
    if len(port_range) == 0:
        port_range = c.default_port_range
    port_range = list(port_range)
    assert isinstance(port_range, list), 'Port range must be a list'
    assert isinstance(port_range[0], int), 'Port range must be a list of integers'
    assert isinstance(port_range[1], int), 'Port range must be a list of integers'
    return port_range




def is_valid_ip(ip:str) -> bool:
    import netaddr
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

def check_used_ports(start_port = 8501, end_port = 8600, timeout=5):
    import commune as c
    port_range = [start_port, end_port]
    used_ports = {}
    for port in range(*port_range):
        used_ports[port] = c.port_used(port)
    return used_ports

def port_used( port: int, ip: str = '0.0.0.0', timeout: int = 1):
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



def is_url( address:str) -> bool:
    import commune as c
    if not isinstance(address, str):
        return False
    if '://' in address:
        return True
    conds = []
    conds.append(isinstance(address, str))
    conds.append(':' in address)
    conds.append(c.is_int(address.split(':')[-1]))
    return all(conds)
