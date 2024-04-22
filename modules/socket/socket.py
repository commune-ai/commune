import commune as c
import socket

class Socket(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    @classmethod
    def connect(cls,ip: str = '0.0.0.0', port: int = 8888, timeout: int = 1):
        if  not isinstance(ip, str):
            raise TypeError('ip must be a string')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set the socket timeout
            sock.settimeout(timeout)

            # Try to connect to the specified IP and port
            try:
                sock.connect((ip, port))
            except Exception as e:

                # If the connection fails, return False
                return False    
            
    
    @classmethod
    def send(cls,data, port=8888, ip: str = '0.0.0.0'):
        socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.connect((ip, port))
        socket.send(data)
        socket.close()
    
    def receive(self, port=8888, ip: str = '0.0.0.0', timeout: int = 1, size: int = 1024):
        socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.connect((ip, port))
        data = socket.recv(size)
        socket.close()
        return data
    

    def serve(self, port=8889, ip: str = '0.0.0.0'):
        import socket
        socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.bind((ip, port))
        socket.listen(1)
        conn, addr = socket.accept()
        c.print('Connected by', addr)
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall(data)

        