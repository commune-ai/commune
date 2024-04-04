# Socket Module

This module helps in creating, managing, and testing sockets using Python.

## Codes & Functions

The Socket Module contains the following functions:

1. `__init__`: This is the initialization function of the Socket class. It takes two arguments - a and b, both integers with default value as 1 and 2 respectively.

2. `call`: This function is used to call the Socket object. The function takes two integer arguments x and y with default values 1 and 2 and returns their sum.

3. `connect`: This is a class method used to connect to the specified IP and Port with a set timeout. It takes three arguments- IP( a string), port ( an integer; default is 8888), timeout ( an integer; default is 1).

4. `send`: This is a class method that sends data to the specified port and IP. It takes three arguments - data, port(an integer; default is 8888), and IP(a string; default is '0.0.0.0').

5. `receive`: This method receives data from the specified port and IP. It takes five arguments - port(an integer; default is 8888), ip ( a string; default is '0.0.0.0'), timeout (an integer; default is 1), and size(an integer; default is 1024). It returns the received data.

6. `serve`: This method binds and listens for connections to the specified port and IP. It takes a port number ( an integer; default is 8889), and an IP ( a string; default is '0.0.0.0') as argument.

## How to Use

1. Import the module using: `import commune as c`.

2. Create a new instance of the Socket class.

3. Call the desired methods for socket connections and operations. For example, `Socket.connect(ip='your_ip', port='your_port')`.
