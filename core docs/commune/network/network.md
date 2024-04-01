# README

This python code contains a class Network which has several static and class methods for handling IP related operations. These operations predominantly include converting IP address (in string format) to integer and vice versa, getting the version of IP (IPv4 or IPv6), creating a formatted IP string, and getting the external IP address of the system.

The methods included in this class are:
- `int_to_ip()`: Converts an integer to a unique IP address string. Throws an exception if the input integer is not valid.
- `ip_to_int()`: Converts an IP string to a unique integer. Throws an exception if the input IP string is not valid.
- `ip_version()`: Returns the version of the IP (either IPv4 or IPv6). Throws an exception if the input IP string is not valid.
- `ip__str__()`: Returns a formatted IP string.
- `get_external_ip()`: Checks for the external IP of your system using various services like CURL/URLLIB/IPIFY/AWS etc. Throws an exception if all external IP attempts fail.
- `upnpc_create_port_map()`: Creates a UPNP port map on your router. Returns the external port mapped to the local port on your machine. Throws an exception if UPNP port mapping fails.
- `unreserve_port()`: Removes a port from the reserved ports list.

This class could be useful in handling complex networking operations in a simplified manner. Use it as a module in your networking application to manage IP related activities efficiently and effectively. 

Make sure to properly handle exceptions as they provide information about any errors that may occur during execution.