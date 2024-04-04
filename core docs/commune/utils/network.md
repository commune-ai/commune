# Bittensor's IP Utility

This Python module provides utilities for handling IP addresses, including converting between integer and IP string representations, determining the IP version (IPV4 or IPV6), and obtaining the external IP of the machine.

## Functions:

### `int_to_ip(int_val: int) -> str`

This function maps an integer to a unique IP string.

### `ip_to_int(str_val: str) -> int`

This function maps an IP string to a unique integer representation.

### `ip_version(str_val: str) -> int`

This function returns the IP version (either IPV4 or IPV6) of the provided IP string.

### `ip__str__(ip_type:int, ip_str:str, port:int)`

This function returns a formatted IP string.

### `get_external_ip() -> str`

This function checks CURL, URLLIB, IPIFY, AWS and other services for the user's external IP.

### `upnpc_create_port_map(port: int)`

This function creates a UPnP port map on a router from the passed external port to the local machine port.

## Exception classes:

### `ExternalIPNotFound`

Raised if we cannot obtain your external IP from CURL/URLLIB/IPIFY/AWS.

### `UPNPCException`

Raised when trying to perform a port mapping on your router.

## Usage:

These utilities are useful in scenarios that involve handling of IP addresses and network port mapping. Notably, they help in the retrieval of external IPs from various web services and enable UPnP port mapping on a router. These are common tasks in networking and distributed systems. Please note that error handling is included for when these tasks fail.