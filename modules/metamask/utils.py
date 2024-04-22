def format_eth_address(address):
    """Format Ethereum address for display."""
    return address[:6] + '...' + address[-4:]