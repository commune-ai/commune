    def hex2str( hex_str):
        return bytes.fromhex(hex_str).decode()

    
    def str2hex( string):
        from hexbytes.main import HexBytes
        return HexBytes(string).hex()
