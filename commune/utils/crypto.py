def is_mnemonic(s: str) -> bool:
    import re
    # Match 12 or 24 words separated by spaces
    return bool(re.match(r'^(\w+ ){11}\w+$', s)) or bool(re.match(r'^(\w+ ){23}\w+$', s))

def is_private_key(s: str) -> bool:
    import re
    # Match a 64-character hexadecimal string
    pattern = r'^[0-9a-fA-F]{64}$'
    return bool(re.match(pattern, s))