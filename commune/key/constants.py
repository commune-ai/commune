from ecdsa.curves import SECP256k1

SS58_FORMAT = 42

ERROR_INVALID_KEY = "Invalid key provided."
ERROR_KEY_GENERATION_FAILED = "Key generation failed."
ERROR_KEY_VALIDATION_FAILED = "Key validation failed."

DEV_PHRASE = 'bottom drive obey lake curtain smoke basket hold race lonely fit walk'

JUNCTION_ID_LEN = 32
RE_JUNCTION = r'(\/\/?)([^/]+)'

NONCE_LENGTH = 24
SCRYPT_LENGTH = 32 + (3 * 4)
PKCS8_DIVIDER = bytes([161, 35, 3, 33, 0])
PKCS8_HEADER = bytes([48, 83, 2, 1, 1, 48, 5, 6, 3, 43, 101, 112, 4, 34, 4, 32])
PUB_LENGTH = 32
SALT_LENGTH = 32
SEC_LENGTH = 64
SEED_LENGTH = 32

SCRYPT_N = 1 << 15
SCRYPT_P = 1
SCRYPT_R = 8

BIP39_PBKDF2_ROUNDS = 2048
BIP39_SALT_MODIFIER = "mnemonic"
BIP32_PRIVDEV = 0x80000000
BIP32_CURVE = SECP256k1
BIP32_SEED_MODIFIER = b"Bitcoin seed"
ETH_DERIVATION_PATH = "m/44'/60'/0'/0"