class ChainTransactionError(Exception):
    """Error for any chain transaction related errors."""


class NetworkError(Exception):
    """Base for any network related errors."""


class NetworkQueryError(NetworkError):
    """Network query related error."""


class NetworkTimeoutError(NetworkError):
    """Timeout error"""


class PasswordError(Exception):
    """Password related error."""


class PasswordNotProvidedError(PasswordError):
    """Password is not provided."""


class InvalidPasswordError(PasswordError):
    """Password is invalid."""


class KeyNotFoundError(Exception):
    """Key not found error."""
