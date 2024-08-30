class ChainTransactionError(Exception):
    """Error for any chain transaction related errors."""


class NetworkError(Exception):
    """Base for any network related errors."""


class NetworkQueryError(NetworkError):
    """Network query related error."""


class NetworkTimeoutError(NetworkError):
    """Timeout error"""
