class NeuralCorpError(Exception):
    """Base class for all exceptions in NeuralCorp."""

    pass


class FetchError(NeuralCorpError):
    """Raised when an async HTTP request fails."""

    def __init__(
        self, url: str, status_code: int, message: str = "Failed to fetch data from URL"
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.message = f"{message}: {url} (Status code: {status_code})"
        super().__init__(self.message)


class ConfigurationError(NeuralCorpError):
    """Raised when there is an issue with the configuration of NeuralCorp."""

    pass
