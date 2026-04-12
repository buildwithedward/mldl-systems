"""Public API for the neuralcorp.utils sub-package.

Importing `from neuralcorp.utils import get_logger` works because
we explicitly re-export it here. This shields callers from the
internal module layout.
"""

from neuralcorp.utils.logger import get_logger_name as get_logger

# Declare the public API — only these names are exported on `import *`
__all__ = ["get_logger"]
