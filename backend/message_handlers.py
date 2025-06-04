# backend/message_handlers.py
# This file now acts as a central point for importing and re-exporting
# all specific message processing functions from the message_processing sub-package.

import logging

# Import all names defined in the __all__ variable of the message_processing sub-package's __init__.py
# This ensures that any function added to message_processing.__all__ is available here.
from .message_processing import * # noqa F403: Ok here as we are re-exporting based on __all__

logger = logging.getLogger(__name__)
logger.info("message_handlers.py: All processing functions re-exported from message_processing sub-package.")

# To be explicit about what this module exports (which will be everything from message_processing.__all__),
# you can dynamically build __all__ here too, though the wildcard import with a well-defined
# __all__ in the sub-package is common for this pattern.
#
# Example of dynamically building __all__ (optional, as the wildcard import above achieves this if
# message_processing.__init__ has __all__ defined):
#
# from . import message_processing
# __all__ = message_processing.__all__
#
# For simplicity and common practice, the wildcard import is often used when the imported
# module carefully defines its __all__.

# The actual function definitions are now in their respective files
# within the backend/message_processing/ directory.
# This file primarily serves to gather them for easier import by server.py.
