# backend/tools/__init__.py

import logging

# Import names that server.py and other modules might be expecting
# directly from the backend.tools package.

from .standard_tools import (
    get_dynamic_tools, # This function internally uses load_tools_from_config
    TEXT_EXTENSIONS,
    # Specific tool classes defined in standard_tools.py if they need to be directly accessible
    # e.g., ReadFileTool, WriteFileTool, TaskWorkspaceShellTool (though these are now loaded via config)
)

# MODIFIED: Import ToolLoadingError from tool_loader and other necessary components
from backend.tool_loader import (
    get_task_workspace_path, 
    BASE_WORKSPACE_ROOT,
    ToolLoadingError # Import the exception class
)


logger = logging.getLogger(__name__)
logger.debug("backend.tools package initialized.")

# Optionally, define __all__ if you want to be explicit about what 'from backend.tools import *' imports
__all__ = [
    "get_dynamic_tools",
    "TEXT_EXTENSIONS",
    "get_task_workspace_path",
    "BASE_WORKSPACE_ROOT",
    "ToolLoadingError" # Add ToolLoadingError to __all__
]
