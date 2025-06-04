# backend/tools/python_repl_tool.py
import asyncio
import logging
from typing import Optional, Type, Any, Dict

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import CallbackManagerForToolRun
# MODIFIED: Using pydantic.v1 for BaseModel and Field as per project context
from pydantic.v1 import BaseModel, Field 

try:
    from langchain_experimental.utilities import PythonREPL
except ImportError:
    PythonREPL = None # type: ignore
    logging.getLogger(__name__).warning(
        "Could not import PythonREPL from langchain_experimental.utilities. "
        "The PythonREPLTool will not be available."
    )

logger = logging.getLogger(__name__)

class PythonREPLInput(BaseModel):
    command: str = Field(description="A valid Python command or multi-line script.")

class PythonREPLTool(BaseTool):
    """
    A tool that executes Python code in a REPL environment.
    Input should be a string containing the Python code to execute.
    The tool will return the standard output (stdout) of the executed code.
    If the code is an expression that has a non-None result, its string
    representation might be returned if no stdout is produced (behavior
    dependent on the underlying PythonREPL utility).
    To ensure output, use `print()` statements in your code.
    """
    name: str = "Python_REPL"
    description: str = (
        "A Python REPL. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`. The tool returns what is printed to standard output."
    )
    args_schema: Type[BaseModel] = PythonREPLInput
    
    # _repl will be initialized as an instance attribute in __init__
    # It is not a Pydantic field.

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs) 
        
        self._repl: Optional[PythonREPL] = None # Initialize as instance attribute
        if PythonREPL is not None:
            try:
                self._repl = PythonREPL() 
                logger.info(f"PythonREPLTool: Initialized self._repl. Type: {type(self._repl)}")
            except Exception as e:
                logger.error(f"Failed to initialize PythonREPL in PythonREPLTool: {e}", exc_info=True)
                self._repl = None 
        else:
            logger.warning("PythonREPL utility not available. PythonREPLTool will not function.")


    def _run(
        self,
        command: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self._repl is None:
            msg = "PythonREPLTool is not available because the PythonREPL utility could not be initialized."
            logger.error(msg)
            raise ToolException(msg)
            
        logger.info(f"Tool '{self.name}' synchronously executing command (first 200 chars): '{command[:200]}'")
        try:
            result = self._repl.run(command)
            output_str = str(result) if result is not None else ""
            logger.info(f"Tool '{self.name}' sync command executed. Output length: {len(output_str)}")
            return output_str
        except Exception as e:
            logger.error(f"Error in PythonREPLTool _run for command '{command[:200]}': {e}", exc_info=True)
            return f"Error executing Python REPL command: {type(e).__name__} - {str(e)}"

    async def _arun(
        self,
        command: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        if self._repl is None:
            msg = "PythonREPLTool is not available because the PythonREPL utility could not be initialized."
            logger.error(msg)
            raise ToolException(msg)

        logger.info(f"Tool '{self.name}' asynchronously executing command (first 200 chars): '{command[:200]}'")
        try:
            result = await asyncio.to_thread(self._repl.run, command)
            output_str = str(result) if result is not None else ""
            logger.info(f"Tool '{self.name}' async command executed. Output length: {len(output_str)}")
            return output_str
        except Exception as e:
            logger.error(f"Error in PythonREPLTool _arun for command '{command[:200]}': {e}", exc_info=True)
            return f"Error executing Python REPL command asynchronously: {type(e).__name__} - {str(e)}"

if __name__ == '__main__':
    async def test_repl_tool():
        logging.basicConfig(level=logging.INFO)
        tool = PythonREPLTool()
        if tool._repl is None: # Check the instance attribute
            print("PythonREPLTool could not be initialized. Test skipped.")
            return

        commands_to_test = [
            "print('Hello from REPL Tool')",
            "x = 5\ny = 10\nprint(x+y)",
            "1+1", 
            "repr('Python is fun!')", 
            "print(repr('Python is fun!'))" 
        ]
        for cmd in commands_to_test:
            print(f"\nTesting command: {cmd}")
            output = await tool.arun(command=cmd) # Pass command as a keyword argument
            print(f"Output:\n>>>\n{output}\n<<<")

    asyncio.run(test_repl_tool())
