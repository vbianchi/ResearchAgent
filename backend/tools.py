# backend/tools.py
import logging
import httpx
from bs4 import BeautifulSoup
from pathlib import Path
import os
import re
import aiofiles
import codecs
import asyncio
import sys
from typing import List, Optional, Dict, Any
import functools

# LangChain Tool Imports
from langchain_core.tools import Tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from Bio import Entrez
from urllib.error import HTTPError

# PDF Import
try:
    import pypdf
    logger = logging.getLogger(__name__) # Define logger early
except ImportError:
    logger = logging.getLogger(__name__) # Define logger even if import fails
    logger.warning("pypdf not installed. PDF reading functionality will be unavailable.")
    pypdf = None

# Project Imports
from backend.config import settings

# --- Define Base Workspace Path ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BASE_WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
    os.makedirs(BASE_WORKSPACE_ROOT, exist_ok=True)
    logger.info(f"Base workspace directory ensured at: {BASE_WORKSPACE_ROOT}")
except OSError as e:
    logger.error(f"Could not create base workspace directory: {e}", exc_info=True)
    raise OSError(f"Required base workspace directory {BASE_WORKSPACE_ROOT} could not be created.") from e
except Exception as e:
    logger.error(f"Error resolving project/workspace path: {e}", exc_info=True)
    raise

# Define recognizable text extensions (used as fallback)
TEXT_EXTENSIONS = {".txt", ".py", ".js", ".css", ".html", ".json", ".csv", ".md", ".log", ".yaml", ".yml"}


# --- Helper Function to get Task-Specific Workspace ---
def get_task_workspace_path(task_id: Optional[str]) -> Path:
    """
    Constructs and ensures the path for a specific task's workspace.
    Raises ValueError/OSError on invalid ID or creation failure.
    """
    if not task_id or not isinstance(task_id, str):
        msg = f"Invalid or missing task_id ('{task_id}') provided for workspace path."
        logger.error(msg)
        raise ValueError(msg)
    if ".." in task_id or "/" in task_id or "\\" in task_id:
       msg = f"Invalid characters detected in task_id: {task_id}. Denying workspace path creation."
       logger.error(msg)
       raise ValueError(msg)
    task_workspace = BASE_WORKSPACE_ROOT / task_id
    try:
        os.makedirs(task_workspace, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create task workspace directory at {task_workspace}: {e}", exc_info=True)
        raise OSError(f"Could not create task workspace {task_workspace}: {e}") from e
    return task_workspace


# --- Tool Implementation Functions ---

async def fetch_and_parse_url(url: str) -> str:
    """Fetches and parses URL content using configured limits."""
    tool_name = "web_page_reader"
    logger.info(f"Tool '{tool_name}' received raw input: '{url}'") # Log raw input

    # Input Validation
    if not isinstance(url, str) or not url.strip():
        logger.error(f"Tool '{tool_name}' received invalid input: Must be a non-empty string.")
        return "Error: Invalid input. Expected a non-empty URL string."

    max_length = settings.tool_web_reader_max_length
    timeout = settings.tool_web_reader_timeout
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    clean_url = url.strip().replace('\n', '').replace('\r', '').replace('\t', '').strip('`')
    if not clean_url:
        logger.error(f"Tool '{tool_name}' input resulted in empty URL after cleaning.")
        return "Error: Received an empty URL after cleaning."

    # Basic URL format check (allows more flexibility than just http/https)
    if not re.match(r"^[a-zA-Z]+://", clean_url):
        # Prepend https:// if no scheme is present
        logger.info(f"Tool '{tool_name}': No scheme found, prepending https:// to '{clean_url}'")
        clean_url = f"https://{clean_url}"

    logger.info(f"Tool '{tool_name}' attempting to fetch and parse cleaned URL: {clean_url} (Timeout: {timeout}s)")
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=HEADERS) as client:
            response = await client.get(clean_url); response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "html" not in content_type:
                logger.warning(f"Tool '{tool_name}': Cannot parse content type '{content_type}' for URL {clean_url}")
                return f"Error: Cannot parse content type '{content_type}'. Only HTML is supported."
            html_content = response.text; soup = BeautifulSoup(html_content, 'lxml')
            content_tags = soup.find('article') or soup.find('main') or soup.find('body')
            if content_tags: texts = content_tags.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']); extracted_text = "\n".join(t.get_text(strip=True) for t in texts if t.get_text(strip=True))
            else: extracted_text = soup.get_text(separator="\n", strip=True)
            if not extracted_text:
                 logger.warning(f"Tool '{tool_name}': Could not extract meaningful text from {clean_url}")
                 return "Error: Could not extract meaningful text from the page."
            truncated_text = extracted_text[:max_length]
            if len(extracted_text) > max_length: truncated_text += "..."
            logger.info(f"Tool '{tool_name}': Successfully extracted ~{len(truncated_text)} chars from {clean_url}")
            return truncated_text
    except httpx.TimeoutException: logger.error(f"Tool '{tool_name}': Timeout fetching {clean_url}"); return f"Error: Timeout fetching URL."
    except httpx.InvalidURL as e: logger.error(f"Tool '{tool_name}': Invalid URL format for {clean_url}: {e}"); return f"Error: Invalid URL format: {e}"
    except httpx.RequestError as e: logger.error(f"Tool '{tool_name}': Request error fetching {clean_url}: {e}"); return f"Error: Could not fetch URL: {e}"
    except httpx.HTTPStatusError as e: logger.error(f"Tool '{tool_name}': HTTP error fetching {clean_url}: {e.response.status_code}"); return f"Error: HTTP {e.response.status_code} fetching URL."
    except ImportError: logger.error(f"Tool '{tool_name}': lxml not installed."); return "Error: HTML parser (lxml) not installed."
    except Exception as e: logger.error(f"Tool '{tool_name}': Error parsing {clean_url}: {e}", exc_info=True); return f"Error parsing URL: {e}"


async def write_to_file_in_task_workspace(input_str: str, task_workspace: Path) -> str:
    """
    Writes text content to a file within the SPECIFIED task workspace.
    Returns 'SUCCESS::write_file:::<relative_path>' on success.
    """
    tool_name = "write_file"
    logger.info(f"Tool '{tool_name}' received raw input: '{input_str[:100]}...' for workspace {task_workspace.name}")

    # Input Validation
    if not isinstance(input_str, str) or ':::' not in input_str:
        logger.error(f"Tool '{tool_name}' received invalid input format. Expected 'file_path:::text_content'. Got: '{input_str[:100]}...'")
        return "Error: Invalid input format. Expected 'file_path:::text_content'."

    relative_path_str = ""
    try:
        parts = input_str.split(':::', 1)
        # Check again after split, although the check above should cover it
        if len(parts) != 2:
             logger.error(f"Tool '{tool_name}': Input split failed unexpectedly. Got: {parts}")
             return "Error: Invalid input format after splitting. Expected 'file_path:::text_content'."

        relative_path_str = parts[0].strip().strip('\'"`')
        raw_text_content = parts[1]

        # Clean path prefixes
        cleaned_relative_path = relative_path_str
        if cleaned_relative_path.startswith((f"workspace/{task_workspace.name}/", f"workspace\\{task_workspace.name}\\" ,f"{task_workspace.name}/", f"{task_workspace.name}\\")):
             cleaned_relative_path = re.sub(r"^[\\/]?(workspace[\\/])?%s[\\/]" % re.escape(task_workspace.name), "", cleaned_relative_path)
             logger.info(f"Tool '{tool_name}': Stripped workspace/task prefix, using relative path: {cleaned_relative_path}")
        elif cleaned_relative_path.startswith(("workspace/", "workspace\\")):
             cleaned_relative_path = re.sub(r"^[\\/]?(workspace[\\/])+", "", cleaned_relative_path)
             logger.info(f"Tool '{tool_name}': Stripped generic 'workspace/' prefix, using: {cleaned_relative_path}")

        if not cleaned_relative_path:
            logger.error(f"Tool '{tool_name}': File path became empty after cleaning.")
            return "Error: File path cannot be empty after cleaning."

        # Decode unicode escapes
        try: text_content = codecs.decode(raw_text_content, 'unicode_escape'); logger.debug(f"Tool '{tool_name}': Decoded unicode escapes.")
        except Exception as decode_err: logger.warning(f"Tool '{tool_name}': Could not decode unicode escapes, using raw content: {decode_err}"); text_content = raw_text_content
        # Remove markdown code fences if present
        text_content = re.sub(r"^```[a-zA-Z]*\s*\n", "", text_content)
        text_content = re.sub(r"\n```$", "", text_content)
        text_content = text_content.strip() # Strip leading/trailing whitespace from content

        # Security Check: Ensure relative path does not try to escape the workspace
        relative_path = Path(cleaned_relative_path)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            logger.error(f"Tool '{tool_name}': Security Error - Invalid file path '{cleaned_relative_path}' attempts traversal.")
            return f"Error: Invalid file path '{cleaned_relative_path}'. Path must be relative and within the workspace."

        full_path = task_workspace.joinpath(relative_path).resolve()

        # Security Check: Ensure the resolved path is truly within the intended workspace
        if not full_path.is_relative_to(task_workspace.resolve()):
             logger.error(f"Tool '{tool_name}': Security Error - Write path resolves outside task workspace! Task: {task_workspace.name}, Resolved: {full_path}")
             return "Error: File path resolves outside the designated task workspace."

        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file asynchronously
        async with aiofiles.open(full_path, mode='w', encoding='utf-8') as f:
            await f.write(text_content)

        logger.info(f"Tool '{tool_name}': Successfully wrote {len(text_content)} bytes to {full_path}")
        return f"SUCCESS::write_file:::{cleaned_relative_path}"

    except Exception as e:
        logger.error(f"Tool '{tool_name}': Error writing file '{relative_path_str}' to workspace {task_workspace.name}: {e}", exc_info=True)
        return f"Error: Failed to write file '{relative_path_str}'. Reason: {type(e).__name__}"


async def read_file_content(relative_path_str: str, task_workspace: Path) -> str:
    """
    Reads content from a file (text or PDF) within the specified task workspace.
    Appends a warning if PDF content exceeds configured length.
    """
    tool_name = "read_file"
    logger.info(f"Tool '{tool_name}' received raw input: '{relative_path_str[:100]}...' in workspace {task_workspace.name}")

    # Input Validation & Cleaning
    if not isinstance(relative_path_str, str) or not relative_path_str.strip():
        logger.error(f"Tool '{tool_name}': Received invalid input. Expected a non-empty relative file path string.")
        return "Error: Invalid input. Expected a non-empty relative file path string."

    # *** MODIFIED: Extract only the first line and clean it ***
    first_line = relative_path_str.splitlines()[0] if relative_path_str else ""
    cleaned_relative_path = first_line.strip().strip('\'"`')
    logger.info(f"Tool '{tool_name}': Cleaned input path to: '{cleaned_relative_path}'")
    # *** END MODIFIED ***

    if not cleaned_relative_path:
        logger.error(f"Tool '{tool_name}': File path became empty after cleaning.")
        return "Error: File path cannot be empty after cleaning."

    # Security Check: Ensure relative path does not try to escape the workspace
    relative_path = Path(cleaned_relative_path)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        logger.error(f"Tool '{tool_name}': Security Error - Invalid read file path '{cleaned_relative_path}' attempts traversal.")
        return f"Error: Invalid file path '{cleaned_relative_path}'. Path must be relative and within the workspace."

    full_path = task_workspace.joinpath(relative_path).resolve()

    # Security Check: Ensure the resolved path is truly within the intended workspace
    if not full_path.is_relative_to(task_workspace.resolve()):
        logger.error(f"Tool '{tool_name}': Security Error - Read path resolves outside task workspace! Task: {task_workspace.name}, Resolved: {full_path}")
        return "Error: File path resolves outside the designated task workspace."

    if not full_path.exists():
        logger.warning(f"Tool '{tool_name}': File not found at {full_path}")
        return f"Error: File not found at path '{cleaned_relative_path}'."
    if not full_path.is_file():
        logger.warning(f"Tool '{tool_name}': Path is not a file: {full_path}")
        return f"Error: Path '{cleaned_relative_path}' is not a file."

    file_extension = full_path.suffix.lower()
    content = ""

    try:
        if file_extension == ".pdf":
            if pypdf is None:
                logger.error(f"Tool '{tool_name}': Attempted to read PDF, but pypdf library is not installed.")
                return "Error: PDF reading library (pypdf) is not installed on the server."

            def read_pdf_sync():
                extracted_text = ""
                try:
                    reader = pypdf.PdfReader(str(full_path))
                    num_pages = len(reader.pages)
                    logger.info(f"Tool '{tool_name}': Reading {num_pages} pages from PDF: {full_path.name}")
                    for i, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text: extracted_text += page_text + "\n"
                        except Exception as page_err:
                            logger.warning(f"Tool '{tool_name}': Error extracting text from page {i+1} of {full_path.name}: {page_err}")
                            extracted_text += f"\n--- Error reading page {i+1} ---\n"
                    return extracted_text.strip()
                except pypdf.errors.PdfReadError as pdf_err:
                    logger.error(f"Tool '{tool_name}': Error reading PDF file {full_path.name}: {pdf_err}")
                    raise RuntimeError(f"Error: Could not read PDF file '{cleaned_relative_path}'. It might be corrupted or encrypted. Error: {pdf_err}") from pdf_err
                except Exception as e:
                    logger.error(f"Tool '{tool_name}': Unexpected error reading PDF {full_path.name}: {e}", exc_info=True)
                    raise RuntimeError(f"Error: An unexpected error occurred while reading PDF '{cleaned_relative_path}'.") from e

            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, read_pdf_sync)
            actual_length = len(content)
            logger.info(f"Tool '{tool_name}': Successfully read {actual_length} chars from PDF '{cleaned_relative_path}'")

            warning_length = settings.tool_pdf_reader_warning_length
            if actual_length > warning_length:
                warning_message = f"\n\n[SYSTEM WARNING: Full PDF content read ({actual_length} chars), which exceeds the warning threshold of {warning_length} chars. This may be too long for the current LLM's context window.]"
                content += warning_message
                logger.warning(f"Tool '{tool_name}': PDF content length ({actual_length}) exceeds warning threshold ({warning_length}). Appending warning.")

        elif file_extension in TEXT_EXTENSIONS:
            async with aiofiles.open(full_path, mode='r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            logger.info(f"Tool '{tool_name}': Successfully read {len(content)} chars from text file '{cleaned_relative_path}'")
        else:
            logger.warning(f"Tool '{tool_name}': Unsupported file extension '{file_extension}' for file '{cleaned_relative_path}'")
            return f"Error: Cannot read file. Unsupported file extension: '{file_extension}'. Supported text: {', '.join(TEXT_EXTENSIONS)}, .pdf"

        return content

    except RuntimeError as rt_err:
        return str(rt_err)
    except Exception as e:
        logger.error(f"Tool '{tool_name}': Error reading file '{cleaned_relative_path}' in workspace {task_workspace.name}: {e}", exc_info=True)
        return f"Error: Failed to read file '{cleaned_relative_path}'. Reason: {type(e).__name__}"


# --- Custom Shell Tool operating in Task Workspace ---
class TaskWorkspaceShellTool(BaseTool):
    name: str = "workspace_shell"
    description: str = (
        f"Use this tool ONLY to execute **non-interactive** shell commands directly within the **current task's dedicated workspace**. "
        f"Useful for running scripts (e.g., 'python my_script.py', 'Rscript analysis.R'), listing files (`ls -l`), checking file details (`wc`, `head`), etc. "
        f"Input MUST be a single, valid, non-interactive shell command string. Do NOT include path prefixes like 'workspace/task_id/'. "
        f"**DO NOT use this for 'pip install' or 'uv venv' or environment modifications.** Use the dedicated 'python_package_installer' tool for installations."
        f"Timeout: {settings.tool_shell_timeout}s. Max output length: {settings.tool_shell_max_output} chars."
    )
    task_workspace: Path
    timeout: int = settings.tool_shell_timeout
    max_output: int = settings.tool_shell_max_output

    def _run(self, command: str) -> str:
        """Synchronous execution wrapper (avoid if possible)."""
        logger.warning("Running TaskWorkspaceShellTool synchronously using _run.")
        try: loop = asyncio.get_running_loop(); result = loop.run_until_complete(self._arun_internal(command))
        except RuntimeError: logger.warning("No running event loop, creating new one for TaskWorkspaceShellTool._run"); result = asyncio.run(self._arun_internal(command))
        return result

    async def _arun(self, command: str) -> str:
        """Asynchronous execution entry point."""
        return await self._arun_internal(command)

    async def _arun_internal(self, command: str) -> str:
        """Internal async helper for running the command in the specific task workspace."""
        tool_name = self.name
        logger.info(f"Tool '{tool_name}' received raw input: '{command}'")

        # Input validation
        if not isinstance(command, str) or not command.strip():
             logger.error(f"Tool '{tool_name}': Received invalid input. Expected a non-empty command string.")
             return "Error: Invalid input. Expected a non-empty command string."

        cwd = str(self.task_workspace.resolve())
        logger.info(f"Tool '{tool_name}' executing command: '{command}' in CWD: {cwd} (Timeout: {self.timeout}s)")
        process = None; stdout_str = ""; stderr_str = ""
        try:
            clean_command = command.strip().strip('`');
            if not clean_command:
                 logger.error(f"Tool '{tool_name}': Command became empty after cleaning.")
                 return "Error: Received empty command after cleaning."
            # Basic security check
            if '&&' in clean_command or '||' in clean_command or ';' in clean_command or '`' in clean_command or '$(' in clean_command:
                 if '|' not in clean_command: logger.warning(f"Tool '{tool_name}': Potentially unsafe shell characters detected in command: {clean_command}")

            process = await asyncio.create_subprocess_shell(clean_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd)
            TIMEOUT_SECONDS = self.timeout
            try: stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                 logger.error(f"Tool '{tool_name}': Timeout executing command: {clean_command}")
                 if process and process.returncode is None:
                     try: process.terminate()
                     except ProcessLookupError: pass
                     await process.wait()
                 return f"Error: Command timed out after {TIMEOUT_SECONDS} seconds."
            stdout_str = stdout.decode(errors='replace').strip(); stderr_str = stderr.decode(errors='replace').strip(); return_code = process.returncode
            result = ""
            if stdout_str: result += f"STDOUT:\n{stdout_str}\n"
            if return_code != 0:
                 logger.warning(f"Tool '{tool_name}' command '{clean_command}' failed. Exit: {return_code}. Stderr: {stderr_str}")
                 result += f"STDERR:\n{stderr_str}\n" if stderr_str else ""
                 result += f"ERROR: Command failed with exit code {return_code}"
            elif stderr_str:
                 logger.info(f"Tool '{tool_name}' command '{clean_command}' succeeded (Exit: {return_code}) but produced STDERR:\n{stderr_str}")
                 result += f"STDERR (Warnings/Info):\n{stderr_str}\n"
            logger.info(f"Tool '{tool_name}' command finished. Exit code: {return_code}. Reporting result length: {len(result)}")
            MAX_OUTPUT_LENGTH = self.max_output
            if len(result) > MAX_OUTPUT_LENGTH: result = result[:MAX_OUTPUT_LENGTH] + f"\n... (output truncated after {MAX_OUTPUT_LENGTH} characters)"
            return result.strip()
        except FileNotFoundError: cmd_part = clean_command.split()[0] if clean_command else "Unknown"; logger.warning(f"Tool '{tool_name}' command not found: {cmd_part}"); return f"Error: Command not found: {cmd_part}"
        except Exception as e: logger.error(f"Tool '{tool_name}': Error executing command '{clean_command}' in task workspace: {e}", exc_info=True); return f"Error executing command: {type(e).__name__}"
        finally:
            if process and process.returncode is None:
                 logger.warning(f"Tool '{tool_name}': Process '{clean_command}' still running in finally block, attempting termination.")
                 try: process.terminate(); await process.wait()
                 except ProcessLookupError: pass
                 except Exception as term_e: logger.error(f"Tool '{tool_name}': Error during final termination attempt: {term_e}")


# --- Python Package Installer Tool Implementation ---
PACKAGE_SPEC_REGEX = re.compile(r"^[a-zA-Z0-9_.-]+(?:\[[a-zA-Z0-9_,-]+\])?(?:[=<>!~]=?\s*[a-zA-Z0-9_.*-]+)?$")
async def install_python_package(package_specifier: str) -> str:
    """Installs a Python package using the system's Python environment with configured timeout."""
    tool_name = "python_package_installer"
    logger.info(f"Tool '{tool_name}' received raw input: '{package_specifier}'")

    # Input Validation
    if not isinstance(package_specifier, str) or not package_specifier.strip():
        logger.error(f"Tool '{tool_name}': Received invalid input. Expected a non-empty package specifier string.")
        return "Error: Invalid input. Expected a non-empty package specifier string."

    timeout = settings.tool_installer_timeout
    cleaned_package_specifier = package_specifier.strip().strip('\'"`')

    if not cleaned_package_specifier:
        logger.error(f"Tool '{tool_name}': Package specifier became empty after cleaning.")
        return "Error: No package specified after cleaning."
    if not PACKAGE_SPEC_REGEX.match(cleaned_package_specifier):
        logger.error(f"Tool '{tool_name}': Invalid package specifier format rejected: '{cleaned_package_specifier}'.")
        return f"Error: Invalid package specifier format: '{cleaned_package_specifier}'."
    if ';' in cleaned_package_specifier or '&' in cleaned_package_specifier or '|' in cleaned_package_specifier or '`' in cleaned_package_specifier or '$(' in cleaned_package_specifier:
        logger.error(f"Tool '{tool_name}': Potential command injection detected in package specifier: '{cleaned_package_specifier}'.")
        return "Error: Invalid characters detected in package specifier."

    logger.info(f"Tool '{tool_name}': Requesting install for package: '{cleaned_package_specifier}' (Timeout: {timeout}s)")

    python_executable = sys.executable
    installer_command_base = [python_executable, "-m", "uv", "pip"]
    try:
        test_process = await asyncio.create_subprocess_exec(python_executable, "-m", "uv", "--version", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await test_process.wait()
        if test_process.returncode == 0: logger.info(f"Tool '{tool_name}': Detected uv, using 'uv pip install'.")
        else: logger.info(f"Tool '{tool_name}': uv check failed or not found, falling back to 'pip install'."); installer_command_base = [python_executable, "-m", "pip"]
    except Exception as e: logger.warning(f"Tool '{tool_name}': Error checking for uv, falling back to pip: {e}"); installer_command_base = [python_executable, "-m", "pip"]

    command = installer_command_base + ["install", cleaned_package_specifier]
    logger.info(f"Tool '{tool_name}': Executing installation command: {' '.join(command)}")
    process = None
    try:
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        TIMEOUT_SECONDS = timeout
        try: stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.error(f"Tool '{tool_name}': Timeout installing package: {cleaned_package_specifier}")
            if process and process.returncode is None:
                 try: process.terminate()
                 except ProcessLookupError: pass
                 await process.wait()
            return f"Error: Package installation timed out after {TIMEOUT_SECONDS} seconds."
        stdout_str = stdout.decode(errors='replace').strip(); stderr_str = stderr.decode(errors='replace').strip(); return_code = process.returncode
        result = f"Installation command executed for '{cleaned_package_specifier}'. Exit Code: {return_code}\n"
        if stdout_str: result += f"--- Installer STDOUT ---\n{stdout_str}\n"
        if stderr_str: result += f"--- Installer STDERR ---\n{stderr_str}\n"
        if return_code == 0:
            logger.info(f"Tool '{tool_name}': Successfully installed package: {cleaned_package_specifier}")
            success_msg = f"Successfully installed {cleaned_package_specifier}."
            if stderr_str: success_msg += f"\nNotes/Warnings:\n{stderr_str[:500]}{'...' if len(stderr_str)>500 else ''}"
            return success_msg
        else:
            logger.error(f"Tool '{tool_name}': Failed to install package: {cleaned_package_specifier}. Exit code: {return_code}. Stderr: {stderr_str}")
            error_details = stderr_str or stdout_str
            return f"Error: Failed to install {cleaned_package_specifier}. Exit code: {return_code}.\nDetails:\n{error_details[:1000]}{'...' if len(error_details)>1000 else ''}"
    except FileNotFoundError: logger.error(f"Tool '{tool_name}': Error installing package: '{installer_command_base[0]}' command not found."); return f"Error: Could not find Python executable '{installer_command_base[0]}'."
    except Exception as e: logger.error(f"Tool '{tool_name}': Error installing package '{cleaned_package_specifier}': {e}", exc_info=True); return f"Error during installation: {type(e).__name__}"
    finally:
         if process and process.returncode is None:
             logger.warning(f"Tool '{tool_name}': Installer process '{' '.join(command)}' still running in finally block, attempting termination.")
             try: process.terminate(); await process.wait()
             except ProcessLookupError: pass
             except Exception as term_e: logger.error(f"Tool '{tool_name}': Error during final termination attempt of installer: {term_e}")


# --- PubMed Search Tool Implementation ---
async def search_pubmed(query: str) -> str:
    """Searches PubMed for biomedical literature using configured settings."""
    tool_name = "pubmed_search"
    logger.info(f"Tool '{tool_name}' received raw input: '{query}'")

    # Input Validation
    if not isinstance(query, str) or not query.strip():
        logger.error(f"Tool '{tool_name}': Received invalid input. Expected a non-empty search query string.")
        return "Error: Invalid input. Expected a non-empty search query string."

    entrez_email = settings.entrez_email
    default_max_results = settings.tool_pubmed_default_max_results
    max_snippet = settings.tool_pubmed_max_snippet

    if not entrez_email:
        logger.error(f"Tool '{tool_name}': Entrez email not configured in settings.")
        return "Error: PubMed Search tool is not configured (Missing Entrez email)."

    Entrez.email = entrez_email

    cleaned_query = query.strip()
    logger.info(f"Tool '{tool_name}': Searching PubMed with query: '{cleaned_query}' (Default Max: {default_max_results})")
    current_max_results = default_max_results
    # Use regex that requires space before max_results for less ambiguity
    match = re.search(r"\smax_results=(\d+)$", cleaned_query)
    if match:
        try:
            num_res = int(match.group(1))
            # Clamp max_results between 1 and 20 (reasonable limits)
            current_max_results = min(max(1, num_res), 20)
            # Remove the max_results part from the query
            cleaned_query = cleaned_query[:match.start()].strip()
            logger.info(f"Tool '{tool_name}': Using max_results={current_max_results} from query.")
        except ValueError:
            logger.warning(f"Tool '{tool_name}': Invalid max_results value in query '{query}', using default {current_max_results}.")
    if not cleaned_query:
        logger.error(f"Tool '{tool_name}': Query became empty after processing max_results.")
        return "Error: No search query provided after processing options."

    try:
        handle = await asyncio.to_thread(Entrez.esearch, db="pubmed", term=cleaned_query, retmax=str(current_max_results), sort="relevance")
        search_results = await asyncio.to_thread(Entrez.read, handle); await asyncio.to_thread(handle.close)
        id_list = search_results["IdList"]
        if not id_list:
            logger.info(f"Tool '{tool_name}': No results found on PubMed for query: '{cleaned_query}'")
            return f"No results found on PubMed for query: '{cleaned_query}'"

        handle = await asyncio.to_thread(Entrez.efetch, db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = await asyncio.to_thread(Entrez.read, handle); await asyncio.to_thread(handle.close)
        summaries = []; pubmed_articles = records.get('PubmedArticle', [])
        if not isinstance(pubmed_articles, list):
            logger.warning(f"Tool '{tool_name}': Unexpected PubMed fetch format for query '{cleaned_query}'. Records: {records}")
            if isinstance(pubmed_articles, dict): pubmed_articles = [pubmed_articles] # Handle single result case
            else: return "Error: Could not parse PubMed results (unexpected format)."

        for i, record in enumerate(pubmed_articles):
            if i >= current_max_results: break # Should be redundant due to retmax, but safety check
            pmid = "Unknown PMID"
            try:
                medline_citation = record.get('MedlineCitation', {}); article = medline_citation.get('Article', {}); pmid = str(medline_citation.get('PMID', 'Unknown PMID'))
                title = article.get('ArticleTitle', 'No Title');
                if not isinstance(title, str): title = str(title)
                authors_list = article.get('AuthorList', []); author_names = []
                if isinstance(authors_list, list):
                    for author in authors_list:
                         if isinstance(author, dict):
                             last_name = author.get('LastName', ''); initials = author.get('Initials', '')
                             if last_name: author_names.append(f"{last_name} {initials}".strip())
                authors = ", ".join(author_names) or "No Authors Listed"
                abstract_text = ""; abstract_section = article.get('Abstract', {}).get('AbstractText', [])
                if isinstance(abstract_section, list):
                    section_texts = []
                    for sec in abstract_section:
                        if isinstance(sec, str): section_texts.append(sec)
                        elif isinstance(sec, dict): section_texts.append(sec.get('#text', ''))
                        elif hasattr(sec, 'attributes') and 'Label' in sec.attributes: section_texts.append(f"\n**{sec.attributes['Label']}**: {str(sec)}")
                        else: section_texts.append(str(sec))
                    abstract_text = " ".join(filter(None, section_texts))
                elif isinstance(abstract_section, str): abstract_text = abstract_section
                else: abstract_text = str(abstract_section) if abstract_section else "No Abstract Available"
                MAX_ABSTRACT_SNIPPET = max_snippet
                abstract_snippet = abstract_text.strip()[:MAX_ABSTRACT_SNIPPET]
                if len(abstract_text.strip()) > MAX_ABSTRACT_SNIPPET: abstract_snippet += "..."
                if not abstract_snippet: abstract_snippet = "No Abstract Available"
                doi = None; article_ids = record.get('PubmedData', {}).get('ArticleIdList', [])
                if isinstance(article_ids, list):
                    for article_id in article_ids:
                        if hasattr(article_id, 'attributes') and article_id.attributes.get('IdType') == 'doi': doi = str(article_id); break
                        elif isinstance(article_id, dict) and article_id.get('IdType') == 'doi': doi = article_id.get('#text'); break
                link = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"; link_text = f"DOI:{doi}" if doi else f"PMID:{pmid}"
                summaries.append(f"**Result {i+1}:**\n**Title:** {title}\n**Authors:** {authors}\n**Link:** [{link_text}]({link})\n**Abstract Snippet:** {abstract_snippet}\n---")
            except Exception as parse_err: logger.error(f"Tool '{tool_name}': Error parsing PubMed record {i+1} (PMID: {pmid}) for query '{cleaned_query}': {parse_err}", exc_info=True); summaries.append(f"**Result {i+1}:**\nError parsing record (PMID: {pmid}).\n---")
        return "\n".join(summaries) if summaries else "No valid PubMed records processed."
    except HTTPError as e: logger.error(f"Tool '{tool_name}': HTTP Error fetching PubMed data for query '{cleaned_query}': {e.code} {e.reason}"); return f"Error: Failed to fetch data from PubMed (HTTP Error {e.code}). Check network or NCBI status."
    except Exception as e: logger.error(f"Tool '{tool_name}': Error searching PubMed for '{cleaned_query}': {e}", exc_info=True); return f"Error: An unexpected error occurred during PubMed search: {type(e).__name__}"


# Create PythonREPL utility instance (conditionally)
try: python_repl_utility = PythonREPL()
except ImportError: logger.warning("Could not import PythonREPL. The Python_REPL tool will not be available."); python_repl_utility = None

# --- Tool Factory Function ---
def get_dynamic_tools(current_task_id: Optional[str]) -> List[BaseTool]:
    """Creates tool instances dynamically, configured for the current task's workspace and settings."""

    stateless_tools = [
        DuckDuckGoSearchRun(description=("A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events or things you don't know. Input MUST be a search query string.")),
        Tool.from_function(
            func=fetch_and_parse_url,
            name="web_page_reader",
            description=(f"Use this tool ONLY to fetch and extract the main text content from a given URL. Input MUST be a single, valid URL string (e.g., 'https://example.com/page'). Max content length: {settings.tool_web_reader_max_length} chars."),
            coroutine=fetch_and_parse_url
        ),
        Tool.from_function(
            func=install_python_package,
            name="python_package_installer",
            description=(f"Use this tool ONLY to install a Python package into the environment using 'uv pip install' or 'pip install'. Input MUST be a single, valid package specifier string (e.g., 'numpy', 'pandas==2.0.0', 'matplotlib>=3.5'). **SECURITY WARNING:** This installs packages into the main environment. Avoid installing untrusted packages. Timeout: {settings.tool_installer_timeout}s."),
            coroutine=install_python_package
        ),
    ]

    if settings.entrez_email:
        stateless_tools.append(
             Tool.from_function(
                 func=search_pubmed,
                 name="pubmed_search",
                 description=(f"Use this tool ONLY to search for biomedical literature abstracts on PubMed. Input MUST be a search query string (e.g., 'CRISPR gene editing cancer therapy'). You can optionally append ' max_results=N' (space required before 'max_results') to the end of the query string to specify the number of results (default is {settings.tool_pubmed_default_max_results}, max is 20). Returns formatted summaries including title, authors, link (DOI or PMID), and abstract snippet (max {settings.tool_pubmed_max_snippet} chars)."),
                 coroutine=search_pubmed
             )
        )
    else:
        logger.warning("Skipping PubMed tool creation as ENTREZ_EMAIL is not set.")

    if python_repl_utility:
        stateless_tools.append(Tool.from_function(
            func=python_repl_utility.run,
            name="Python_REPL",
            description=("Use this tool to execute Python code snippets. Input MUST be valid Python code string. Output will be the result of the execution (stdout, stderr, or return value). **Security Note:** This executes code directly in the backend environment. Be extremely cautious.")))
    else:
        logger.warning("Python REPL tool not created (utility unavailable).")

    # --- Task-Specific Tools ---
    if not current_task_id:
        logger.warning("No active task ID provided to get_dynamic_tools, returning only stateless tools.")
        return stateless_tools

    try:
        task_workspace = get_task_workspace_path(current_task_id)
        logger.info(f"Configuring file/shell tools for workspace: {task_workspace}")
    except (ValueError, OSError) as e:
        logger.error(f"Failed to get or create task workspace for {current_task_id}: {e}. Returning only stateless tools.")
        return stateless_tools

    task_specific_tools = [
        TaskWorkspaceShellTool(task_workspace=task_workspace),
        Tool.from_function(
            func=lambda path_str: read_file_content(path_str, task_workspace),
            name="read_file",
            description=(f"Use this tool ONLY to read the entire contents of a file (including text and PDF files) located within the current task's workspace ('{task_workspace.name}'). Input MUST be the relative path string to the file from the workspace root (e.g., 'my_data.csv', 'report.pdf', 'scripts/analysis.py'). Returns the full text content or an error message. For PDFs, a warning is appended if the content exceeds {settings.tool_pdf_reader_warning_length} characters."),
            coroutine=lambda path_str: read_file_content(path_str, task_workspace)
        ),
        Tool.from_function(
            func=lambda input_str: write_to_file_in_task_workspace(input_str, task_workspace),
            name="write_file",
            description=(f"Use this tool ONLY to write or overwrite text content to a file within the current task's workspace ('{task_workspace.name}'). Input MUST be a single string in the format 'relative_file_path:::text_content' (e.g., 'results.txt:::Analysis complete.\\nFinal score: 95'). Handles subdirectory creation. Do NOT use workspace path prefix in 'relative_file_path'."),
            coroutine=lambda input_str: write_to_file_in_task_workspace(input_str, task_workspace)
        )
    ]

    all_tools = stateless_tools + task_specific_tools
    logger.info(f"Returning tools for task {current_task_id}: {[tool.name for tool in all_tools]}")
    return all_tools

