# backend/config.py
import os
import logging
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path  # <<< FIX: Added this import

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

def parse_comma_separated_list(env_var: Optional[str], default: Optional[List[str]] = None) -> List[str]:
    """Parses a comma-separated string from env var into a list, cleaning prefixes."""
    if env_var:
        # Split by comma and clean up each item
        return [item.strip().split("::", 1)[-1] for item in env_var.split(',') if item.strip()]
    return default if default is not None else []

@dataclass
class Settings:
    """Holds application configuration settings loaded from environment variables."""

    # --- Required API Keys ---
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY'))
    entrez_email: Optional[str] = field(default_factory=lambda: os.getenv('ENTREZ_EMAIL'))
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv('TAVILY_API_KEY'))

    # --- Core LLM Configuration (Simplified) ---
    default_llm_id: str = field(default_factory=lambda: os.getenv('DEFAULT_LLM_ID', 'gemini::gemini-1.5-flash'))
    
    ollama_base_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    _gemini_available_models_str: Optional[str] = field(default_factory=lambda: os.getenv('GEMINI_AVAILABLE_MODELS'))
    _ollama_available_models_str: Optional[str] = field(default_factory=lambda: os.getenv('OLLAMA_AVAILABLE_MODELS'))
    
    gemini_available_models: List[str] = field(default_factory=list, init=False)
    ollama_available_models: List[str] = field(default_factory=list, init=False)

    # --- Agent & LLM Tuning ---
    agent_max_iterations: int = field(default_factory=lambda: int(os.getenv('AGENT_MAX_ITERATIONS', '15')))
    agent_memory_window_k: int = field(default_factory=lambda: int(os.getenv('AGENT_MEMORY_WINDOW_K', '10')))
    gemini_temperature: float = field(default_factory=lambda: float(os.getenv('GEMINI_TEMPERATURE', '0.7')))
    ollama_temperature: float = field(default_factory=lambda: float(os.getenv('OLLAMA_TEMPERATURE', '0.5')))

    # --- Tool Settings (Unchanged) ---
    tool_web_reader_max_length: int = field(default_factory=lambda: int(os.getenv('TOOL_WEB_READER_MAX_LENGTH', '4000')))
    tool_web_reader_timeout: float = field(default_factory=lambda: float(os.getenv('TOOL_WEB_READER_TIMEOUT', '15.0')))
    tool_shell_timeout: int = field(default_factory=lambda: int(os.getenv('TOOL_SHELL_TIMEOUT', '60')))
    tool_shell_max_output: int = field(default_factory=lambda: int(os.getenv('TOOL_SHELL_MAX_OUTPUT', '3000')))
    tool_installer_timeout: int = field(default_factory=lambda: int(os.getenv('TOOL_INSTALLER_TIMEOUT', '300')))
    tool_pubmed_default_max_results: int = field(default_factory=lambda: int(os.getenv('TOOL_PUBMED_DEFAULT_MAX_RESULTS', '5')))
    tool_pubmed_max_snippet: int = field(default_factory=lambda: int(os.getenv('TOOL_PUBMED_MAX_SNIPPET', '250')))
    tool_pdf_reader_warning_length: int = field(default_factory=lambda: int(os.getenv('TOOL_PDF_READER_WARNING_LENGTH', '20000')))

    # --- Server Settings (Unchanged) ---
    websocket_max_size_bytes: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_MAX_SIZE_BYTES', '16777216')))
    websocket_ping_interval: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_PING_INTERVAL', '20')))
    websocket_ping_timeout: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_PING_TIMEOUT', '30')))
    file_server_hostname: str = field(default_factory=lambda: os.getenv('FILE_SERVER_HOSTNAME', 'localhost'))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO').upper())

    # --- Parsed/Derived values (Simplified) ---
    default_provider: str = field(init=False)
    default_model_name: str = field(init=False)

    def __post_init__(self):
        """Validate settings and parse derived values after initialization."""
        logging.getLogger().setLevel(self.log_level)

        self.gemini_available_models = parse_comma_separated_list(self._gemini_available_models_str, default=['gemini-1.5-flash'])
        self.ollama_available_models = parse_comma_separated_list(self._ollama_available_models_str, default=['llama3:latest'])

        try:
            self.default_provider, self.default_model_name = self.default_llm_id.split("::", 1)
        except ValueError:
            logger.critical(f"Invalid DEFAULT_LLM_ID format: '{self.default_llm_id}'. Must be 'provider::model_name'.")
            raise

        logger.info("--- Simplified Configuration Summary ---")
        logger.info(f"Default LLM: {self.default_provider}::{self.default_model_name}")
        logger.info(f"Ollama Base URL: {self.ollama_base_url}")
        logger.info(f"Available Gemini Models: {self.gemini_available_models}")
        logger.info(f"Available Ollama Models: {self.ollama_available_models}")
        logger.info(f"Agent Max Iterations: {self.agent_max_iterations}")
        logger.info(f"Log Level: {self.log_level}")
        logger.info("---------------------------------------")

def load_settings() -> Settings:
    """Loads settings from .env file and environment variables."""
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        logger.info(f"Loading environment variables from: {env_path}")
        load_dotenv(dotenv_path=env_path)
    return Settings()

settings = load_settings()
