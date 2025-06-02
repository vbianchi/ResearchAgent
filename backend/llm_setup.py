# backend/llm_setup.py
import logging
from typing import Optional, List, Union # MODIFIED: Added Union
from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager # MODIFIED: Added BaseCallbackManager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM 
from langchain_core.language_models.base import BaseLanguageModel
from backend.config import Settings 

logger = logging.getLogger(__name__)

def get_llm(
    settings: Settings, 
    provider: str, 
    model_name: str, 
    is_fallback_attempt: bool = False, 
    requested_for_role: Optional[str] = None,
    # MODIFIED: Changed type hint for callbacks
    callbacks: Union[List[BaseCallbackHandler], BaseCallbackManager, None] = None 
) -> BaseLanguageModel:
    """
    Initializes and returns the appropriate LangChain LLM wrapper.
    Accepts a list of BaseCallbackHandler instances or a BaseCallbackManager instance.
    """
    role_context = f" for role '{requested_for_role}'" if requested_for_role else ""
    attempt_type = " (Fallback to system default)" if is_fallback_attempt else ""
    logger.info(f"Attempting to initialize LLM{role_context}: Provider='{provider}', Model='{model_name}'{attempt_type}")
    
    if callbacks:
        logger.debug(f"get_llm (Role: {requested_for_role}) received callbacks of type: {type(callbacks)}")
    else:
        logger.debug(f"get_llm (Role: {requested_for_role}) received no callbacks.")

    try:
        if provider == "gemini":
            if not settings.google_api_key:
                logger.error(f"Cannot initialize Gemini LLM{role_context}: GOOGLE_API_KEY is not set.")
                raise ValueError(f"GOOGLE_API_KEY is required for Gemini provider{role_context}.")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.google_api_key,
                temperature=settings.gemini_temperature,
                callbacks=callbacks # MODIFIED: Pass callbacks directly
            )
            logger.info(f"Successfully initialized ChatGoogleGenerativeAI{role_context}: Model='{model_name}', Temp='{settings.gemini_temperature}'")
            return llm

        elif provider == "ollama":
            # Note: OllamaLLM from langchain_ollama might have a slightly different signature or expectation
            # for callbacks compared to ChatGoogleGenerativeAI. Let's assume it also accepts Callbacks type.
            # If it specifically needs a list, this might need further adjustment for Ollama.
            llm = OllamaLLM( 
                base_url=settings.ollama_base_url,
                model=model_name,
                temperature=settings.ollama_temperature,
                callbacks=callbacks # MODIFIED: Pass callbacks directly
            )
            logger.info(f"Successfully initialized OllamaLLM{role_context}: Model='{model_name}', URL='{settings.ollama_base_url}', Temp='{settings.ollama_temperature}'")
            return llm

        else:
            logger.error(f"Unsupported LLM provider requested{role_context}: {provider}")
            raise ValueError(f"Unsupported LLM provider requested{role_context}: {provider}")

    except Exception as e:
        logger.warning(f"Failed to initialize LLM '{provider}::{model_name}'{role_context}: {e}", exc_info=True) # Added exc_info
        if not is_fallback_attempt:
            fallback_role_context = f" (originally for role '{requested_for_role}')" if requested_for_role else ""
            logger.warning(f"Attempting fallback to system default LLM: {settings.default_provider}::{settings.default_model_name}{fallback_role_context}")
            try:
                # Fallback should not inherit component-specific callbacks if the primary failed due to them.
                # Or, if the issue is systemic, it might also fail. For now, pass None.
                return get_llm(settings, settings.default_provider, settings.default_model_name, is_fallback_attempt=True, requested_for_role=f"System Default Fallback (was for {requested_for_role or 'Unknown Role'})", callbacks=None)
            except Exception as fallback_e:
                logger.error(f"Fallback LLM initialization failed{fallback_role_context}: {fallback_e}", exc_info=True)
                raise ConnectionError(f"Failed to initialize both primary LLM ('{provider}::{model_name}'{role_context}) and fallback LLM ('{settings.default_provider}::{settings.default_model_name}'). Original error: {e}. Fallback error: {fallback_e}") from fallback_e
        else:
            logger.error(f"Fallback LLM initialization for '{provider}::{model_name}'{role_context} also failed: {e}", exc_info=True)
            raise ConnectionError(f"Fallback LLM initialization failed for '{provider}::{model_name}'{role_context}: {e}") from e
