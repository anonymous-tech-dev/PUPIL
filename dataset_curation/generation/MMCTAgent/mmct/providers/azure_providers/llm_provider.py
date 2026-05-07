from mmct.providers.base import LLMProvider
from loguru import logger
from openai import AsyncAzureOpenAI
from azure.identity import get_bearer_token_provider
from mmct.utils.error_handler import ProviderException, ConfigurationException
from typing import Dict, Any, List
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.providers.credentials import AzureCredentials
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# --- Azure CONSTANTS ---
Azure_ENDPOINTS = [
    'https://<AZURE_OPENAI_ENDPOINT>',
    'https://<AZURE_OPENAI_ENDPOINT>'
]
Azure_API_VERSION = '2024-10-21'
Azure_SCOPE = "api://azure/.default"
Azure_MODEL_NAME = "gpt-5.1_2025-11-13" # Downgraded to 5.1 for stability/limits

# Capability info required by AutoGen for custom models
Azure_MODEL_INFO = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": "gpt-5", # Updated family
}
# -----------------------

class AzureLLMProvider(LLMProvider):
    """Azure OpenAI LLM provider implementation configured for Azure with Endpoint Fallback."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_credentials()
        # Initialize a client for every endpoint in the fallback chain
        self.clients = [self._initialize_client(ep) for ep in Azure_ENDPOINTS]
    
    def _initialize_client(self, endpoint: str):
        """Initialize a single Azure OpenAI client for a specific endpoint."""
        try:
            api_version = Azure_API_VERSION
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            token_provider = get_bearer_token_provider(
                self.credential, 
                Azure_SCOPE
            )
            
            logger.info(f"Initializing Azure Client: {Azure_MODEL_NAME} @ {endpoint} ({api_version})")

            return AsyncAzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                max_retries=max_retries,
                timeout=timeout
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure client for {endpoint}: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        deployment_name = Azure_MODEL_NAME
        max_tokens = kwargs.get("max_tokens", 15000)
        response_format = kwargs.get("response_format")
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "response_format"]}
        
        last_exception = None

        # Fallback Loop: Iterate through the initialized clients
        for attempt_idx, client in enumerate(self.clients):
            endpoint_used = Azure_ENDPOINTS[attempt_idx]
            try:
                from pydantic import BaseModel
                if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    response = await client.chat.completions.parse(
                        model=deployment_name,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        reasoning_effort="low",
                        response_format=response_format,
                        **filtered_kwargs
                    )
                    
                    return {
                        "content": response.choices[0].message.parsed,
                        "usage": response.usage.model_dump() if response.usage else None,
                        "model": response.model,
                        "finish_reason": response.choices[0].finish_reason
                    }
                else:
                    completion_kwargs = {
                        "model": deployment_name,
                        "messages": messages,
                        "max_completion_tokens": max_tokens,
                        "reasoning_effort": "low",
                        **filtered_kwargs
                    }

                    if response_format:
                        completion_kwargs["response_format"] = response_format
                    
                    response = await client.chat.completions.create(**completion_kwargs)
                    
                    return {
                        "content": response.choices[0].message.content,
                        "usage": response.usage.model_dump() if response.usage else None,
                        "model": response.model,
                        "finish_reason": response.choices[0].finish_reason
                    }

            except Exception as e:
                last_exception = e
                # Check if we have more endpoints to try
                if attempt_idx < len(self.clients) - 1:
                    logger.warning(f"Azure request hit rate limit/error on {endpoint_used}: {e}. Falling back to next endpoint...")
                else:
                    logger.error(f"Azure request failed on final fallback endpoint {endpoint_used}: {e}")
        
        # If the loop completes without returning, all endpoints failed
        raise ProviderException(f"Azure OpenAI chat completion failed on all Azure endpoints. Last error: {last_exception}")

    def get_autogen_client(self) -> List[AzureOpenAIChatCompletionClient]:
        """Get autogen-compatible clients. Returns a list for AutoGen's native round-robin fallback."""
        autogen_clients = []
        api_version = Azure_API_VERSION
        deployment_name = Azure_MODEL_NAME
        timeout = self.config.get("timeout", 200)
        temperature = self.config.get("temperature", 0)

        token_provider = get_bearer_token_provider(
            self.credential, 
            Azure_SCOPE
        )

        for endpoint in Azure_ENDPOINTS:
            try:
                logger.info(f"Initializing Azure AutoGen Client: {deployment_name} @ {endpoint}")
                client = AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=timeout,
                    temperature=temperature,
                    model_info=Azure_MODEL_INFO
                )
                autogen_clients.append(client)
            except Exception as e:
                logger.error(f"Failed to create Azure autogen client for {endpoint}: {e}")
        
        if not autogen_clients:
            raise ProviderException("Failed to initialize any AutoGen fallback clients.")
        
        # Note: We return a list so AutoGen can use them as a fallback config_list.
        # If your internal `mmct` framework strictly expects a single client object, 
        # change this return statement to: return autogen_clients[0]
        return autogen_clients[0]

    async def close(self):
        """Close all LLM clients and cleanup resources."""
        for idx, client in enumerate(self.clients):
            if client:
                logger.info(f"Closing Azure OpenAI (Azure) LLM client for {Azure_ENDPOINTS[idx]}")
                await client.close()