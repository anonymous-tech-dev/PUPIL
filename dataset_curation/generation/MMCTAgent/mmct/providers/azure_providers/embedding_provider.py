from mmct.providers.base import EmbeddingProvider
from typing import Dict, Any, List
from azure.identity import get_bearer_token_provider
from loguru import logger
from mmct.utils.error_handler import ProviderException, ConfigurationException
from openai import AsyncAzureOpenAI
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.providers.credentials import AzureCredentials

# --- Azure CONSTANTS (Hardcoded for your setup) ---
Azure_ENDPOINT = 'https://<AZURE_OPENAI_ENDPOINT>'
Azure_API_VERSION = '2024-10-21' 
Azure_SCOPE = "api://azure/.default"
# Matches the deployment name you confirmed:
Azure_EMBEDDING_DEPLOYMENT = "text-embedding-3-large_1" 
# --------------------------------------------------

class AzureEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI embedding provider implementation configured for Azure."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_credentials()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client with Azure configuration."""
        try:
            # Hardcoded Azure settings override whatever is in config/.env
            endpoint = Azure_ENDPOINT
            api_version = Azure_API_VERSION
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            token_provider = get_bearer_token_provider(
                self.credential, 
                Azure_SCOPE
            )
            
            logger.info(f"Initializing Azure Embedding Client: {Azure_EMBEDDING_DEPLOYMENT} @ {endpoint}")

            return AsyncAzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                max_retries=max_retries,
                timeout=timeout
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure embedding client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using Azure OpenAI (Azure)."""
        try:
            # Force the model name
            deployment_name = Azure_EMBEDDING_DEPLOYMENT
            
            response = await self.client.embeddings.create(
                model=deployment_name,
                input=text,
                **kwargs
            )
            
            if not response.data or not response.data[0].embedding:
                raise ProviderException("Azure returned empty embedding data")

            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure embedding failed: {e}")
            raise ProviderException(f"Azure embedding failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts using Azure OpenAI (Azure)."""
        try:
            deployment_name = Azure_EMBEDDING_DEPLOYMENT
            
            response = await self.client.embeddings.create(
                model=deployment_name,
                input=texts,
                **kwargs
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Azure batch embedding failed: {e}")
            raise ProviderException(f"Azure batch embedding failed: {e}")

    def get_async_client(self):
        """Get async OpenAI client for direct embeddings API access."""
        return self.client

    async def close(self):
        """Close the embedding client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure embedding client")
            await self.client.close()