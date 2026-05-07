import os
import nest_asyncio
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from mmct.video_pipeline import VideoAgent
from config import MMCT_ENDPOINT

nest_asyncio.apply()

class ModelManager:
    def __init__(self):
        self.credential = AzureCliCredential()
        self.token_provider = get_bearer_token_provider(self.credential, "api://azure/.default")
        
        self.gpt_client = AzureOpenAI(
            azure_endpoint=MMCT_ENDPOINT,
            azure_ad_token_provider=self.token_provider,
            api_version="2024-10-21"
        )

    def get_gpt_completion(self, messages, model, temperature=0.0, json_mode=False):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.gpt_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ GPT Error: {e}")
            return None

    def get_video_agent(self, index_name, query):
        return VideoAgent(
            query=query,
            index_name=index_name,
            use_critic_agent=False,
            stream=False
        )