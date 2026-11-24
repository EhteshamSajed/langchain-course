from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    if os.environ.get("AZURE_OPENAI_DEPLOYMENT") is None:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT environment variable is not set.")
    return AzureChatOpenAI(
        temperature=0,
        api_version="2024-12-01-preview",
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default")
    )
