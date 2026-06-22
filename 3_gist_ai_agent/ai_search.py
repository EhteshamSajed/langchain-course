# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tavily import TavilyClient
# from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """Schema for source used by the agent."""

    url: str = Field(description="Source URL")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Agents answer to the query")
    sources: List[Source] = Field(default_factory= list, description="List of source URLs to generate answers")


@tool
def search(query: str) -> str:
    """
    Search using Tavily API over internet

    :param query: search query
    :return: search results
    """

    tavily_client = TavilyClient()
    tavily_client.session.verify = False  # BASF proxy cert missing AKI extension (non-compliant with Python 3.14+ OpenSSL)

    print("Searching ...")
    return tavily_client.search(query=query)


def ai_search():
    llm = AzureChatOpenAI(
        temperature=0,
        api_version="2024-12-01-preview",
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default")
    )
    tools = [search]
    # tools=[TavilySearch()] # ssl error
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    # result = agent.invoke({"messages": HumanMessage(content="What is the current weather in Berlin?")})
    result = agent.invoke({"messages": HumanMessage(
        content="Search 3 jobs in linkdin in Berlin related to Agentic "
                "AI, Langchain and summerize the reqirements for all jobs.")})
    print(result['messages'][3].content)

if __name__ == "__main__":
    ai_search()