from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_tavily import TavilySearch
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import os
import requests

import agent_selector

load_dotenv()

class Source(BaseModel):
    """Schema for source used by the agent."""

    url: str = Field(description="Source URL")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Agents answer to the query")
    sources: List[Source] = Field(default_factory= list, description="List of source URLs to generate answers")


@tool
def triple(num: float) -> float:
    """
    Triple the input number.

    :param num: Input number
    :return: Tripled value
    """
    print(f"Tripling {num} ...")
    return num * 3

@tool
def get_weather(location: str) -> str:
    """
    Get the weather for a given location. Use [GET]https://wttr.in/{location}?format=j1
    returns the json response. You can use the json to get the weather information.

    :param location: The location query
    :return: The weather information as a string
    """
    print(f"Getting weather for {location} ...")
    if os.environ.get("AZURE_OPENAI_DEPLOYMENT") is None:
        # On personal machine, we can use the wttr.in API to get the weather information.
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current_condition = data['current_condition'][0]
            weather_desc = current_condition['weatherDesc'][0]['value']
            temp_c = current_condition['temp_C']
            return f"The current weather in {location} is {weather_desc} with a temperature of {temp_c}°C."
        else:
            return f"Could not retrieve weather information for {location}. Please try again later."

    # Corporate environment: blocks external API calls. Mock it.
    return f"The current weather in {location} is Sunny with a temperature of 25°C."

if os.environ.get("AZURE_OPENAI_DEPLOYMENT") is None:
    tools = [TavilySearch(max_results=1), triple, get_weather]
else:
    tools = [triple, get_weather]

llm = agent_selector.get_llm().bind_tools(tools)
