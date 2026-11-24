from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from tenacity import retry_unless_exception_type

from react import llm, tools

load_dotenv()


SYSTEM_MESSAGE = """You are a helpful assistant that can answer questions and provide information based on the provided tools.
You have access to the following tools:
{tools}
When you receive a question, you should first determine if any of the tools can help answer the question. If a tool can help, you should use it to get the information needed to answer the question. If no tools are needed, you should answer the question directly.
When using a tool, you should provide the tool with the necessary input to get the information you need. Once you have the information from the tool, you should use it to answer the question.
"""

def run_agent_reasoning(state: MessagesState)-> MessagesState:
    """
    Run the agent node

    :param state: The current state of messages.
    :return: The updated state after reasoning.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": response}

tool_node = ToolNode(tools=tools)
