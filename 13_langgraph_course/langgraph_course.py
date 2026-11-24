import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

from nodes import tool_node, run_agent_reasoning

load_dotenv()

AGENT_REASON = "agent_reason"
ACT="act"
LAST=-1

def should_continue(state: MessagesState) -> str:
    """
    Determine if the agent should continue reasoning or end the flow.

    :param state: The current state of the message.
    :return: The next state, either END or ACT.
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

if os.environ.get("AZURE_OPENAI_DEPLOYMENT") is None:
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("ReAct LangGraph with Function calling")
    res = app.invoke({
        "messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it.")]
    })
    print(res["messages"][LAST].content)
