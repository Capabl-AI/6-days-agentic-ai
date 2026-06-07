from App.config import settings

from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, HumanMessage

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from App.agents.faq_agent import FAQAgent

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return END

class FAQWorkflow:
    def __init__(self):
        # Build WOrkflow
        agent_builder = StateGraph(MessagesState)

        # Add nodes
        agent_builder.add_node("faq_agent", FAQAgent().llm_call)
        agent_builder.add_node("tool_node", FAQAgent().tool_node)

        # Add edges
        agent_builder.add_edge(START, "faq_agent")
        agent_builder.add_conditional_edges(
            "faq_agent",
            should_continue,
            ["tool_node", END]
        )
        agent_builder.add_edge("tool_node", "faq_agent")

        # Compile the agent
        self.faq_agent = agent_builder.compile()

    def run_faq_workflow(self, query: str) -> str:
        result = self.faq_agent.invoke({"messages": [HumanMessage(content=query)]})
        return result