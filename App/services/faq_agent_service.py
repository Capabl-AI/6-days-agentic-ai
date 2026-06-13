from App.config import settings

from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from App.agents.faq_agent import FAQAgent
from App.services.short_term_memory_manager import ShortTermMemoryManager
from App.services.long_term_memory_manager import LongTermMemoryManager

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

    def run_faq_workflow(self, query: str, session_id: str) -> str:
        long_term_memory = LongTermMemoryManager()
        lt_summary = long_term_memory.get_summary(session_id)
        
        short_term_memory = ShortTermMemoryManager()
        history_db = short_term_memory.get_chat_history()
        chat_history = history_db.messages

        messages = []
        if lt_summary:
            messages.append(SystemMessage(content=f"Summary of previous conversations: {lt_summary}"))
        
        messages.extend(chat_history)
        messages.append(HumanMessage(content=query))

        result = self.faq_agent.invoke({"messages": messages})
        response_msg = result["messages"][-1].content

        history_db.add_user_message(query)
        history_db.add_ai_message(response_msg)

        if len(history_db.messages) !=0 and len(history_db.messages) % 10 == 0:
            new_summary = self._generate_summary(history_db.messages, lt_summary)
            long_term_memory.save_summary(session_id, new_summary)

            history_db.clear()
            history_db.add_message(SystemMessage(content=f"Conversation summarized. New summary saved."))
        return result