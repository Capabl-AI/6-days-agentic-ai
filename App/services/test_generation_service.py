from App.config import settings

from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, HumanMessage

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from App.agents.subject_expert_agent import SubjectExpertAgent
from App.agents.question_generation_agent import QuestionGenerationAgent

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "use_tool"

    return "__end__"

class TestGenerationService:
    def __init__(self):
        # Build WOrkflow
        agent_builder = StateGraph(MessagesState)

        # Add nodes
        agent_builder.add_node("subject_expert_agent", SubjectExpertAgent().llm_call)
        agent_builder.add_node("subject_expert_tool_node", SubjectExpertAgent().tool_node)
        agent_builder.add_node("question_generation_agent", QuestionGenerationAgent().llm_call)
        agent_builder.add_node("question_generation_tool_node", QuestionGenerationAgent().tool_node)

        # Add edges
        agent_builder.add_edge(START, "subject_expert_agent")
        agent_builder.add_conditional_edges(
            "subject_expert_agent",
            should_continue,
            {
                "use_tool": "subject_expert_tool_node",
                "__end__": "question_generation_agent"
            }
        )
        agent_builder.add_edge("subject_expert_tool_node", "question_generation_agent")
        agent_builder.add_conditional_edges(
            "question_generation_agent",
            should_continue,
            {
                "use_tool": "question_generation_tool_node",
                "__end__": END
            }
        )
        agent_builder.add_edge("question_generation_tool_node", END)

        # Compile the agent
        self.test_generation_agent = agent_builder.compile()

    def run_test_generation_workflow(self, query: str) -> str:
        result = self.test_generation_agent.invoke({"messages": [HumanMessage(content=query)]})
        return result