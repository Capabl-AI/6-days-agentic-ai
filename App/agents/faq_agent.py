from App.config import settings

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.messages import SystemMessage, ToolMessage

from App.services.embedding_service import EmbeddingService

@tool
def retrieve_answer_from_context(query: str) -> list[str]:
    """
    This tool is used to retrieve the answer from the FAQ documents.
    The retriever best matches the questions of the user with the questions of the FAQ documents.
    And returns the reference answer to the FAQ question
    """
    embedding_service = EmbeddingService()
    context = embedding_service.retrieve_faq(query)
    context = [faq.answer for faq in context]
    return context

class FAQAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.2,
            max_tokens=512
        )

        self.tools_list = [retrieve_answer_from_context]
        self.tools_by_name = {tool.name: tool for tool in self.tools_list}

    def llm_call(self, state: dict) -> str:
        llm_with_tools = self.llm.bind_tools(self.tools_list)
        
        return {
            "messages": [
                llm_with_tools.invoke(
                    [
                        SystemMessage(
                            content="""
                            You are an helpful FAQ Agent who is an expert in answering questions related to the C programming course.
                            You have access to a "retrieve_answer_from_context" tool that can be used to retrieve the answer from the FAQ documents.
                            Use this tool to answer the user's question.
                            If the tool does not provide the answer, respond politely saying that the question is out of scope.
                            """
                        )
                    ]
                    + state["messages"])
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }

    def tool_node(self, state: dict) -> dict:
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:

            if 'self' in tool_call["args"]:
                del tool_call["args"]['self']

            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])

            if isinstance(observation, list):
                content_string = "\n".join(observation)
            else:
                content_string = str(observation)

            result.append(ToolMessage(content=content_string, tool_call_id=tool_call["id"]))
        return {"messages": result}