from App.config import settings

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.messages import SystemMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun

# from App.services.embedding_service import EmbeddingService
from App.tools.shared_tools import retrieve_context_from_textbook

# @tool
# def retrieve_context_from_textbook(query: str) -> list[str]:
#     """
#     This tool is used to retrieve the answer from the summary document.
#     The retriever best matches the questions of the user with the chunks of the summary document.
#     And returns the closest matching chunk to the user's query
#     """
#     embedding_service = EmbeddingService(collection_name="c_programming_summary")
#     context = embedding_service.retrieve_textbook_context(query)
#     return context

web_search = DuckDuckGoSearchRun()

class QuestionGenerationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.2,
            max_tokens=512
        )

        self.tools_list = [retrieve_context_from_textbook, web_search]
        self.tools_by_name = {tool.name: tool for tool in self.tools_list}

    def llm_call(self, state: dict) -> str:
        llm_with_tools = self.llm.bind_tools(self.tools_list)
        
        return {
            "messages": [
                llm_with_tools.invoke(
                    [
                        SystemMessage(
                            content="""
                            You are an helpful Question Generation Agent who is an expert in generating questions on C Programming.
                            You have access to a "retrieve_context_from_textbook" tool that can be used to retrieve context from the textbook.
                            Use this tool to generate the questions for test / homework.
                            If the tool does not provide the context, respond politely saying that you cannot generate the questions.
                            The User Prompt will simply include the topic or list of topics on which questions need to be generated.
                            For eg: if the user asks for "arrays", "strings", "pointers", "functions", etc. in C Programming, you need to generate the questions on these topics only.
                            Do not generate the questions on any other topic.
                            Do not generate questions on subtopics not present in the book.
                            Unless the user explicitly asks for refer to "web_search" tool to acquire context and generate questions.
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