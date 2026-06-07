from langchain_core.tools import tool
from App.services.embedding_service import EmbeddingService

@tool
def retrieve_context_from_textbook(query: str) -> list[str]:
    """
    This tool is used to retrieve the answer from the summary document.
    The retriever best matches the questions of the user with the chunks of the summary document.
    And returns the closest matching chunk to the user's query
    """
    embedding_service = EmbeddingService(collection_name="c_programming_summary")
    context = embedding_service.retrieve_textbook_context(query)
    return context