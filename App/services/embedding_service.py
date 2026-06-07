from App.config import settings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from App.models.schemas import FAQItem

import json

class EmbeddingService:
    def __init__(self, collection_name: str = "faq_collection", persist_directory: str = "./mit_vectordb_directory"):
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.HF_MODEL_NAME,
            show_progress=True
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    def _embed_text_and_store_in_vector_db(self, documents: list[Document]) -> None:
        """
        Generates embeddings of the given text and stores them to ChromaDB
        """
        self.vector_store.add_documents(documents=documents)

    def process_faq_documents(self) -> None:
        """
        Processes a list of FAQ items and stores them to ChromaDB
        """
        file_path = "./assets/faq.json"
        documents = []

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                documents.append(Document(
                    page_content=item["Question"],
                    metadata={"Answer": item["Answer"]}
                ))

        self._embed_text_and_store_in_vector_db(documents=documents)

    def retrieve_faq(self, query: str) -> list[FAQItem]:
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 2})
        response = retriever.invoke(query)
        response = [FAQItem(question=doc.page_content, answer=doc.metadata["Answer"]) for doc in response]
        # print(response)
        # print(response[0].answer)
        return response

    def process_pdf_document(self) -> None:
        file_path = "./assets/Let us c - Summary.pdf"
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
        )

        split_documents = splitter.split_documents(documents=documents)
        
        self._embed_text_and_store_in_vector_db(documents=split_documents)

    def retrieve_textbook_context(self, query: str) -> list[str]:
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 2})
        response = retriever.invoke(query)
        return [doc.page_content for doc in response]
        