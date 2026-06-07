from fastapi import FastAPI

from App.services.embedding_service import EmbeddingService
from App.services.faq_agent_service import FAQWorkflow  
from App.services.test_generation_service import TestGenerationService

app = FastAPI(title="Admissions Helper Agent", version="0.0.1")

@app.get("/")
def root():
    return {"message": "Welcome to the Learning Support Agent"}

@app.post("/faq/ingest-faq")
def ingest_faq():
    embedding_service = EmbeddingService()
    embedding_service.process_faq_documents()
    return {"message": "FAQ ingested successfully"}

@app.get("/agent/query-faq")
def query_faq(query: str):
    # embedding_service = EmbeddingService()
    # embedding_service.retrieve_faq(query)
    faq_workflow = FAQWorkflow()
    response = faq_workflow.run_faq_workflow(query)
    return {"message": response["messages"][-1].content}

@app.post("/pdf/ingest-pdf")
def ingest_pdf():
    embedding_service = EmbeddingService(collection_name="c_programming_summary")
    embedding_service.process_pdf_document()
    return {"message": "PDF ingested successfully"}

@app.get("/agent/generate-questions")
def query_pdf(query: str):
    test_generation_service = TestGenerationService()
    response = test_generation_service.run_test_generation_workflow(query)
    return {"message": response["messages"][-1].content}