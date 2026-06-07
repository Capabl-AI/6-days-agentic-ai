from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FAQItem(BaseModel):
    question: str = Field(..., description="The question in the FAQ document")
    answer: str = Field(..., description="The answer to the question in the FAQ document")

