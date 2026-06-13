from pydantic import BaseModel, Field
from typing import List, Dict, Any

from datetime import datetime

from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class FAQItem(BaseModel):
    question: str = Field(..., description="The question in the FAQ document")
    answer: str = Field(..., description="The answer to the question in the FAQ document")

class ConversationSummary(Base):
    __tablename__ = 'conversation_summaries'

    session_id = Column(String, primary_key=True)
    summary = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)