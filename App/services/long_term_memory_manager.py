from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from App.config import settings

from App.models.schemas import ConversationSummary, Base

class LongTermMemoryManager:
    def __init__(self):
        db_url = f"sqlite:///{settings.SQLITE_DB_PATH}"
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    def get_summary(self, session_id: str) -> str:
        session = self.Session()
        try:
            record = session.query(ConversationSummary).filter_by(session_id=session_id).first()
            return record.summary if record else ""
        finally:
            session.close()
    def save_summary(self, session_id: str, summary: str):
        session = self.Session()
        try:
            record = session.query(ConversationSummary).filter_by(session_id=session_id).first()
            if record:
                record.summary = summary
                record.updated_at = datetime.utcnow()
            else:
                record = ConversationSummary(session_id=session_id, summary=summary)
                session.add(record)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()