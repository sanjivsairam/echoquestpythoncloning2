from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.db.database import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = {"schema": "valuevista"}

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    method = Column(String)
    path = Column(String)
    user_identity = Column(String)
    client_ip = Column(String)
    user_agent = Column(String)
    status_code = Column(Integer)
    duration_ms = Column(Float)
