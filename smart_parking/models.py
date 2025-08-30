from sqlalchemy import Column, Integer, String, DateTime, Float
from database import Base
from datetime import datetime

class VehicleLog(Base):
    __tablename__ = "vehicle_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, index=True)
    vehicle_type = Column(String, default="oto")
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    image_path = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    status = Column(String, default="in")  
    created_at = Column(DateTime, default=datetime.utcnow)