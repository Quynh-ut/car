# crud.py
from sqlalchemy.orm import Session
from models import VehicleLog
from schemas import VehicleEntry
from datetime import datetime, date
from database import SessionLocal

def create_vehicle_entry(data: VehicleEntry):
    db = SessionLocal()
    new_entry = VehicleLog(
        plate_number=data.plate_number,
        vehicle_type=data.vehicle_type,
        image_path=data.image_path
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    db.close()
    return {"status": "ok", "id": new_entry.id}

def get_daily_stats():
    db = SessionLocal()
    today = date.today()
    count = db.query(VehicleLog).filter(VehicleLog.entry_time >= today).count()
    db.close()
    return {"total_entries_today": count}
