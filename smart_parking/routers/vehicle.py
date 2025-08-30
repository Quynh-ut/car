# routers/vehicle.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database import SessionLocal
from models import VehicleLog
from schemas import VehicleEntry, VehicleOut
from datetime import datetime, date
import os
from typing import List

router = APIRouter(
    prefix="/api/vehicle",
    tags=["vehicle"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/entry", response_model=dict)
async def create_vehicle_entry(data: VehicleEntry):
    """Tạo bản ghi xe vào bãi"""
    db = SessionLocal()
    try:
        new_entry = VehicleLog(
            plate_number=data.plate_number,
            vehicle_type=data.vehicle_type,
            entry_time=datetime.now(),
            status="in",
            image_path=data.image_path
        )
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        return {
            "status": "success",
            "message": "Vehicle entry recorded",
            "id": new_entry.id,
            "plate_number": new_entry.plate_number
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.put("/exit/{plate_number}")
async def record_vehicle_exit(plate_number: str):
    """Ghi nhận xe ra khỏi bãi"""
    db = SessionLocal()
    try:
        # Tìm xe đang trong bãi (status = 'in')
        vehicle = db.query(VehicleLog).filter(
            VehicleLog.plate_number == plate_number,
            VehicleLog.status == "in"
        ).first()
        
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found or already exited")
        
        vehicle.exit_time = datetime.now()
        vehicle.status = "out"
        db.commit()
        
        return {
            "status": "success",
            "message": "Vehicle exit recorded",
            "plate_number": plate_number,
            "parking_duration": str(vehicle.exit_time - vehicle.entry_time)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/history")
async def get_vehicle_history(
    plate_number: str = None,
    limit: int = 50,
    offset: int = 0
):
    """Lấy lịch sử xe ra vào"""
    db = SessionLocal()
    try:
        query = db.query(VehicleLog)
        
        if plate_number:
            query = query.filter(VehicleLog.plate_number.contains(plate_number))
        
        vehicles = query.order_by(VehicleLog.entry_time.desc()).offset(offset).limit(limit).all()
        
        result = []
        for vehicle in vehicles:
            result.append({
                "id": vehicle.id,
                "plate_number": vehicle.plate_number,
                "vehicle_type": vehicle.vehicle_type,
                "entry_time": vehicle.entry_time.isoformat() if vehicle.entry_time else None,
                "exit_time": vehicle.exit_time.isoformat() if vehicle.exit_time else None,
                "status": vehicle.status,
                "image_path": vehicle.image_path
            })
        
        return {
            "status": "success",
            "data": result,
            "total": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/current")
async def get_current_vehicles():
    """Lấy danh sách xe đang trong bãi"""
    db = SessionLocal()
    try:
        vehicles = db.query(VehicleLog).filter(VehicleLog.status == "in").all()
        
        result = []
        for vehicle in vehicles:
            result.append({
                "id": vehicle.id,
                "plate_number": vehicle.plate_number,
                "vehicle_type": vehicle.vehicle_type,
                "entry_time": vehicle.entry_time.isoformat(),
                "image_path": vehicle.image_path
            })
        
        return {
            "status": "success",
            "data": result,
            "total": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()