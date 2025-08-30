# routers/stats.py
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from database import SessionLocal
from models import VehicleLog
from datetime import datetime, date, timedelta

router = APIRouter(
    prefix="/api/stats",
    tags=["statistics"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/daily")
async def get_daily_stats():
    """Thống kê theo ngày"""
    db = SessionLocal()
    try:
        today = date.today()
        
        # Xe vào hôm nay
        today_in = db.query(VehicleLog).filter(
            func.date(VehicleLog.entry_time) == today
        ).count()
        
        # Xe ra hôm nay  
        today_out = db.query(VehicleLog).filter(
            func.date(VehicleLog.exit_time) == today
        ).count()
        
        # Xe đang gửi
        current_parking = db.query(VehicleLog).filter(
            VehicleLog.status == "in"
        ).count()
        
        return {
            "status": "success",
            "data": {
                "today_in": today_in,
                "today_out": today_out,
                "current_parking": current_parking,
                "date": today.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/monthly")
async def get_monthly_stats():
    """Thống kê theo tháng"""
    db = SessionLocal()
    try:
        # Tháng hiện tại
        now = datetime.now()
        start_of_month = date(now.year, now.month, 1)
        
        # Tổng xe trong tháng
        monthly_total = db.query(VehicleLog).filter(
            VehicleLog.entry_time >= start_of_month
        ).count()
        
        # Thống kê theo loại xe
        vehicle_types = db.query(
            VehicleLog.vehicle_type,
            func.count(VehicleLog.id).label('count')
        ).filter(
            VehicleLog.entry_time >= start_of_month
        ).group_by(VehicleLog.vehicle_type).all()
        
        type_stats = {vtype: count for vtype, count in vehicle_types}
        
        return {
            "status": "success",
            "data": {
                "monthly_total": monthly_total,
                "vehicle_types": type_stats,
                "month": f"{now.year}-{now.month:02d}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/hourly/{target_date}")
async def get_hourly_stats(target_date: str):
    """Thống kê theo giờ trong ngày"""
    db = SessionLocal()
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        
        # Thống kê theo giờ
        hourly_data = db.query(
            func.extract('hour', VehicleLog.entry_time).label('hour'),
            func.count(VehicleLog.id).label('count')
        ).filter(
            func.date(VehicleLog.entry_time) == target
        ).group_by(func.extract('hour', VehicleLog.entry_time)).all()
        
        # Tạo dict với tất cả 24 giờ (0 nếu không có dữ liệu)
        hourly_stats = {hour: 0 for hour in range(24)}
        for hour, count in hourly_data:
            hourly_stats[int(hour)] = count
        
        return {
            "status": "success",
            "data": {
                "hourly_stats": hourly_stats,
                "date": target_date,
                "total_day": sum(hourly_stats.values())
            }
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/top-plates")
async def get_frequent_plates(limit: int = 10):
    """Top biển số xe ra vào thường xuyên"""
    db = SessionLocal()
    try:
        frequent_plates = db.query(
            VehicleLog.plate_number,
            func.count(VehicleLog.id).label('visit_count'),
            func.max(VehicleLog.entry_time).label('last_visit')
        ).group_by(
            VehicleLog.plate_number
        ).order_by(
            func.count(VehicleLog.id).desc()
        ).limit(limit).all()
        
        result = []
        for plate, count, last_visit in frequent_plates:
            result.append({
                "plate_number": plate,
                "visit_count": count,
                "last_visit": last_visit.isoformat() if last_visit else None
            })
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/summary")
async def get_summary_stats():
    """Tổng quan thống kê"""
    db = SessionLocal()
    try:
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        # Thống kê hôm nay
        today_entries = db.query(VehicleLog).filter(
            func.date(VehicleLog.entry_time) == today
        ).count()
        
        # Thống kê hôm qua
        yesterday_entries = db.query(VehicleLog).filter(
            func.date(VehicleLog.entry_time) == yesterday
        ).count()
        
        # Xe đang trong bãi
        current_vehicles = db.query(VehicleLog).filter(
            VehicleLog.status == "in"
        ).count()
        
        # Tổng số lượt
        total_entries = db.query(VehicleLog).count()
        
        # % thay đổi so với hôm qua
        change_percent = 0
        if yesterday_entries > 0:
            change_percent = round(((today_entries - yesterday_entries) / yesterday_entries) * 100, 2)
        
        return {
            "status": "success",
            "data": {
                "today_entries": today_entries,
                "yesterday_entries": yesterday_entries,
                "current_vehicles": current_vehicles,
                "total_entries": total_entries,
                "change_percent": change_percent,
                "date": today.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()