# schemas.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class VehicleEntry(BaseModel):
    plate_number: str
    vehicle_type: str
    image_path: Optional[str] = None

class VehicleOut(VehicleEntry):
    entry_time: datetime
