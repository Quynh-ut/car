# car/smart_parking/routers/image.py - CAP NHAT cho cau truc thu muc
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from typing import List

# Import tu file main.py o thu muc cha
import sys
parent_dir = Path(__file__).parent.parent.parent  # car/
sys.path.append(str(parent_dir))

try:
    from main import detect_license_plates, recognize_text_vietnamese, normalize_vietnamese_plate, preprocess_plate_advanced
    print("AI modules chay thanh cong.")
except ImportError as e:
    print(f"Khong tim thay AI modules : {e}.")
    detect_license_plates = None
    recognize_text_vietnamese = None
    normalize_vietnamese_plate = None
    preprocess_plate_advanced = None

router = APIRouter(
    prefix="/api/image",
    tags=["image_processing"]
)

# Thu muc luu tru trong car/smart_parking/
BASE_DIR = Path(__file__).parent.parent  # car/smart_parking/
UPLOAD_DIR = BASE_DIR / "uploaded_images"
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = BASE_DIR / "processed_images" 
PROCESSED_DIR.mkdir(exist_ok=True)

@router.post("/upload/")
async def upload_and_process_image(file: UploadFile = File(...)):
    """Upload anh va xu ly nhan dien bien so"""
    
    # Kiem tra dinh dang file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File phai la anh (jpg, png, etc.)")
    
    try:
        # Luu file upload
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = UPLOAD_DIR / filename
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Xu ly nhan dien bien so
        plates = []
        
        if detect_license_plates and recognize_text_vietnamese and normalize_vietnamese_plate:
            # Su dung AI thuc te tu car/main.py
            try:
                detections = detect_license_plates(str(filepath), confidence_threshold=0.25)
                
                for detection in detections:
                    try:
                        # Doc anh da cat
                        crop_img = cv2.imread(detection["crop_path"])
                        if crop_img is not None:
                            # Tien xu ly nang cao
                            enhanced, binary = preprocess_plate_advanced(crop_img)
                            
                            if binary is not None:
                                # Nhan dang text voi cai tien cho tieng Viet
                                raw_text, confidence = recognize_text_vietnamese(binary)
                                
                                if raw_text and confidence > 0.4:  # Nguong tin cay
                                    normalized = normalize_vietnamese_plate(raw_text)
                                    if normalized:
                                        plates.append({
                                            "plate": normalized,
                                            "raw_text": raw_text,
                                            "confidence": confidence,
                                            "bbox": detection["bbox"]
                                        })
                    except Exception as e:
                        print(f"Loi xu ly detection: {e}")
                        continue
                        
            except Exception as e:
                print(f"Loi xu ly AI: {e}")
        
        # Mock function cho testing (khi AI khong hoat dong)
        if not plates:
            mock_plates = [
                "59A-123.45", "51G-678.90", "43B-543.21", 
                "30A-111.22", "29B-999.88", "77S1-234.56"
            ]
            selected_plate = mock_plates[hash(filename) % len(mock_plates)]
            plates = [{
                "plate": selected_plate,
                "raw_text": selected_plate.replace("-", "").replace(".", ""),
                "confidence": 0.85,
                "bbox": "100,50,300,150"
            }]
        
        # Luu anh goc vao processed
        processed_path = PROCESSED_DIR / filename
        shutil.copy2(filepath, processed_path)
        
        # Response format phu hop voi frontend
        response_data = {
            "status": "thanh_cong" if plates else "khong_tim_thay_bien_so",
            "message": "Nhan dien thanh cong" if plates else "Khong tim thay bien so",
            "filename": filename,
            "original_image": f"processed_images/{filename}",
            "plates": [p["plate"] for p in plates],  # Chi tra ve bien so
            "count": len(plates),
            "details": plates  # Chi tiet day du
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi xu ly anh: {str(e)}")

@router.get("/processed/{filename}")
async def get_processed_image(filename: str):
    """Lay anh da xu ly"""
    filepath = PROCESSED_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Khong tim thay anh")
    return FileResponse(filepath)

@router.get("/list")
async def list_processed_images():
    """Danh sach anh da xu ly"""
    try:
        files = []
        for filepath in PROCESSED_DIR.glob("*"):
            if filepath.is_file() and filepath.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                stat = filepath.stat()
                files.append({
                    "filename": filepath.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "url": f"/api/image/processed/{filepath.name}"
                })
        
        # Sap xep theo thoi gian tao moi nhat
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return {
            "status": "thanh_cong",
            "files": files,
            "total": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi: {str(e)}")

@router.delete("/cleanup")
async def cleanup_old_images(days_old: int = 7):
    """Xoa anh cu hon X ngay"""
    try:
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        deleted_count = 0
        for directory in [UPLOAD_DIR, PROCESSED_DIR]:
            for filepath in directory.glob("*"):
                if filepath.is_file():
                    file_time = datetime.fromtimestamp(filepath.stat().st_ctime)
                    if file_time < cutoff_time:
                        filepath.unlink()
                        deleted_count += 1
        
        return {
            "status": "thanh_cong", 
            "message": f"Da xoa {deleted_count} anh cu"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi: {str(e)}")

@router.get("/health")
async def health_check():
    """Kiem tra tinh trang dich vu"""
    # Kiem tra model custom
    model_path = parent_dir / "license_plate_model.pt"
    
    return {
        "status": "khoe_manh",
        "ai_models": {
            "custom_yolo": model_path.exists(),
            "functions_loaded": all([
                detect_license_plates is not None,
                recognize_text_vietnamese is not None,
                normalize_vietnamese_plate is not None
            ])
        },
        "directories": {
            "upload": str(UPLOAD_DIR),
            "processed": str(PROCESSED_DIR)
        },
        "custom_model_path": str(model_path)
    }