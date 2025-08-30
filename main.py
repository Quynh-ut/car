import pandas as pd
import cv2
import re
import requests
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import easyocr
import os
import numpy as np

# ======================================================================
# PHAN 1: CAU HINH VA BIEN TOAN CUC - CAP NHAT CHO THU MUC CAR/
# ======================================================================

# Base directory la thu muc car/
BASE_DIR = Path(__file__).parent  # car/

# Duong dan den thu muc chua tat ca ket qua
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Duong dan den thu muc chua anh bien so da cat
CROPS_DIR = RESULTS_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Duong dan den thu muc chua anh bien so da tien xu ly
PREPROC_DIR = RESULTS_DIR / "preproc"
PREPROC_DIR.mkdir(parents=True, exist_ok=True)

# Tai mo hinh CUSTOM license plate detection (trong thu muc car/)
try:
    custom_model_path = BASE_DIR / "license_plate_model.pt"
    if custom_model_path.exists():
        print(f"Dang tai mo hinh CUSTOM nhan dien bien so tu: {custom_model_path}")
        yolo_model = YOLO(str(custom_model_path))
        print("Mo hinh YOLO custom da tai thanh cong!")
        print(f"Cac lop cua mo hinh: {yolo_model.names}")
    else:
        print("Khong tim thay mo hinh custom, su dung YOLOv8 tong quat...")
        yolo_model = YOLO('yolov8n.pt')
        print("Su dung mo hinh tong quat - do chinh xac co the thap hon cho bien so Viet Nam")
        
except Exception as e:
    print(f"Loi khi tai mo hinh YOLO: {e}")
    yolo_model = None

# Tai mo hinh EasyOCR
try:
    print("Dang tai EasyOCR cho bien so Viet Nam...")
    ocr_reader = easyocr.Reader(['en', 'vi'], gpu=False) # Dat gpu=True neu co
    print("EasyOCR da tai thanh cong")
except Exception as e:
    print(f"Loi khi tai EasyOCR: {e}")
    ocr_reader = None

# ======================================================================
# PHAN 2: CAC HAM XU LY CHINH - TOI UU CHO BIEN SO VIET NAM
# ======================================================================

def detect_license_plates(image_path: str, confidence_threshold: float = 0.25):
    """
    Phat hien bien so trong anh bang mo hinh CUSTOM YOLO
    """
    if not yolo_model:
        print("Mo hinh YOLO khong kha dung")
        return []

    try:
        # Chay inference voi custom model
        results = yolo_model(image_path, conf=confidence_threshold, verbose=False)
        detections = []
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Khong the doc anh: {image_path}")
            return []
        
        img_name = Path(image_path).stem
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Kiem tra kich thuoc bounding box
                width = x2 - x1
                height = y2 - y1
                
                if width < 50 or height < 20:  # Qua nho cho bien so
                    continue
                
                if width / height < 1.5:  # Bien so thuong rong hon
                    continue
                
                # Dam bao toa do hop le
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Them padding cho OCR tot hon
                padding = 5
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x2 + padding)
                y2_pad = min(h, y2 + padding)
                
                crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size == 0:
                    continue
                
                # Luu anh da cat
                filename = f"{img_name}_plate_{i:02d}_conf{conf:.2f}.png"
                crop_path = CROPS_DIR / filename
                cv2.imwrite(str(crop_path), crop)
                
                detections.append({
                    "image_path": image_path,
                    "crop_path": str(crop_path),
                    "bbox": f"{x1},{y1},{x2},{y2}",
                    "confidence_yolo": conf,
                    "width": width,
                    "height": height,
                    "area": width * height
                })
        
        print(f"Phat hien {len(detections)} ung vien bien so trong {Path(image_path).name}")
        return detections
        
    except Exception as e:
        print(f"Loi trong phat hien bien so: {e}")
        return []

def preprocess_plate_advanced(img):
    """
    Tien xu ly nang cao cho bien so Viet Nam
    """
    try:
        if img is None or img.size == 0:
            return None, None
            
        # Thay doi kich thuoc thong minh dua tren ti le khung hinh
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 3.5:  # Bien so mot dong (dinh dang cu)
            new_w, new_h = 320, 80
        elif aspect_ratio > 2.5:  # Dinh dang hien dai
            new_w, new_h = 280, 100  
        else:  # Dinh dang vuong (xe may)
            new_w, new_h = 200, 120
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Chuyen sang anh xam
        if len(img_resized.shape) == 3:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_resized.copy()
        
        # Quy trinh xu ly anh nang cao
        
        # 1. Giam nhieu
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Tang cuong do tuong phan voi CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Lam net
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # 4. Nhi phan hoa thich ung
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Phep toan hinh thai hoc de lam sach
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 6. Loai bo nhieu cuoi cung
        binary = cv2.medianBlur(binary, 3)
        
        return enhanced, binary
        
    except Exception as e:
        print(f"Loi trong tien xu ly anh: {e}")
        return None, None

def recognize_text_vietnamese(image):
    """
    OCR toi uu cho bien so Viet Nam
    """
    if not ocr_reader:
        print("EasyOCR khong kha dung")
        return "", 0.0
    
    try:
        # Tham so toi uu cho bien so Viet Nam
        results = ocr_reader.readtext(
            image, 
            allowlist='0123456789ABCDEFGHKLMNPQRSTUVXYZ-.',  
            paragraph=False,
            width_ths=0.8,   # Nguong chieu rong
            height_ths=0.7,  # Nguong chieu cao
            detail=1,
            batch_size=1
        )
        
        if not results:
            return "", 0.0
        
        # Chon ket qua tot nhat dua tren nhieu tieu chi
        valid_results = []
        
        for bbox, text, confidence in results:
            text_clean = text.strip().upper()
            
            # Loc ket qua: toi thieu 5 ky tu va confidence > 0.3
            if len(text_clean) >= 5 and confidence > 0.3:
                # Tinh diem dua tren do dai va confidence
                length_score = min(len(text_clean) / 8.0, 1.0)  # Chuan hoa toi da 8 ky tu
                final_score = confidence * 0.7 + length_score * 0.3
                
                valid_results.append((text_clean, confidence, final_score))
        
        if valid_results:
            # Sap xep theo diem cuoi cung va tra ve ket qua tot nhat
            best_result = max(valid_results, key=lambda x: x[2])
            return best_result[0], best_result[1]
        
        return "", 0.0
        
    except Exception as e:
        print(f"Loi trong OCR: {e}")
        return "", 0.0

def normalize_vietnamese_plate(raw_text: str) -> str:
    """
    Chuan hoa dinh dang bien so Viet Nam
    """
    if not raw_text:
        return ""
    
    # Ban do nham lan mo rong cho tieng Viet
    CONFUSION_MAP = {
        'O': '0', 'o': '0', 'Q': '0', 
        'I': '1', 'l': '1', '|': '1', 
        'Z': '2', 'S': '5', 'B': '8', 'G': '6',
        'T': '7', 'Y': '4', 'A': 'A'  # Giu A nguyen
    }
    
    # Lam sach va chuan hoa
    text = raw_text.upper().strip()
    text = "".join(CONFUSION_MAP.get(ch, ch) for ch in text)
    
    # Loai bo ky tu khong hop le nhung giu dau cham va gach ngang
    text = re.sub(r"[^A-Z0-9\.\-]", "", text)
    
    # Cac mau bien so Viet Nam
    patterns = [
        # Dinh dang cu: 12A-34567 hoac 12A-345.67
        r"^(\d{2})([A-Z])(\d{3,5})\.?(\d{0,3})$",
        # Dinh dang moi: 12L1-234.56  
        r"^(\d{2})([A-Z])(\d{1})(\d{3})\.?(\d{2,3})$",
        # Tien to 3 chu so: 123A-45678
        r"^(\d{3})([A-Z])(\d{4,6})$",
        # Co gach ngang san: 12-A12345
        r"^(\d{2,3})\-?([A-Z]\d?)(\d{3,6})$"
    ]
    
    # Thu khop voi cac mau
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            groups = match.groups()
            
            if len(groups) == 4 and groups[3]:  # Co hau to
                if len(groups[2]) >= 3:
                    # Dinh dang: 12A-123.45
                    return f"{groups[0]}{groups[1]}-{groups[2]}.{groups[3]}"
                else:
                    return f"{groups[0]}{groups[1]}-{groups[2]}{groups[3]}"
            elif len(groups) == 5:  # Dinh dang moi
                return f"{groups[0]}{groups[1]}{groups[2]}-{groups[3]}.{groups[4]}"
            elif len(groups) == 3:  # Dinh dang don gian
                num_part = groups[2]
                if len(num_part) >= 4:
                    return f"{groups[0]}{groups[1]}-{num_part[:-2]}.{num_part[-2:]}"
                else:
                    return f"{groups[0]}{groups[1]}-{num_part}"
    
    # Neu khong khop mau nao, thu dinh dang don gian
    alpha_match = re.search(r'[A-Z]', text)
    if alpha_match:
        alpha_pos = alpha_match.start()
        if alpha_pos > 1:
            prefix = text[:alpha_pos]
            suffix = text[alpha_pos:]
            
            alpha_part = re.match(r'^([A-Z]+)(.*)$', suffix)
            if alpha_part:
                letters = alpha_part.group(1)
                numbers = alpha_part.group(2)
                
                if len(numbers) >= 4:
                    return f"{prefix}{letters}-{numbers[:-2]}.{numbers[-2:]}"
                else:
                    return f"{prefix}{letters}-{numbers}"
    
    return text

def send_to_backend(data: dict):
    """
    Gui ket qua den FastAPI backend
    """
    # Cap nhat URL cho cau truc car/smart_parking
    backend_url = "http://127.0.0.1:8000/api/vehicle/entry"
    try:
        payload = {
            "plate_number": data.get("license_plate", ""),
            "vehicle_type": "oto",  # Mac dinh
            "image_path": data.get("image_path", "")
        }
        
        response = requests.post(backend_url, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"Da gui {payload['plate_number']} len backend thanh cong")
            return response.json()
        else:
            print(f"Loi backend {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Loi ket noi backend: {e}")
        return None

# ======================================================================
# MAIN PIPELINE FUNCTION
# ======================================================================

def run_license_plate_pipeline(image_list: list, save_results: bool = True):
    """
    Quy trinh hoan chinh cho nhan dien bien so Viet Nam
    """
    print("HE THONG NHAN DIEN BIEN SO VIET NAM")
    print("=" * 60)
    
    if not image_list:
        print("Khong co anh nao duoc cung cap")
        return []
    
    all_results = []
    
    # Buoc 1: Phat hien bien so voi mo hinh Custom
    print("Buoc 1: Phat hien bien so...")
    detection_rows = []
    
    for img_path in tqdm(image_list, desc="Dang phat hien bien so"):
        detections = detect_license_plates(img_path, confidence_threshold=0.25)
        detection_rows.extend(detections)
    
    if not detection_rows:
        print("Khong phat hien duoc bien so nao")
        return []
        
    print(f"Da phat hien {len(detection_rows)} bien so tiem nang")
    
    # Buoc 2: Tien xu ly anh nang cao
    print("\nBuoc 2: Tien xu ly anh nang cao...")
    processed_rows = []
    
    for row in tqdm(detection_rows, desc="Dang tien xu ly"):
        crop_path = Path(row["crop_path"])
        img = cv2.imread(str(crop_path))
        
        if img is None:
            continue
        
        enhanced, binary = preprocess_plate_advanced(img)
        if enhanced is None or binary is None:
            continue
            
        # Luu anh da xu ly
        out_enh = PREPROC_DIR / f"{crop_path.stem}_enhanced.png"
        out_bin = PREPROC_DIR / f"{crop_path.stem}_binary.png"
        cv2.imwrite(str(out_enh), enhanced)
        cv2.imwrite(str(out_bin), binary)
        
        processed_rows.append({
            **row,
            "enhanced_path": str(out_enh),
            "binary_path": str(out_bin),
        })
    
    print(f"Da tien xu ly {len(processed_rows)} anh")

    # Buoc 3: Nhan dien OCR tieng Viet
    print("\nBuoc 3: Nhan dien OCR tieng Viet...")
    final_results = []
    
    for row in tqdm(processed_rows, desc="Nhan dien OCR"):
        binary_img = cv2.imread(row["binary_path"])
        enhanced_img = cv2.imread(row["enhanced_path"])
        
        if binary_img is None or enhanced_img is None:
            continue
            
        # Thu ca anh tang cuong va anh nhi phan
        text_binary, conf_binary = recognize_text_vietnamese(binary_img)
        text_enhanced, conf_enhanced = recognize_text_vietnamese(enhanced_img)
        
        # Chon ket qua tot nhat
        if conf_binary > conf_enhanced:
            raw_text, confidence = text_binary, conf_binary
            source = "binary"
        else:
            raw_text, confidence = text_enhanced, conf_enhanced
            source = "enhanced"
        
        # Chuan hoa dinh dang bien so
        normalized = normalize_vietnamese_plate(raw_text)
        
        result = {
            **row,
            "raw_text": raw_text,
            "normalized_plate": normalized,
            "confidence": confidence,
            "ocr_source": source,
            "is_valid": bool(normalized and confidence > 0.4 and len(normalized) >= 6)
        }
        
        final_results.append(result)
        all_results.append(result)
    
    print(f"OCR hoan thanh: {len(final_results)} ket qua")
    
    # Buoc 4: Gui ket qua hop le len backend
    print("\nBuoc 4: Dang gui ket qua len backend...")
    valid_results = [r for r in final_results if r["is_valid"]]
    sent_count = 0
    
    for result in tqdm(valid_results, desc="Dang gui len backend"):
        data_to_send = {
            "license_plate": result["normalized_plate"],
            "raw_text": result["raw_text"],
            "confidence": result["confidence"],
            "image_path": result["image_path"],
            "bbox": result["bbox"],
        }
        
        backend_response = send_to_backend(data_to_send)
        if backend_response:
            sent_count += 1
    
    print(f"Da gui {sent_count}/{len(valid_results)} ket qua hop le")
    
    # Buoc 5: Luu ket qua
    if save_results and final_results:
        df_results = pd.DataFrame(final_results)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = RESULTS_DIR / f"license_plate_results_{timestamp}.csv"
        df_results.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Ket qua da luu vao: {output_file}")
    
    # Thong ke cuoi cung
    print("\n" + "=" * 60)
    print("THONG KE CUOI CUNG:")
    print(f"Anh da xu ly: {len(image_list)}")
    print(f"Bien so phat hien: {len(detection_rows)}")
    print(f"Ket qua hop le: {len(valid_results)}")
    print(f"Da gui len backend: {sent_count}")
    
    if valid_results:
        plates = [r["normalized_plate"] for r in valid_results]
        print(f"Bien so tim thay: {', '.join(plates)}")
    
    return all_results

# ======================================================================
# MAIN FUNCTION
# ======================================================================

def main():
    """Ham chinh de test he thong"""
    print("HE THONG NHAN DIEN BIEN SO VIET NAM")
    print("Mo hinh YOLO Custom + Quy trinh OCR Nang cao")
    print("=" * 60)
    
    # Tim anh test
    test_images = []
    
    # Tim trong thu muc hien tai (car/)
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        test_images.extend(list(BASE_DIR.glob(ext)))
    
    # Tim trong thu muc con demo_images
    demo_dir = BASE_DIR / "demo_images"
    demo_dir.mkdir(exist_ok=True)
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(demo_dir.glob(ext)))
    
    if not test_images:
        print("Khong tim thay anh test nao. Dang tao anh demo...")
        # Tao anh demo don gian
        demo_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(demo_img, (50, 70), (350, 130), (0, 0, 0), 2)
        cv2.putText(demo_img, "59A-123.45", (80, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        demo_path = demo_dir / "demo_plate.jpg"
        cv2.imwrite(str(demo_path), demo_img)
        test_images = [demo_path]
        print(f"Da tao anh demo: {demo_path}")
    
    print(f"Tim thay {len(test_images)} anh de xu ly:")
    for img in test_images[:5]:  # Hien thi toi da 5 anh
        print(f"  - {img.name}")
    
    if len(test_images) > 5:
        print(f"  ... va {len(test_images) - 5} anh khac")
    
    # Chay quy trinh
    results = run_license_plate_pipeline([str(img) for img in test_images])
    
    print(f"\nQuy trinh hoan thanh! Da xu ly {len(results)} ket qua.")
    
    # Hien thi tom tat
    valid_plates = [r["normalized_plate"] for r in results if r.get("is_valid")]
    if valid_plates:
        print(f"Da nhan dien thanh cong bien so: {', '.join(set(valid_plates))}")
    else:
        print("Khong nhan dien duoc bien so hop le nao")
    
    return results

if __name__ == '__main__':
    main()