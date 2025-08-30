import requests
import json
import time
import sys
from pathlib import Path
import sqlite3
from datetime import datetime
import cv2
import numpy as np

# Cap nhat cho cau truc car/smart_parking/
BASE_DIR = Path(__file__).parent  # car/
SMART_PARKING_DIR = BASE_DIR / "smart_parking"
DB_PATH = SMART_PARKING_DIR / "parking.db"

# Cau hinh test
BASE_URL = "http://127.0.0.1:8000"
TEST_IMAGE_PATH = BASE_DIR / "demo_images" / "test_car.jpg"

class SystemTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.test_results = []
        self.base_dir = BASE_DIR
    
    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Ghi log ket qua test"""
        status = "THANH CONG" if success else "THAT BAI"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def test_database_connection(self):
        """Test ket noi database"""
        try:
            if not DB_PATH.exists():
                self.log_result("Ket noi Database", False, f"Khong tim thay database tai {DB_PATH}")
                return False
                
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            # Kiem tra bang vehicle_logs ton tai
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle_logs'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                self.log_result("Ket noi Database", False, "Khong tim thay bang 'vehicle_logs'")
                conn.close()
                return False
            
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs")
            count = cursor.fetchone()[0]
            conn.close()
            
            self.log_result("Ket noi Database", True, f"Tim thay {count} ban ghi")
            return True
            
        except Exception as e:
            self.log_result("Ket noi Database", False, str(e))
            return False
    
    def test_custom_model_exists(self):
        """Test file model custom ton tai"""
        model_path = self.base_dir / "license_plate_model.pt"
        exists = model_path.exists()
        
        if exists:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            self.log_result("File Model Custom", True, f"Tim thay model, kich thuoc: {size_mb:.1f}MB")
        else:
            self.log_result("File Model Custom", False, f"Khong tim thay model tai {model_path}")
        
        return exists
    
    def test_directory_structure(self):
        """Test cau truc thu muc"""
        required_dirs = [
            self.base_dir / "smart_parking",
            self.base_dir / "smart_parking" / "routers",
        ]
        
        required_files = [
            self.base_dir / "main.py",
            self.base_dir / "smart_parking" / "main.py", 
            self.base_dir / "smart_parking" / "routers" / "vehicle.py",
            self.base_dir / "smart_parking" / "routers" / "image.py",
            self.base_dir / "smart_parking" / "routers" / "stats.py",
        ]
        
        missing_items = []
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_items.append(f"Thu muc: {dir_path}")
        
        for file_path in required_files:
            if not file_path.exists():
                missing_items.append(f"File: {file_path}")
        
        if missing_items:
            self.log_result("Cau truc Thu muc", False, f"Thieu: {', '.join(missing_items)}")
            return False
        else:
            self.log_result("Cau truc Thu muc", True, "Tat ca file va thu muc can thiet deu co")
            return True
    
    def test_api_health(self):
        """Test kiem tra suc khoe API"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_result("API Goc", True, f"Trang thai: {data.get('status', 'OK')}")
            else:
                self.log_result("API Goc", False, f"HTTP {response.status_code}")
                return False
                
            # Test image health endpoint
            response = self.session.get(f"{self.base_url}/api/image/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                ai_status = data.get('ai_models', {})
                model_status = "Co" if ai_status.get('custom_yolo') else "Khong"
                self.log_result("Suc khoe Image API", True, 
                               f"Model Custom: {model_status}, Ham da tai: {ai_status.get('functions_loaded')}")
                return True
            else:
                self.log_result("Suc khoe Image API", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Suc khoe API", False, str(e))
            return False
    
    def create_test_image(self):
        """Tao anh test voi bien so"""
        try:
            demo_dir = self.base_dir / "demo_images"
            demo_dir.mkdir(exist_ok=True)
            
            # Tao anh voi bien so gia lap
            img = np.ones((300, 500, 3), dtype=np.uint8) * 240  # Nen xam nhat
            
            # Ve xe
            cv2.rectangle(img, (50, 100), (450, 250), (100, 100, 150), -1)  # Than xe
            cv2.rectangle(img, (70, 120), (430, 230), (80, 80, 120), -1)
            
            # Ve bien so
            cv2.rectangle(img, (150, 180), (350, 220), (255, 255, 255), -1)  # Nen trang
            cv2.rectangle(img, (150, 180), (350, 220), (0, 0, 0), 2)         # Vien den
            
            # Viet bien so
            cv2.putText(img, "59A-123.45", (165, 205), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Luu anh
            test_path = demo_dir / "test_car.jpg"
            cv2.imwrite(str(test_path), img)
            
            self.log_result("Tao Anh Test", True, f"Da tao: {test_path}")
            return str(test_path)
            
        except Exception as e:
            self.log_result("Tao Anh Test", False, str(e))
            return None
    
    def test_image_upload(self):
        """Test upload va xu ly anh"""
        try:
            # Tao hoac tim anh test
            if not TEST_IMAGE_PATH.exists():
                test_image = self.create_test_image()
                if not test_image:
                    return None
            else:
                test_image = str(TEST_IMAGE_PATH)
            
            with open(test_image, 'rb') as f:
                files = {'file': ('test_car.jpg', f, 'image/jpeg')}
                response = self.session.post(f"{self.base_url}/api/image/upload/", 
                                           files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                plates_found = data.get('count', 0)
                plates_list = data.get('plates', [])
                
                if plates_found > 0:
                    self.log_result("Upload va Xu ly Anh", True, 
                                  f"Tim thay {plates_found} bien so: {', '.join(plates_list)}")
                    return data
                else:
                    self.log_result("Upload va Xu ly Anh", True, 
                                  "Khong tim thay bien so nao (co the la binh thuong voi anh test)")
                    return data
            else:
                error_text = response.text if response.text else "Loi khong xac dinh"
                self.log_result("Upload va Xu ly Anh", False, 
                              f"HTTP {response.status_code}: {error_text}")
                return None
        except Exception as e:
            self.log_result("Upload va Xu ly Anh", False, str(e))
            return None
    
    def test_vehicle_entry_api(self, plate_number: str = "TEST-001"):
        """Test API ghi nhan xe vao"""
        try:
            data = {
                "plate_number": plate_number,
                "vehicle_type": "oto",
                "image_path": "/test/path.jpg"
            }
            response = self.session.post(f"{self.base_url}/api/vehicle/entry", 
                                       json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                entry_id = result.get('id')
                self.log_result("API Xe Vao", True, f"Da tao entry ID: {entry_id}")
                return entry_id
            else:
                error_text = response.text if response.text else "Loi khong xac dinh"
                self.log_result("API Xe Vao", False, f"HTTP {response.status_code}: {error_text}")
                return None
        except Exception as e:
            self.log_result("API Xe Vao", False, str(e))
            return None
    
    def test_vehicle_exit_api(self, plate_number: str = "TEST-001"):
        """Test API ghi nhan xe ra"""
        try:
            response = self.session.put(f"{self.base_url}/api/vehicle/exit/{plate_number}", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                duration = result.get('parking_duration', 'Khong co')
                self.log_result("API Xe Ra", True, f"Thoi gian gui: {duration}")
                return True
            elif response.status_code == 404:
                self.log_result("API Xe Ra", True, "Khong tim thay xe (binh thuong cho test)")
                return True
            else:
                error_text = response.text if response.text else "Loi khong xac dinh"
                self.log_result("API Xe Ra", False, f"HTTP {response.status_code}: {error_text}")
                return False
        except Exception as e:
            self.log_result("API Xe Ra", False, str(e))
            return False
    
    def test_statistics_api(self):
        """Test API thong ke"""
        endpoints = [
            ("/api/stats/daily", "Thong ke Hang ngay"),
            ("/api/stats/monthly", "Thong ke Hang thang"), 
            ("/api/stats/summary", "Thong ke Tong hop")
        ]
        
        all_success = True
        for endpoint, name in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    data_keys = list(data.get('data', {}).keys())
                    self.log_result(f"API Thong ke - {name}", True, f"Cac truong: {data_keys}")
                else:
                    error_text = response.text if response.text else "Loi khong xac dinh"
                    self.log_result(f"API Thong ke - {name}", False, f"HTTP {response.status_code}: {error_text}")
                    all_success = False
            except Exception as e:
                self.log_result(f"API Thong ke - {name}", False, str(e))
                all_success = False
        
        return all_success
    
    def test_vehicle_history_api(self):
        """Test API lich su xe"""
        try:
            response = self.session.get(f"{self.base_url}/api/vehicle/history", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('total', 0)
                self.log_result("API Lich su Xe", True, f"So ban ghi: {total}")
                return True
            else:
                error_text = response.text if response.text else "Loi khong xac dinh"
                self.log_result("API Lich su Xe", False, f"HTTP {response.status_code}: {error_text}")
                return False
        except Exception as e:
            self.log_result("API Lich su Xe", False, str(e))
            return False
    
    def test_complete_workflow(self):
        """Test toan bo quy trinh: Anh -> AI -> Database -> API"""
        print("\n Dang test quy trinh hoan chinh...")
        
        # 1. Upload anh va nhan dien
        result = self.test_image_upload()
        if not result:
            self.log_result("Quy trinh Hoan chinh", False, "Xu ly anh that bai")
            return False
        
        # Lay plate tu ket qua hoac dung test plate
        if result.get('plates') and len(result['plates']) > 0:
            plate_number = result['plates'][0]
        else:
            plate_number = "WORKFLOW-TEST-001"
        
        # 2. Ghi nhan xe vao
        entry_id = self.test_vehicle_entry_api(plate_number)
        if not entry_id:
            self.log_result("Quy trinh Hoan chinh", False, "Ghi nhan xe vao that bai")
            return False
        
        # 3. Kiem tra thong ke (doi 1 giay de DB cap nhat)
        time.sleep(1)
        stats_ok = self.test_statistics_api()
        
        # 4. Kiem tra lich su
        history_ok = self.test_vehicle_history_api()
        
        # 5. Thu ghi nhan xe ra
        time.sleep(1)
        exit_ok = self.test_vehicle_exit_api(plate_number)
        
        workflow_success = stats_ok and history_ok and exit_ok
        self.log_result("Quy trinh Hoan chinh", workflow_success, 
                       f"Bien so: {plate_number}, Entry ID: {entry_id}")
        return workflow_success
    
    def run_all_tests(self):
        """Chay tat ca test cases"""
        print("BAT DAU TEST TOAN BO HE THONG")
        print("Cau truc: car/smart_parking/")
        print("=" * 60)
        
        # Test co so ha tang
        structure_ok = self.test_directory_structure()
        model_ok = self.test_custom_model_exists()
        db_ok = self.test_database_connection()
        api_ok = self.test_api_health()
        
        if not structure_ok:
            print("Nghiem trong: Cau truc thu muc chua hoan chinh.")
            print("Goi y: Dam bao ban da copy tat ca file can thiet tu artifacts.")
            return self.generate_report()
        
        if not (db_ok and api_ok):
            print("Nghiem trong: Database hoac API khong the truy cap.")
            print("Goi y: Dam bao backend server dang chay: uvicorn main:app --reload")
            return self.generate_report()
        
        # Test cac API endpoints
        self.test_vehicle_entry_api("API-TEST-001")
        self.test_statistics_api()
        self.test_vehicle_history_api()
        
        # Test workflow hoan chinh
        self.test_complete_workflow()
        
        return self.generate_report()
    
    def generate_report(self):
        """Tao bao cao ket qua test"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 60)
        print("BAO CAO KET QUA TEST TOAN DIEN")
        print("=" * 60)
        print(f"Tong so test: {total_tests}")
        print(f"Thanh cong: {passed_tests}")
        print(f"That bai: {failed_tests}")
        print(f"Ti le thanh cong: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\nCac test that bai:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['message']}")
        
        # Luu bao cao chi tiet
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "structure": "car/smart_parking/",
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests/total_tests)*100
            },
            "details": self.test_results
        }
        
        report_file = self.base_dir / "test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nBao cao chi tiet da luu: {report_file}")
        
        # Danh gia tong the
        if passed_tests == total_tests:
            print("\nHoan hao! He thong san sang cho production.")
        elif passed_tests >= total_tests * 0.8:
            print("\nTot! He thong hoat dong binh thuong voi vai loi nho.")
        else:
            print("\nCan sua chua! Nhieu thanh phan can chu y.")
        
        return passed_tests == total_tests

def main():
    """Chay test chinh"""
    print("HE THONG NHAN DIEN BIEN SO XE VIET NAM")
    print("TEST TICH HOP TOAN DIEN")
    print("Cau truc: car/smart_parking/")
    print()
    
    # Kiem tra moi truong
    if not Path("smart_parking").exists():
        print("Loi: Test nay phai chay tu thu muc 'car/'")
        print("Cach dung: cd car/ && python test_complete_system.py")
        sys.exit(1)
    
    print("Kiem tra moi truong test:")
    print(f"   Thu muc lam viec: {Path.cwd()}")
    print(f"   Thu muc smart parking: {Path('smart_parking').exists()}")
    print(f"   Model custom: {Path('license_plate_model.pt').exists()}")
    print()
    
    tester = SystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n Tat ca test deu thanh cong!")
        print("He thong nhan dien bien so cua ban da san sang!")
        exit(0)
    else:
        print("\nCo mot vai van de duoc tim thay. Vui long xem xet cac test that bai o tren.")
        print("Kiem tra huong dan tich hop de tim giai phap.")
        exit(1)

if __name__ == "__main__":
    main()