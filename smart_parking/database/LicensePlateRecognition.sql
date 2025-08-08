-- 1. tao database neu chua co
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'LicensePlateRecognition')
BEGIN
    CREATE DATABASE LicensePlateRecognition;
END
GO

--2.Su dung database
USE LicensePlateRecognition;
GO

-- 3. Tao bang vehicle_logs neu chua co
IF NOT EXISTS (
    SELECT * FROM sys.objects 
    WHERE object_id = OBJECT_ID(N'vehicle_logs') AND type = 'U'
)
BEGIN
    CREATE TABLE vehicle_logs (
        id INT IDENTITY(1,1) PRIMARY KEY,
        license_plate VARCHAR(20) NOT NULL,
        vehicle_type VARCHAR(10) CHECK (vehicle_type IN ('xemay', 'oto')) DEFAULT 'xemay',
        entry_time DATETIME NULL,
        exit_time DATETIME NULL,
        image_path VARCHAR(255) NULL,
        confidence_score FLOAT DEFAULT NULL,
        status VARCHAR(10) CHECK (status IN ('in', 'out', 'unknown')) DEFAULT 'unknown',
        created_at DATETIME DEFAULT GETDATE()
    );
END
GO

-- 4. Tao index tim kiem nhanh
CREATE INDEX idx_license_plate ON vehicle_logs(license_plate);
CREATE INDEX idx_entry_time ON vehicle_logs(entry_time);
CREATE INDEX idx_exit_time ON vehicle_logs(exit_time);
CREATE INDEX idx_status ON vehicle_logs(status);
GO

-- 5. Them du lieu vao mau
INSERT INTO vehicle_logs 
(license_plate, vehicle_type, entry_time, exit_time, image_path, confidence_score, status)
VALUES
('59A-12345', 'oto', DATEADD(HOUR, -2, GETDATE()), DATEADD(HOUR, -1, GETDATE()), '59A-12345_1.jpg', 0.95, 'out'),
('51G-67890', 'oto', DATEADD(HOUR, -3, GETDATE()), DATEADD(HOUR, -2, GETDATE()), '51G-67890_1.jpg', 0.91, 'out'),
('43B-54321', 'xemay', DATEADD(HOUR, -1, GETDATE()), NULL, '43B-54321_1.jpg', 0.88, 'in');
GO

-- 6. Kiem tra 
SELECT * FROM vehicle_logs;
GO
