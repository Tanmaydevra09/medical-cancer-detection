CREATE DATABASE IF NOT EXISTS `medical_cancer_detection`;
USE `medical_cancer_detection`;

CREATE TABLE IF NOT EXISTS `User` (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    role VARCHAR(20) NOT NULL
);

CREATE TABLE IF NOT EXISTS `Image` (
    image_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    cancer_type VARCHAR(50) NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES `User`(user_id)
);

CREATE TABLE IF NOT EXISTS `Prediction` (
    prediction_id INT PRIMARY KEY AUTO_INCREMENT,
    image_id INT NOT NULL,
    result VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES `Image`(image_id)
);

CREATE TABLE IF NOT EXISTS `Blood_Data` (
    record_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    feature1 FLOAT,
    feature2 FLOAT,
    feature3 FLOAT,
    result VARCHAR(20),
    FOREIGN KEY (user_id) REFERENCES `User`(user_id)
);

CREATE TABLE IF NOT EXISTS `Model_Log` (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    model_name VARCHAR(50) NOT NULL,
    accuracy FLOAT,
    execution_time FLOAT,
    date DATETIME DEFAULT CURRENT_TIMESTAMP
);
