# Medical Cancer Detection using Deep Learning

This project implements deep learning models to detect cancer from medical images.
It includes both Brain Tumor Detection and Lung Cancer Detection using CNN models.

The project is intended for educational and academic purposes.

--------------------------------------------------

PROJECT MODULES

1. Brain Tumor Detection
- Binary classification of brain tumor images
- Dataset preparation
- Model training
- Single image prediction

2. Lung Cancer Detection
- Lung cancer classification from CT scan images
- Dataset splitting
- Model training
- Image prediction

--------------------------------------------------

TECHNOLOGIES USED

- Python 3.10
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

--------------------------------------------------

PROJECT STRUCTURE

medical_cancer_detection
|
|-- brain
|   |-- brain_binary
|   |-- brain_segmentation
|   |-- brain_binary_model.keras
|   |-- prepare_brain_binary_dataset.py
|   |-- train_brain_binary_model.py
|   |-- predict_single_image.py
|   |-- img1.jpg
|
|-- lung
|   |-- dataset_split
|   |-- lung_cancer_model.keras
|   |-- train_lung_model.py
|   |-- predict_lung_image.py
|   |-- split_dataset.py
|   |-- img1.jpg
|
|-- requirements.txt
|-- .gitignore
|-- README.md

--------------------------------------------------

INSTALLATION STEPS

1. Clone the repository
git clone https://github.com/Tanmaydevra09/medical_cancer_detection.git

2. Move into the project directory
cd medical_cancer_detection

3. Create virtual environment
python -m venv venv

4. Activate virtual environment
source venv/bin/activate

5. Install dependencies
pip install -r requirements.txt

--------------------------------------------------

TRAINING THE MODELS

Brain model training
python brain/train_brain_binary_model.py

Lung model training
python lung/train_lung_model.py

--------------------------------------------------

PREDICTION

Brain image prediction
python brain/predict_single_image.py

Lung image prediction
python lung/predict_lung_image.py

--------------------------------------------------

DATASET NOTE

The dataset is currently included in the repository.
It may be removed and replaced with external download links in future updates.

--------------------------------------------------

AUTHOR

Tanmay Devra,Sania Tanweer
B.Tech Computer Science Engineering  
GitHub: https://github.com/Tanmaydevra09

--------------------------------------------------

DISCLAIMER

This project is for educational and research purposes only.
It should not be used for real medical diagnosis.

--------------------------------------------------

MYSQL DATABASE SETUP

This project now logs users, uploaded images, predictions, blood records, and model execution logs in MySQL.

1. Create schema manually (optional, backend also auto-creates on startup):
`mysql -u root -p < backend/schema.sql`

2. Configure DB credentials before running backend:

- `DB_HOST` (default: `127.0.0.1`)
- `DB_PORT` (default: `3306`)
- `DB_USER` (default: `root`)
- `DB_PASSWORD` (default: empty)
- `DB_NAME` (default: `medical_cancer_detection`)

3. Start backend:
`python backend/app.py`

4. Verify database connection:
`GET http://127.0.0.1:5000/health/db`
