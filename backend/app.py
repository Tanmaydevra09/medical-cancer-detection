from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import ctypes, os as _os
import json
import subprocess
import tempfile
from pathlib import Path
import time
import gc
import onnxruntime as ort

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson.objectid import ObjectId
import datetime

# Fix for XGBoost on macOS without Homebrew
_libomp = str(
    Path(__file__).resolve().parent.parent
    / "venv_backend"
    / "lib"
    / "python3.10"
    / "site-packages"
    / "sklearn"
    / ".dylibs"
    / "libomp.dylib"
)
# libomp preloading can crash on some macOS Python builds; keep it opt-in.
if _os.getenv("LOAD_SKLEARN_LIBOMP", "0") == "1" and _os.path.exists(_libomp):
    ctypes.CDLL(_libomp)



app = Flask(__name__)
CORS(app)


IMG_SIZE = (224, 224)
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
IMAGE_MODEL_PYTHON = _os.getenv("IMAGE_MODEL_PYTHON") or str(ROOT_DIR / "venv_backend" / "bin" / "python")
LUNG_MODEL_PYTHON = _os.getenv("LUNG_MODEL_PYTHON") or str(ROOT_DIR / "venv_local" / "bin" / "python")
RUNTIME_SCRIPT = str(BACKEND_DIR / "predict_image_runtime.py")
ALLOW_MOCK = _os.getenv("ALLOW_MOCK_PREDICTIONS", "0") == "1"  # Set to "0" to disable mock predictions
# Blood model: must use venv_backend (sklearn 1.7.2 matches model save version)
BLOOD_PYTHON = _os.getenv("BLOOD_PYTHON") or str(ROOT_DIR / "venv_backend" / "bin" / "python")
# libomp path so XGBoost can load on macOS without Homebrew
_SKLEARN_DYLIBS = str(ROOT_DIR / "venv_backend" / "lib" / "python3.10" / "site-packages" / "sklearn" / ".dylibs")
BLOOD_RUNTIME_SCRIPT = str(BACKEND_DIR / "predict_blood_runtime.py")

MONGO_URI = _os.getenv("MONGO_URI", "mongodb://localhost:27017/medical_cancer_detection")



def _mock_binary_prediction(img_array):
    # Disabled: Mock predictions were generating random results
    # Instead, this will raise an error to force proper model usage
    raise RuntimeError("Mock predictions disabled. Please ensure ML models are properly configured.")


def _mock_lung_prediction(img_array):
    # Disabled: Mock predictions were generating random results
    # Instead, this will raise an error to force proper model usage
    raise RuntimeError("Mock predictions disabled. Please ensure ML models are properly configured.")


# Inline CLAHE + preprocessing (avoids fragile cross-module import that breaks in Docker)
def _apply_clahe(image_array):
    import cv2
    try:
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.uint8)
    except Exception:
        return image_array.astype(np.uint8)


def _preprocess_for_model(pil_image):
    arr = np.array(pil_image).astype(np.uint8)
    arr = _apply_clahe(arr)
    arr = arr.astype(np.float32)  # keep [0,255] — EfficientNet rescales internally
    return np.expand_dims(arr, axis=0)  # (1,224,224,3)


_LUNG_CLASSES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma", "Other"]

# ---- ONNX model cache: ~30MB RAM per session vs ~300MB for TF ----
_ort_cache: dict = {}  # task -> ort.InferenceSession

def _get_or_load_ort_session(task: str) -> ort.InferenceSession:
    """Load ONNX model once and cache. Uses ~30MB RAM vs TF's ~300MB."""
    if task not in _ort_cache:
        onnx_paths = {
            "brain":  str(ROOT_DIR / "brain"  / "brain.onnx"),
            "lung":   str(ROOT_DIR / "lung"   / "lung.onnx"),
            "breast": str(ROOT_DIR / "breast" / "breast.onnx"),
        }
        path = onnx_paths[task]
        if not _os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path}. ROOT_DIR={ROOT_DIR}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        _ort_cache[task] = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return _ort_cache[task]


def _run_image_model_subprocess(task, image_bytes):
    """Run inference via ONNX Runtime — uses ~30MB RAM, safe for Render free tier."""
    # Load and preprocess image
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = _preprocess_for_model(image)  # shape (1,224,224,3), float32 [0,255]

    session = _get_or_load_ort_session(task)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    if task == "brain" or task == "breast":
        pred = float(outputs[0][0][0])
        if pred >= 0.5:
            return {"prediction": "Cancer", "confidence": round(pred * 100, 2)}
        return {"prediction": "No Cancer", "confidence": round((1 - pred) * 100, 2)}
    elif task == "lung":
        preds = outputs[0][0]
        idx = int(np.argmax(preds))
        return {"prediction": _LUNG_CLASSES[idx], "confidence": round(float(preds[idx] * 100), 2)}
    else:
        raise ValueError(f"Unknown task: {task}")


def _run_blood_model_subprocess(payload):
    model_path = str(ROOT_DIR / "blood" / "blood_cancer_model.pkl")
    
    # Check if model file exists
    if not _os.path.exists(model_path):
        raise FileNotFoundError(f"Blood model file not found: {model_path}")
    
    # Build subprocess env with DYLD so XGBoost finds libomp on macOS
    sub_env = _os.environ.copy()
    existing_dyld = sub_env.get("DYLD_LIBRARY_PATH", "")
    sub_env["DYLD_LIBRARY_PATH"] = (
        f"{_SKLEARN_DYLIBS}:{existing_dyld}" if existing_dyld else _SKLEARN_DYLIBS
    )
    sub_env["_BLOOD_LIBOMP_FIXED"] = "1"  # prevent double-relaunch in runtime

    try:
        proc = subprocess.run(
            [
                BLOOD_PYTHON,
                BLOOD_RUNTIME_SCRIPT,
                model_path,
                json.dumps(payload),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(BACKEND_DIR),
            env=sub_env,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("Empty output from blood model")
        
        result = json.loads(lines[-1])
        
        # Validate result structure
        if "prediction" not in result or "confidence" not in result:
            raise RuntimeError(f"Invalid result structure from blood model: {result}")
            
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Blood model subprocess failed: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON output from blood model: {e}")

lung_classes = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma",
    "Other"
]

breast_classes = ["No Cancer", "Cancer"]

def _get_db_connection():
    client = MongoClient(MONGO_URI)
    db = client.get_database("medical_cancer_detection")
    return client, db


def init_database():
    client, db = _get_db_connection()
    client.admin.command('ping')
    client.close()


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_or_create_user(name, email, role):
    client, db = _get_db_connection()
    try:
        user = db.User.find_one({"email": email})
        if user:
            db.User.update_one(
                {"_id": user["_id"]},
                {"$set": {"name": name, "role": role}}
            )
            return str(user["_id"])
        
        result = db.User.insert_one({
            "name": name,
            "email": email,
            "role": role
        })
        return str(result.inserted_id)
    finally:
        client.close()


def _log_image_prediction(user_id, image_path, cancer_type, result, confidence, model_used, execution_time):
    client, db = _get_db_connection()
    try:
        image_result = db.Image.insert_one({
            "user_id": ObjectId(user_id) if len(str(user_id)) == 24 else user_id,
            "image_path": image_path,
            "cancer_type": cancer_type,
            "upload_date": datetime.datetime.utcnow()
        })
        image_id = str(image_result.inserted_id)
        
        prediction_result = db.Prediction.insert_one({
            "image_id": image_result.inserted_id,
            "result": result,
            "confidence": confidence,
            "model_used": model_used,
            "prediction_date": datetime.datetime.utcnow()
        })
        prediction_id = str(prediction_result.inserted_id)
        
        log_result = db.Model_Log.insert_one({
            "model_name": model_used,
            "accuracy": confidence,
            "execution_time": execution_time,
            "date": datetime.datetime.utcnow()
        })
        log_id = str(log_result.inserted_id)
        
        return image_id, prediction_id, log_id
    finally:
        client.close()


def _log_blood_prediction(user_id, payload, result, confidence, execution_time):
    client, db = _get_db_connection()
    try:
        record_result = db.Blood_Data.insert_one({
            "user_id": ObjectId(user_id) if len(str(user_id)) == 24 else user_id,
            "Age": _safe_float(payload.get("Age")),
            "Gender": payload.get("Gender"),
            "Country": payload.get("Country"),
            "WBC_Count": _safe_float(payload.get("WBC_Count")),
            "RBC_Count": _safe_float(payload.get("RBC_Count")),
            "Platelet_Count": _safe_float(payload.get("Platelet_Count")),
            "Hemoglobin_Level": _safe_float(payload.get("Hemoglobin_Level")),
            "Bone_Marrow_Blasts": _safe_float(payload.get("Bone_Marrow_Blasts")),
            "BMI": _safe_float(payload.get("BMI")),
            "Genetic_Mutation": payload.get("Genetic_Mutation"),
            "Family_History": payload.get("Family_History"),
            "Smoking_Status": payload.get("Smoking_Status"),
            "Alcohol_Consumption": payload.get("Alcohol_Consumption"),
            "Radiation_Exposure": payload.get("Radiation_Exposure"),
            "Infection_History": payload.get("Infection_History"),
            "Chronic_Illness": payload.get("Chronic_Illness"),
            "Immune_Disorders": payload.get("Immune_Disorders"),
            "result": result,
            "confidence": confidence,
            "date": datetime.datetime.utcnow()
        })
        record_id = str(record_result.inserted_id)
        
        log_result = db.Model_Log.insert_one({
            "model_name": "XGBoost",
            "accuracy": confidence,
            "execution_time": execution_time,
            "date": datetime.datetime.utcnow()
        })
        log_id = str(log_result.inserted_id)
        
        return record_id, log_id
    finally:
        client.close()


def _user_from_form_data():
    name = request.form.get("user_name", "Guest User")
    email = request.form.get("user_email", "guest@example.com")
    role = request.form.get("user_role", "patient")
    return name, email, role

# ===================== BRAIN API =====================
@app.route("/predict/brain", methods=["POST"])
def predict_brain():
    start = time.perf_counter()
    file = request.files["image"]
    file_bytes = file.read()
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        model_output = _run_image_model_subprocess("brain", file_bytes)
        model_output["mode"] = "model"
    except Exception as err:
        if ALLOW_MOCK:
            prediction, confidence = _mock_binary_prediction(img_array)
            model_output = {"prediction": prediction, "confidence": confidence, "mode": "mock"}
        else:
            return jsonify({"error": f"Brain model failed: {err}"}), 500

    db_error = None
    try:
        user_id = _get_or_create_user(*_user_from_form_data())
        image_id, prediction_id, log_id = _log_image_prediction(
            user_id=user_id,
            image_path=file.filename or "uploaded_brain_image.png",
            cancer_type="brain",
            result=model_output.get("prediction", "Unknown"),
            confidence=float(model_output.get("confidence", 0)),
            model_used="EfficientNet",
            execution_time=round(time.perf_counter() - start, 4),
        )
        model_output.update({
            "user_id": user_id,
            "image_id": image_id,
            "prediction_id": prediction_id,
            "log_id": log_id,
        })
    except PyMongoError as err:
        db_error = str(err)

    if db_error:
        model_output["db_error"] = db_error
    return jsonify(model_output)

# ===================== LUNG API =====================
@app.route("/predict/lung", methods=["POST"])
def predict_lung():
    start = time.perf_counter()
    file = request.files["image"]
    file_bytes = file.read()
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        model_output = _run_image_model_subprocess("lung", file_bytes)
        model_output["mode"] = "model"
    except Exception as err:
        if ALLOW_MOCK:
            prediction, confidence = _mock_lung_prediction(img_array)
            model_output = {"prediction": prediction, "confidence": confidence, "mode": "mock"}
        else:
            return jsonify({"error": f"Lung model failed: {err}"}), 500

    db_error = None
    try:
        user_id = _get_or_create_user(*_user_from_form_data())
        image_id, prediction_id, log_id = _log_image_prediction(
            user_id=user_id,
            image_path=file.filename or "uploaded_lung_image.png",
            cancer_type="lung",
            result=model_output.get("prediction", "Unknown"),
            confidence=float(model_output.get("confidence", 0)),
            model_used="EfficientNet",
            execution_time=round(time.perf_counter() - start, 4),
        )
        model_output.update({
            "user_id": user_id,
            "image_id": image_id,
            "prediction_id": prediction_id,
            "log_id": log_id,
        })
    except PyMongoError as err:
        db_error = str(err)

    if db_error:
        model_output["db_error"] = db_error
    return jsonify(model_output)

# ===================== BLOOD API =====================
@app.route("/predict/blood", methods=["POST"])
def predict_blood():
    data = request.get_json()
    try:
        start = time.perf_counter()
        model_output = _run_blood_model_subprocess(data)
        model_output["mode"] = "model"
        db_error = None
        try:
            user_id = _get_or_create_user(
                data.get("user_name", "Guest User"),
                data.get("user_email", "guest@example.com"),
                data.get("user_role", "patient"),
            )
            record_id, log_id = _log_blood_prediction(
                user_id=user_id,
                payload=data,
                result=model_output.get("prediction", "Unknown"),
                confidence=float(model_output.get("confidence", 0)),
                execution_time=round(time.perf_counter() - start, 4),
            )
            model_output.update({
                "user_id": user_id,
                "record_id": record_id,
                "log_id": log_id,
            })
        except PyMongoError as err:
            db_error = str(err)
        if db_error:
            model_output["db_error"] = db_error
        return jsonify(model_output)
    except Exception:
        return jsonify({"prediction": "Error", "confidence": 0}), 500

# ===================== BREAST API =====================
@app.route("/predict/breast", methods=["POST"])
def predict_breast():
    start = time.perf_counter()
    file = request.files["image"]
    file_bytes = file.read()
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0  # Normalize to 0-1 range
    img_array = np.expand_dims(img_array, axis=0)

    try:
        model_output = _run_image_model_subprocess("breast", file_bytes)
        model_output["mode"] = "model"
    except Exception as err:
        if ALLOW_MOCK:
            prediction, confidence = _mock_binary_prediction(img_array)
            model_output = {"prediction": prediction, "confidence": confidence, "mode": "mock"}
        else:
            return jsonify({"error": f"Breast model failed: {err}"}), 500

    db_error = None
    try:
        user_id = _get_or_create_user(*_user_from_form_data())
        image_id, prediction_id, log_id = _log_image_prediction(
            user_id=user_id,
            image_path=file.filename or "uploaded_breast_image.png",
            cancer_type="breast",
            result=model_output.get("prediction", "Unknown"),
            confidence=float(model_output.get("confidence", 0)),
            model_used="EfficientNet",
            execution_time=round(time.perf_counter() - start, 4),
        )
        model_output.update({
            "user_id": user_id,
            "image_id": image_id,
            "prediction_id": prediction_id,
            "log_id": log_id,
        })
    except PyMongoError as err:
        db_error = str(err)

    if db_error:
        model_output["db_error"] = db_error
    return jsonify(model_output)


@app.route("/health/db", methods=["GET"])
def db_health():
    try:
        client, db = _get_db_connection()
        client.admin.command('ping')
        client.close()
        return jsonify({"ok": True, "database": "mongodb"})
    except PyMongoError as err:
        return jsonify({"ok": False, "error": str(err)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "running", "version": "1.0"})

if __name__ == "__main__":
    try:
        init_database()
        print("Database initialized.")
    except Exception as err:
        print(f"Database init warning: {err}")
    app.run(debug=False, use_reloader=False)

# //http://127.0.0.1:5000 by default 