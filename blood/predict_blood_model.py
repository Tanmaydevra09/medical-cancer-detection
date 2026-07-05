import os
import sys

# Fix for XGBoost on macOS without Homebrew
_LIBOMP_DIR = (
    "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/"
    "site-packages/sklearn/.dylibs"
)
if os.path.isdir(_LIBOMP_DIR) and "_XGBOOST_FIXED" not in os.environ:
    import subprocess
    env = os.environ.copy()
    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{_LIBOMP_DIR}:{existing}" if existing else _LIBOMP_DIR
    env["_XGBOOST_FIXED"] = "1"
    result = subprocess.run([sys.executable] + sys.argv, env=env)
    sys.exit(result.returncode)


import pandas as pd
import joblib

def predict_leukemia(patient_data):
    """
    Predicts leukemia status based on patient data.
    
    Args:
        patient_data (dict or pd.DataFrame): The patient's data. 
            Keys should match the dataset columns (excluding Patient_ID and Leukemia_Status).
            
    Returns:
        dict: A dictionary containing the prediction ('Positive' or 'Negative') 
              and the probability of being positive.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'blood_cancer_model.pkl')
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run the training script first.")
        
    model_pipeline = joblib.load(MODEL_PATH)
    
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data
        
    prediction_num = model_pipeline.predict(df)[0]
    probability = model_pipeline.predict_proba(df)[0][1]
    
    prediction_label = 'Positive' if prediction_num == 1 else 'Negative'
    
    return {
        'prediction': prediction_label,
        'probability_positive': float(probability)
    }

if __name__ == '__main__':
    # Example usage with a sample patient
    sample_patient = {
        'Age': 52,
        'Gender': 'Male',
        'Country': 'China',
        'WBC_Count': 2698,
        'RBC_Count': 5.36,
        'Platelet_Count': 262493,
        'Hemoglobin_Level': 12.2,
        'Bone_Marrow_Blasts': 72,
        'Genetic_Mutation': 'Yes',
        'Family_History': 'No',
        'Smoking_Status': 'Yes',
        'Alcohol_Consumption': 'No',
        'Radiation_Exposure': 'No',
        'Infection_History': 'No',
        'BMI': 24.0,
        'Chronic_Illness': 'No',
        'Immune_Disorders': 'No',
        'Ethnicity': 'Ethnic_Group_B',
        'Socioeconomic_Status': 'Low',
        'Urban_Rural': 'Rural'
    }
    
    print("Running prediction for sample patient:")
    # Print the sample beautifully
    for k, v in sample_patient.items():
        print(f"{k.ljust(25)}: {v}")
        
    print("\n--- Model Inference ---")
    try:
        result = predict_leukemia(sample_patient)
        
        print("\n=== Prediction Results ===")
        print(f"Status: {result['prediction']}")
        print(f"Probability (Positive): {result['probability_positive']:.2%}")
    except FileNotFoundError as e:
        print(str(e))
