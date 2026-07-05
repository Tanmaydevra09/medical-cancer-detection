import os
import sys

# Fix for XGBoost on macOS without Homebrew:
# Re-launch with DYLD_LIBRARY_PATH pointing at sklearn's bundled libomp.dylib
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
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import xgboost as xgb
import matplotlib.pyplot as plt

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'leukemia_dataset.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'blood_cancer_model.pkl')
    
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if 'Patient_ID' in df.columns:
        df = df.drop(columns=['Patient_ID'])
        
    target_col = 'Leukemia_Status'
    
    # Map target
    y = df[target_col].map({'Positive': 1, 'Negative': 0})
    if y.isnull().any():
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df[target_col]))
        
    X = df.drop(columns=[target_col])
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Class imbalance scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Split data first
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Preprocess the data
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create XGBoost model with early stopping
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        early_stopping_rounds=10,
        verbose=True
    )
    
    print("Training XGBoost model with validation tracking...")
    
    # Train with evaluation sets to capture curves
    eval_set = [(X_train_processed, y_train), (X_val_processed, y_val)]
    model.fit(X_train_processed, y_train, eval_set=eval_set, verbose=False)
    
    # Extract training curves
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    # Create accuracy curves
    train_acc = []
    val_acc = []
    train_loss = results['validation_0']['logloss']
    val_loss = results['validation_1']['logloss']
    
    # Calculate accuracy for each epoch
    for i in range(epochs):
        # Get predictions at this epoch
        train_pred_proba = model.predict_proba(X_train_processed, output_margin=True, iteration_range=(0, i+1))
        val_pred_proba = model.predict_proba(X_val_processed, output_margin=True, iteration_range=(0, i+1))
        
        # Convert to binary predictions
        train_pred = (train_pred_proba[:, 1] > 0.5).astype(int)
        val_pred = (val_pred_proba[:, 1] > 0.5).astype(int)
        
        # Calculate accuracy
        train_acc.append(accuracy_score(y_train, train_pred))
        val_acc.append(accuracy_score(y_val, val_pred))
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    ax1.plot(x_axis, train_acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(x_axis, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title('XGBoost Blood Cancer Model - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.0)
    
    # Loss plot
    ax2.plot(x_axis, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
    ax2.plot(x_axis, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
    ax2.set_title('XGBoost Blood Cancer Model - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Log Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'blood_training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(BASE_DIR, 'blood_training_curves.png')}")
    
    # Create final model pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Fit on full training data
    print("Final training on full dataset...")
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {acc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - XGBoost Blood Cancer Model')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'blood_confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(BASE_DIR, 'blood_confusion_matrix.png')}")
    
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel pipeline saved successfully to {MODEL_PATH}")
    
    # Print final metrics
    print(f"\n=== Final Model Performance ===")
    print(f"Best validation accuracy: {max(val_acc):.4f}")
    print(f"Final validation accuracy: {val_acc[-1]:.4f}")
    print(f"Best validation loss: {min(val_loss):.4f}")
    print(f"Final validation loss: {val_loss[-1]:.4f}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
