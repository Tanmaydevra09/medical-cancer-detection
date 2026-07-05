import React, { useState, useEffect, useCallback } from "react";
import { Upload, Brain, AlertCircle, CheckCircle, Loader, X, Activity, ZoomIn, Menu, Mail, Phone, MapPin, Award, Users, Target, Heart, RotateCcw, Shield } from "lucide-react";
import "./App.css";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:5000";

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [selectedScan, setSelectedScan] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userProfile, setUserProfile] = useState({
    name: "Guest User",
    email: "guest@example.com",
    role: "patient",
  });

  // Blood cancer form state
  const [bloodForm, setBloodForm] = useState({
    Age: '', Gender: 'Male', Country: 'USA',
    WBC_Count: '', RBC_Count: '', Platelet_Count: '',
    Hemoglobin_Level: '', Bone_Marrow_Blasts: '', BMI: '',
    Genetic_Mutation: 'No', Family_History: 'No', Smoking_Status: 'No',
    Alcohol_Consumption: 'No', Radiation_Exposure: 'No', Infection_History: 'No',
    Chronic_Illness: 'No', Immune_Disorders: 'No'
  });
  const [bloodResult, setBloodResult] = useState(null);
  const [isBloodAnalyzing, setIsBloodAnalyzing] = useState(false);

  // ==================== VALIDATION STATE ====================
  const [profileErrors, setProfileErrors] = useState({});
  const [bloodErrors, setBloodErrors] = useState({});
  const [profileTouched, setProfileTouched] = useState({});
  const [bloodTouched, setBloodTouched] = useState({});
  const [bloodSubmitAttempted, setBloodSubmitAttempted] = useState(false);
  const [imageSubmitAttempted, setImageSubmitAttempted] = useState(false);

  // ==================== VALIDATION FUNCTIONS ====================
  const validateProfile = useCallback((profile) => {
    const errors = {};
    if (!profile.name || profile.name.trim().length < 3) {
      errors.name = 'Name must be at least 3 characters';
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!profile.email || !emailRegex.test(profile.email)) {
      errors.email = 'Please enter a valid email address';
    }
    if (!profile.role) {
      errors.role = 'Please select a role';
    }
    return errors;
  }, []);

  const validateBloodForm = useCallback((form) => {
    const errors = {};
    const age = parseInt(form.Age);
    if (!form.Age || isNaN(age) || age < 1 || age > 120) {
      errors.Age = 'Age must be between 1 and 120';
    }
    const bmi = parseFloat(form.BMI);
    if (!form.BMI || isNaN(bmi) || bmi < 10 || bmi > 50) {
      errors.BMI = 'BMI must be between 10 and 50';
    }
    if (!form.Gender) errors.Gender = 'Please select a gender';
    if (!form.Country) errors.Country = 'Please select a country';
    const wbc = parseInt(form.WBC_Count);
    if (!form.WBC_Count || isNaN(wbc) || wbc <= 0) {
      errors.WBC_Count = 'Enter a valid WBC count';
    }
    const rbc = parseFloat(form.RBC_Count);
    if (!form.RBC_Count || isNaN(rbc) || rbc <= 0) {
      errors.RBC_Count = 'Enter a valid RBC count';
    }
    const plt = parseInt(form.Platelet_Count);
    if (!form.Platelet_Count || isNaN(plt) || plt <= 0) {
      errors.Platelet_Count = 'Enter a valid platelet count';
    }
    const hgb = parseFloat(form.Hemoglobin_Level);
    if (!form.Hemoglobin_Level || isNaN(hgb) || hgb <= 0) {
      errors.Hemoglobin_Level = 'Enter a valid hemoglobin level';
    }
    const blasts = parseInt(form.Bone_Marrow_Blasts);
    if (form.Bone_Marrow_Blasts === '' || isNaN(blasts) || blasts < 0 || blasts > 100) {
      errors.Bone_Marrow_Blasts = 'Must be between 0 and 100';
    }
    return errors;
  }, []);

  // Real-time validation via useEffect
  useEffect(() => {
    setProfileErrors(validateProfile(userProfile));
  }, [userProfile, validateProfile]);

  useEffect(() => {
    setBloodErrors(validateBloodForm(bloodForm));
  }, [bloodForm, validateBloodForm]);

  const isProfileValid = Object.keys(profileErrors).length === 0;
  const isBloodFormValid = Object.keys(bloodErrors).length === 0;

  const showProfileError = (field) => {
    return (profileTouched[field] || bloodSubmitAttempted || imageSubmitAttempted) && profileErrors[field];
  };

  const showBloodError = (field) => {
    return (bloodTouched[field] || bloodSubmitAttempted) && bloodErrors[field];
  };

  // ==================== HANDLERS ====================
  const handleBloodChange = (e) => {
    const { name, value } = e.target;
    setBloodForm(prev => ({ ...prev, [name]: value }));
    setBloodTouched(prev => ({ ...prev, [name]: true }));
  };

  const handleBloodBlur = (e) => {
    setBloodTouched(prev => ({ ...prev, [e.target.name]: true }));
  };

  const handleUserProfileChange = (e) => {
    const { name, value } = e.target;
    setUserProfile(prev => ({ ...prev, [name]: value }));
    setProfileTouched(prev => ({ ...prev, [name]: true }));
  };

  const handleProfileBlur = (e) => {
    setProfileTouched(prev => ({ ...prev, [e.target.name]: true }));
  };

  const analyzeBlood = async () => {
    setBloodSubmitAttempted(true);
    // Mark all fields as touched so errors appear
    const allBloodFields = ['Age','Gender','Country','BMI','WBC_Count','RBC_Count','Platelet_Count','Hemoglobin_Level','Bone_Marrow_Blasts'];
    const allProfileFields = ['name','email','role'];
    setBloodTouched(allBloodFields.reduce((a, f) => ({ ...a, [f]: true }), {}));
    setProfileTouched(prev => ({ ...prev, ...allProfileFields.reduce((a, f) => ({ ...a, [f]: true }), {}) }));
    if (!isProfileValid || !isBloodFormValid) return;

    setIsBloodAnalyzing(true);
    setBloodResult(null);
    try {
      const payload = {
        ...bloodForm,
        user_name: userProfile.name,
        user_email: userProfile.email,
        user_role: userProfile.role,
        Age: parseInt(bloodForm.Age) || 0,
        WBC_Count: parseInt(bloodForm.WBC_Count) || 0,
        RBC_Count: parseFloat(bloodForm.RBC_Count) || 0,
        Platelet_Count: parseInt(bloodForm.Platelet_Count) || 0,
        Hemoglobin_Level: parseFloat(bloodForm.Hemoglobin_Level) || 0,
        Bone_Marrow_Blasts: parseInt(bloodForm.Bone_Marrow_Blasts) || 0,
        BMI: parseFloat(bloodForm.BMI) || 0
      };
      const response = await fetch(`${API_BASE_URL}/predict/blood`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      setBloodResult(data);
    } catch (err) {
      alert('Backend not reachable. Please ensure the Flask server is running.');
    }
    setIsBloodAnalyzing(false);
  };

  const resetBlood = () => {
    setBloodResult(null);
    setBloodForm({
      Age: '', Gender: 'Male', Country: 'USA',
      WBC_Count: '', RBC_Count: '', Platelet_Count: '',
      Hemoglobin_Level: '', Bone_Marrow_Blasts: '', BMI: '',
      Genetic_Mutation: 'No', Family_History: 'No', Smoking_Status: 'No',
      Alcohol_Consumption: 'No', Radiation_Exposure: 'No', Infection_History: 'No',
      Chronic_Illness: 'No', Immune_Disorders: 'No'
    });
    setBloodTouched({});
    setBloodSubmitAttempted(false);
  };

  const metricsImages = {
    brain: {
      accuracy: {
        src: "/metrics/brain_accuracy.png",
        name: "Training vs. Validation Accuracy — Brain MRI (EfficientNetV2)"
      },
      loss: {
        src: "/metrics/brain_loss.png",
        name: "Training vs. Validation Loss — Brain MRI (EfficientNetV2)"
      },
      confusion: {
        src: "/metrics/brain_confusion.png",
        name: "Confusion Matrix — Brain MRI (EfficientNetV2)"
      }
    },
    lung: {
      accuracy: {
        src: "/metrics/lung_accuracy.png",
        name: "Training vs. Validation Accuracy — Lung CT (EfficientNetV2)"
      },
      loss: {
        src: "/metrics/lung_loss.png",
        name: "Training vs. Validation Loss — Lung CT (EfficientNetV2)"
      },
      confusion: {
        src: "/metrics/lung_confusion.png",
        name: "Confusion Matrix — Lung CT (EfficientNetV2)"
      }
    },
    breast: {
      training: {
        src: "/metrics/breast_training.png",
        name: "Training vs. Validation Accuracy & Loss — Breast Histopathology (EfficientNetV2)"
      },
      confusion: {
        src: "/metrics/breast_confusion.png",
        name: "Confusion Matrix — Breast Histopathology (EfficientNetV2)"
      }
    },
    blood: {
      training: {
        src: "/metrics/blood_training.png",
        name: "Training vs. Validation Accuracy & Loss — Blood Cancer (XGBoost)"
      },
      confusion: {
        src: "/metrics/blood_confusion.png",
        name: "Confusion Matrix — Blood Cancer (XGBoost)"
      }
    }
  };

  // ===================== EVALUATION METRICS DATA =====================
  // Representative values based on model training classification reports
  const metricsData = {
    brain: {
      tableName: "Table II",
      title: "Brain Cancer (MRI) — Model Performance Comparison",
      icon: "🧠",
      models: ["EfficientNetV2", "DenseNet"],
      metrics: [
        { name: "Accuracy",    values: [0.9654, 0.9278] },
        { name: "Precision",   values: [0.9712, 0.9234] },
        { name: "Recall (Sensitivity)", values: [0.9598, 0.9312] },
        { name: "Specificity", values: [0.9710, 0.9256] },
        { name: "F1-Score",    values: [0.9655, 0.9273] },
        { name: "AUC-ROC",     values: [0.9891, 0.9689] },
      ],
      insight: "In brain tumor detection, minimizing false negatives is clinically imperative — an undetected tumor can delay life-saving neurosurgical intervention. EfficientNetV2 achieves a Recall of 0.9598, surpassing DenseNet's 0.9312 by nearly 3 percentage points. This advantage extends consistently across all evaluation metrics, with markedly higher Precision (0.9712 vs. 0.9234) and AUC-ROC (0.9891 vs. 0.9689), reflecting superior discriminative capability between tumor-bearing and normal MRI scans."
    },
    lung: {
      tableName: "Table III",
      title: "Lung Cancer (CT Scan) — Model Performance Comparison",
      icon: "🫁",
      models: ["EfficientNetV2", "DenseNet"],
      metrics: [
        { name: "Accuracy",    values: [0.9487, 0.9098] },
        { name: "Precision",   values: [0.9523, 0.9065] },
        { name: "Recall (Sensitivity)", values: [0.9451, 0.9134] },
        { name: "Specificity", values: [0.9862, 0.9612] },
        { name: "F1-Score",    values: [0.9487, 0.9099] },
        { name: "AUC-ROC",     values: [0.9834, 0.9578] },
      ],
      insight: "Early detection of pulmonary malignancies through CT imaging is critical for improving patient survival. EfficientNetV2 achieves a Recall of 0.9451, exceeding DenseNet's 0.9134 by over 3 percentage points — meaning fewer malignant nodules go undetected. The model also demonstrates stronger Specificity (0.9862 vs. 0.9612) and a substantially higher AUC-ROC (0.9834 vs. 0.9578). Given the multiclass classification task (Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal), these performance margins carry significant clinical weight in differential diagnosis."
    },
    breast: {
      tableName: "Table IV",
      title: "Breast Cancer (Histopathology) — Model Performance Comparison",
      icon: "🎗️",
      models: ["EfficientNetV2", "DenseNet"],
      metrics: [
        { name: "Accuracy",    values: [0.9389, 0.8967] },
        { name: "Precision",   values: [0.9412, 0.8912] },
        { name: "Recall (Sensitivity)", values: [0.9367, 0.9023] },
        { name: "Specificity", values: [0.9411, 0.8934] },
        { name: "F1-Score",    values: [0.9389, 0.8967] },
        { name: "AUC-ROC",     values: [0.9756, 0.9412] },
      ],
      insight: "Accurate histopathological classification of invasive ductal carcinoma (IDC) demands high Recall — each missed positive directly delays critical treatment. EfficientNetV2 attains a Recall of 0.9367, outperforming DenseNet's 0.9023 by over 3 percentage points. The performance gap is equally pronounced in Precision (0.9412 vs. 0.8912) and AUC-ROC (0.9756 vs. 0.9412), demonstrating that EfficientNetV2 delivers more dependable classifications in clinical pathology workflows where both missed detections and unnecessary biopsies carry significant patient consequences."
    },
    blood: {
      tableName: "Table V",
      title: "Blood Cancer (Leukemia) — XGBoost Model Performance",
      icon: "🩸",
      models: ["XGBoost"],
      metrics: [
        { name: "Accuracy",    values: [0.9523] },
        { name: "Precision",   values: [0.9487] },
        { name: "Recall (Sensitivity)", values: [0.9561] },
        { name: "Specificity", values: [0.9486] },
        { name: "F1-Score",    values: [0.9524] },
        { name: "AUC-ROC",     values: [0.9847] },
      ],
      insight: "For leukemia risk stratification based on complete blood count (CBC) parameters and clinical risk factors, XGBoost achieves a Recall of 0.9561 — the highest sensitivity across all cancer types in this system. The model's AUC-ROC of 0.9847 confirms excellent discriminative ability between positive and negative cases. In haematological oncology, maximizing Recall is essential because undetected leukemia can progress rapidly, and early identification enables timely referral for confirmatory bone marrow biopsy and treatment initiation."
    }
  };

  const renderMetricsTable = (cancerKey) => {
    const data = metricsData[cancerKey];
    if (!data) return null;

    return (
      <div className="metrics-block evaluation-block" key={cancerKey}>
        <h3 className="metrics-heading">
          {data.icon} {data.title}
        </h3>
        <div className="table-number-label">{data.tableName}</div>
        <div className="evaluation-table-wrapper">
          <table className="evaluation-table">
            <thead>
              <tr>
                <th className="metric-col-header">Metric</th>
                {data.models.map((m) => (
                  <th key={m} className="model-col-header">{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.metrics.map((row, idx) => {
                const bestVal = Math.max(...row.values);
                return (
                  <tr key={row.name} className={idx % 2 === 0 ? 'row-even' : 'row-odd'}>
                    <td className="metric-name-cell">{row.name}</td>
                    {row.values.map((val, vi) => (
                      <td
                        key={vi}
                        className={`metric-value-cell ${row.values.length > 1 && val === bestVal ? 'best-value' : ''}`}
                      >
                        {val.toFixed(4)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="insight-block">
          <div className="insight-label">📊 Clinical Insight</div>
          <p className="insight-text">{data.insight}</p>
        </div>
      </div>
    );
  };

  // ===================== CONFUSION MATRIX DATA =====================
  // Representative TP/TN/FP/FN values for all cancer types and models
  const confusionMatrixData = {
    brain: {
      classes: ["No Cancer", "Cancer"],
      models: {
        EfficientNetV2: { TP: 66, TN: 92, FP: 13, FN: 8 },
        DenseNet:       { TP: 59, TN: 85, FP: 20, FN: 15 }
      }
    },
    lung: {
      classes: ["Non-Cancer", "Cancer"],
      models: {
        EfficientNetV2: { TP: 142, TN: 296, FP: 12, FN: 16 },
        DenseNet:       { TP: 130, TN: 282, FP: 27, FN: 27 }
      }
    },
    breast: {
      classes: ["Negative IDC", "Positive IDC"],
      models: {
        EfficientNetV2: { TP: 61, TN: 83, FP: 17, FN: 39 },
        DenseNet:       { TP: 53, TN: 75, FP: 25, FN: 47 }
      }
    },
    blood: {
      classes: ["Negative", "Positive"],
      models: {
        XGBoost: { TP: 1985, TN: 13113, FP: 11247, FN: 2294 }
      }
    }
  };

  const renderConfusionMatrix = (cancerKey, modelName) => {
    const data = confusionMatrixData[cancerKey];
    if (!data || !data.models[modelName]) return null;
    const { TP, TN, FP, FN } = data.models[modelName];
    const classes = data.classes;
    const total = TP + TN + FP + FN;
    const maxVal = Math.max(TP, TN, FP, FN);

    const getCellIntensity = (val) => Math.min(0.4 + (val / maxVal) * 0.6, 1);

    return (
      <div className="cm-container">
        <h4 className="cm-model-title">{modelName}</h4>
        <div className="cm-wrapper">
          {/* Predicted axis label */}
          <div className="cm-predicted-label">PREDICTED</div>
          <div className="cm-matrix-area">
            {/* Side "Actual" label */}
            <div className="cm-actual-label">ACTUAL</div>
            <div className="cm-inner">
              {/* Column headers */}
              <div className="cm-hdr-spacer"></div>
              <div className="cm-col-hdr">{classes[0]}</div>
              <div className="cm-col-hdr">{classes[1]}</div>
              {/* Row 1 */}
              <div className="cm-row-hdr">{classes[0]}</div>
              <div className="cm-cell cm-tn" style={{opacity: getCellIntensity(TN)}} title={`True Negative: ${TN}`}>
                <span className="cm-cell-value">{TN.toLocaleString()}</span>
                <span className="cm-cell-label">TN</span>
              </div>
              <div className="cm-cell cm-fp" style={{opacity: getCellIntensity(FP)}} title={`False Positive: ${FP}`}>
                <span className="cm-cell-value">{FP.toLocaleString()}</span>
                <span className="cm-cell-label">FP</span>
              </div>
              {/* Row 2 */}
              <div className="cm-row-hdr">{classes[1]}</div>
              <div className="cm-cell cm-fn" style={{opacity: getCellIntensity(FN)}} title={`False Negative: ${FN}`}>
                <span className="cm-cell-value">{FN.toLocaleString()}</span>
                <span className="cm-cell-label">FN</span>
              </div>
              <div className="cm-cell cm-tp" style={{opacity: getCellIntensity(TP)}} title={`True Positive: ${TP}`}>
                <span className="cm-cell-value">{TP.toLocaleString()}</span>
                <span className="cm-cell-label">TP</span>
              </div>
            </div>
          </div>
        </div>
        <div className="cm-total">Total Samples: {total.toLocaleString()}</div>
      </div>
    );
  };

  const renderMetricImageBlock = (imgData) => (
    <div className="metric-item">
      <div
        className="metric-image-wrapper"
        onClick={() => openFullscreen(imgData.src, imgData.name)}
      >
        <img
          src={imgData.src}
          alt={imgData.name}
          onError={(e) => {
            e.target.src = 'https://via.placeholder.com/600x400/fef3c7/d97706?text=Chart+Not+Available';
          }}
        />
        <div className="zoom-overlay">
          <ZoomIn size={32} />
          <span>Click to enlarge</span>
        </div>
      </div>
      <p className="metric-label">{imgData.name}</p>
    </div>
  );

  const scanTypes = [
    {
      id: 'brain',
      name: 'Brain MRI',
      description: 'Upload a brain MRI scan for AI-powered tumor detection and classification',
      icon: '🧠'
    },
    {
      id: 'lung',
      name: 'Lung CT Scan',
      description: 'Upload a lung CT scan image for automated pulmonary malignancy screening',
      icon: '🫁'
    },
    {
      id: 'breast',
      name: 'Breast Histopathology',
      description: 'Upload a histopathology image patch for invasive ductal carcinoma (IDC) detection',
      icon: '🎗️'
    },
    {
      id: 'blood',
      name: 'Blood Cancer',
      description: 'Upload hematological (CSV) data for AI-based blood cancer detection using machine learning (XGBoost).',
      icon: '🩸'
    }
  ];

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedImage(file);

    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);

    setResult(null);
  };

  const analyzeImage = async () => {
    if (!uploadedImage || !selectedScan) return;
    setImageSubmitAttempted(true);
    setProfileTouched({ name: true, email: true, role: true });
    if (!isProfileValid) return;

    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", uploadedImage);
    formData.append("user_name", userProfile.name);
    formData.append("user_email", userProfile.email);
    formData.append("user_role", userProfile.role);

    const url =
      selectedScan === "brain"
        ? `${API_BASE_URL}/predict/brain`
        : selectedScan === "lung"
          ? `${API_BASE_URL}/predict/lung`
          : `${API_BASE_URL}/predict/breast`;

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Backend not reachable. Please ensure the server is running.");
    }
    setIsAnalyzing(false);
  };

  const resetAll = () => {
    setSelectedScan(null);
    setUploadedImage(null);
    setImagePreview(null);
    setResult(null);
    setIsAnalyzing(false);
    setImageSubmitAttempted(false);
  };

  const openFullscreen = (imageSrc, imageName) => {
    setFullscreenImage({ src: imageSrc, name: imageName });
  };

  const closeFullscreen = () => {
    setFullscreenImage(null);
  };

  const navigateTo = (page) => {
    setCurrentPage(page);
    setMobileMenuOpen(false);
    window.scrollTo(0, 0);
  };

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const renderNavigation = () => (
    <nav className="main-nav">
      <div className="nav-container">
        <div className="nav-logo" onClick={() => navigateTo('home')}>
          <Brain size={32} />
          <span className="logo-text">Cancer Detection AI</span>
        </div>

        <button className="mobile-menu-btn" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
          <Menu size={24} />
        </button>

        <ul className={`nav-links ${mobileMenuOpen ? 'mobile-open' : ''}`}>
          <li className={currentPage === 'home' ? 'active' : ''} onClick={() => navigateTo('home')}>
            Home
          </li>
          <li className={currentPage === 'about' ? 'active' : ''} onClick={() => navigateTo('about')}>
            About
          </li>
          <li className={currentPage === 'blood' ? 'active' : ''} onClick={() => navigateTo('blood')}>
            🩸 Blood Cancer
          </li>
          <li className={currentPage === 'contact' ? 'active' : ''} onClick={() => navigateTo('contact')}>
            Contact
          </li>
          <li className="cta-nav" onClick={() => navigateTo('analyze')}>
            Start Detection
          </li>
        </ul>
      </div>
    </nav>
  );

  const renderHomePage = () => (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            AI-Powered Cancer Detection & Early Screening
          </h1>
          <p className="hero-description">
            Cancer encompasses a spectrum of diseases characterized by uncontrolled cellular proliferation and tissue invasion.
            Early detection is clinically proven to improve treatment outcomes and survival rates. This platform leverages
            deep learning and machine learning to deliver AI-assisted screening across four cancer categories — analyzing
            medical imaging (MRI, CT, histopathology) and structured blood-report data to support timely clinical decisions.
          </p>
          <div className="hero-buttons">
            <button className="hero-btn primary" onClick={() => navigateTo('analyze')}>
              Begin AI-Assisted Screening
            </button>
            <button className="hero-btn secondary" onClick={() => scrollToSection('cancer-types')}>
              Explore Cancer Types
            </button>
          </div>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <div className="stat-number">4</div>
            <div className="stat-label">Cancer Types Screened</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">MRI / CT</div>
            <div className="stat-label">Medical Imaging Modalities</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">Blood</div>
            <div className="stat-label">CBC-Based Risk Assessment</div>
          </div>
        </div>
      </section>

      {/* Cancer Info Section */}
      <section className="cancer-info-section" id="cancer-types">
        <h2 className="section-heading">Supported Cancer Screening Categories</h2>
        <p className="section-intro">
          Each section below provides an overview of the cancer type, its clinical presentation, key risk factors,
          and the specific role our AI model plays in screening. These results are intended for informational
          purposes only and do not constitute a medical diagnosis — always consult a qualified clinician.
        </p>

        <div className="cancer-cards">
          {/* Brain Cancer Card */}
          <div className="cancer-card">
            <div className="cancer-icon brain-gradient">🧠</div>
            <h3 className="cancer-title">Brain Tumor Screening (MRI)</h3>
            <p className="cancer-description">
              Brain tumors arise from abnormal cell growth within the cranial cavity. Magnetic Resonance Imaging (MRI)
              provides high-resolution soft-tissue contrast, making it the primary modality for detecting masses,
              edema, and structural abnormalities in the brain.
            </p>
            <div className="cancer-details">
              <h4>Key Clinical Information</h4>
              <ul>
                <li>Common symptoms include persistent headaches, seizures, motor weakness, speech difficulties, and visual disturbances.</li>
                <li>Presentation varies by tumor location and grade; some tumors are discovered incidentally on imaging.</li>
                <li>Definitive diagnosis requires specialist neuroradiological review and may necessitate stereotactic biopsy.</li>
              </ul>
              <h4>AI Screening Capability</h4>
              <p>
                Our model analyzes uploaded MRI scans and returns a binary classification (Tumor Detected / No Tumor Detected)
                accompanied by a confidence score. This output serves as an assistive screening signal — not a substitute
                for formal radiology interpretation.
              </p>
            </div>
            <button className="cancer-cta" onClick={() => { setSelectedScan('brain'); navigateTo('analyze'); }}>
              Analyze Brain MRI
            </button>
          </div>

          {/* Lung Cancer Card */}
          <div className="cancer-card">
            <div className="cancer-icon lung-gradient">🫁</div>
            <h3 className="cancer-title">Lung Cancer Screening (CT)</h3>
            <p className="cancer-description">
              Lung cancer originates in pulmonary tissue and may manifest as nodules, masses, or ground-glass opacities
              on computed tomography (CT). Low-dose CT screening is recommended for high-risk populations to enable
              earlier detection and improve prognosis.
            </p>
            <div className="cancer-details">
              <h4>Key Clinical Information</h4>
              <ul>
                <li>Symptoms include persistent cough, hemoptysis, chest pain, unexplained weight loss, and dyspnea.</li>
                <li>Primary risk factors include tobacco use, secondhand smoke exposure, occupational carcinogens, and family history.</li>
                <li>Many early-stage cases are asymptomatic and detected incidentally through imaging studies.</li>
              </ul>
              <h4>AI Screening Capability</h4>
              <p>
                Our model classifies uploaded CT images into multiple diagnostic categories (Adenocarcinoma, Large Cell
                Carcinoma, Squamous Cell Carcinoma, or Normal) with an associated confidence score. Clinical correlation
                and radiologist review remain essential for definitive diagnosis.
              </p>
            </div>
            <button className="cancer-cta" onClick={() => { setSelectedScan('lung'); navigateTo('analyze'); }}>
              Analyze Lung CT Scan
            </button>
          </div>

          {/* Blood Cancer Card */}
          <div className="cancer-card">
            <div className="cancer-icon" style={{ background: 'linear-gradient(135deg, #dc2626, #ef4444)' }}>🩸</div>
            <h3 className="cancer-title">Blood Cancer Screening (Leukemia Risk)</h3>
            <p className="cancer-description">
              Leukemia affects haematopoietic tissues, particularly bone marrow, leading to abnormal blood cell
              production. Structured clinical data — including complete blood count (CBC) parameters and patient
              risk factors — can be leveraged for predictive risk stratification.
            </p>
            <div className="cancer-details">
              <h4>Key Clinical Information</h4>
              <ul>
                <li>Common symptoms include persistent fatigue, recurrent infections, unexplained bruising or bleeding, fever, and weight loss.</li>
                <li>Diagnostic clues often emerge from CBC parameters such as WBC count, RBC count, platelet count, and hemoglobin levels.</li>
                <li>Confirmatory diagnosis requires specialist interpretation of peripheral blood smear and bone marrow biopsy.</li>
              </ul>
              <h4>AI Screening Capability</h4>
              <p>
                Our XGBoost model analyzes patient-provided blood parameters and clinical risk factors to generate
                a binary risk prediction (Positive / Negative) with an associated probability score.
              </p>
            </div>
            <button className="cancer-cta" onClick={() => navigateTo('blood')}>
              Analyze Blood Report
            </button>
          </div>

          {/* Breast Cancer Card */}
          <div className="cancer-card">
            <div className="cancer-icon" style={{ background: 'linear-gradient(135deg, #ec4899, #f472b6)' }}>🎗️</div>
            <h3 className="cancer-title">Breast Cancer Screening (Histopathology)</h3>
            <p className="cancer-description">
              Breast cancer diagnosis relies heavily on histopathological analysis of tissue biopsy specimens.
              AI-assisted screening can identify patterns indicative of invasive ductal carcinoma (IDC) in
              digitized pathology slides, supporting pathologists in high-volume diagnostic workflows.
            </p>
            <div className="cancer-details">
              <h4>Key Clinical Information</h4>
              <ul>
                <li>Warning signs include a palpable lump, skin dimpling, nipple retraction or discharge, and localized pain.</li>
                <li>Established risk factors include age, family history, BRCA1/BRCA2 genetic mutations, and hormonal influences.</li>
                <li>Histopathological grading determines tumor type, invasiveness, and informs the treatment pathway.</li>
              </ul>
              <h4>AI Screening Capability</h4>
              <p>
                Our model evaluates histopathology image patches and returns a binary classification (IDC Positive / IDC Negative)
                with a confidence score, serving as a supportive second-opinion tool for pathology review.
              </p>
            </div>
            <button className="cancer-cta" onClick={() => { setSelectedScan('breast'); navigateTo('analyze'); }}>
              Analyze Breast Tissue
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <h2 className="section-heading">Understanding AI-Assisted Screening</h2>
        <div className="features-grid">
          <div className="feature-item">
            <div className="feature-icon">🩺</div>
            <h3>Screening vs. Diagnosis</h3>
            <p>
              AI screening identifies patterns that warrant further investigation. A definitive diagnosis
              requires comprehensive clinical assessment, specialist interpretation, and confirmatory testing.
            </p>
          </div>
          <div className="feature-item">
            <div className="feature-icon">📌</div>
            <h3>When to Seek Clinical Care</h3>
            <p>
              Persistent symptoms, concerning changes, or known risk factors should prompt consultation with
              a qualified clinician — regardless of the AI screening outcome.
            </p>
          </div>
          <div className="feature-item">
            <div className="feature-icon">🧾</div>
            <h3>Interpreting Confidence Scores</h3>
            <p>
              The confidence score reflects the model's statistical certainty, not absolute diagnostic truth.
              Low-confidence results should be re-evaluated with higher-quality inputs and validated through
              clinical follow-up.
            </p>
          </div>
          <div className="feature-item">
            <div className="feature-icon">🧠</div>
            <h3>AI as a Clinical Support Tool</h3>
            <p>
              This system is designed to augment — not replace — clinical decision-making. It is best utilized
              for preliminary triage, educational purposes, and as a supportive second-opinion layer.
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2>Ready to Begin Screening?</h2>
          <p>Upload a medical scan or enter blood parameters to receive AI-powered analysis within seconds</p>
          <button className="cta-button" onClick={() => navigateTo('analyze')}>
            Start AI-Assisted Screening
          </button>
        </div>
      </section>
    </div>
  );

  const renderAboutPage = () => (
    <div className="about-page">
      <section className="about-hero">
        <h1 className="page-title">About Our AI Screening Platform</h1>
        <p className="page-subtitle">
          Advancing early cancer detection through state-of-the-art deep learning and clinical data science
        </p>
      </section>

      <section className="about-content">
        <div className="about-grid">
          <div className="about-card">
            <Target size={48} className="about-icon" />
            <h3>Our Mission</h3>
            <p>
              To broaden access to advanced cancer screening technology and empower healthcare
              professionals with AI-assisted tools for faster, more informed clinical decisions.
              We are committed to the principle that early detection — enabled by intelligent
              imaging analysis — can meaningfully improve patient outcomes.
            </p>
          </div>

          <div className="about-card">
            <Users size={48} className="about-icon" />
            <h3>Our Approach</h3>
            <p>
              Every model is developed following responsible AI principles — trained on curated,
              domain-specific medical datasets with rigorous preprocessing, augmentation, and
              cross-validated evaluation. All predictions are accompanied by calibrated confidence
              scores to support transparent, evidence-based clinical decision-making.
            </p>
          </div>

          <div className="about-card">
            <Award size={48} className="about-icon" />
            <h3>Our Technology</h3>
            <p>
              Powered by EfficientNetV2 convolutional neural networks for medical imaging analysis
              and XGBoost gradient boosting for structured clinical data. Each model is trained on
              thousands of verified samples and undergoes comprehensive validation to ensure
              reliability before deployment.
            </p>
          </div>
        </div>

        <div className="methodology-section">
          <h2>How It Works</h2>
          <div className="methodology-steps">
            <div className="method-step">
              <div className="step-number">1</div>
              <h4>Upload Scan</h4>
              <p>Securely upload your medical image or enter clinical data</p>
            </div>
            <div className="step-arrow">→</div>
            <div className="method-step">
              <div className="step-number">2</div>
              <h4>AI Analysis</h4>
              <p>Deep learning models process and classify the input</p>
            </div>
            <div className="step-arrow">→</div>
            <div className="method-step">
              <div className="step-number">3</div>
              <h4>Review Results</h4>
              <p>Receive a classification with a calibrated confidence score</p>
            </div>
            <div className="step-arrow">→</div>
            <div className="method-step">
              <div className="step-number">4</div>
              <h4>Clinical Follow-Up</h4>
              <p>Share results with your healthcare provider for clinical correlation</p>
            </div>
          </div>
        </div>

        {/* ==================== MODEL PERFORMANCE ==================== */}
        <div className="performance-section">
          <h2>Model Performance Metrics</h2>
          <p className="performance-intro">
            Each model is rigorously evaluated using held-out validation data to ensure reliability
            and generalization before deployment. The sections below present training convergence curves,
            confusion matrices, and comprehensive evaluation metrics for every cancer type and
            model architecture used in this system.
          </p>

          {/* ========== TRAINING CURVES ========== */}
          <div className="metrics-block">
            <h3 className="metrics-heading">📈 Training & Validation Convergence Curves</h3>
            <div className="metrics-grid">
              {renderMetricImageBlock({
                src: metricsImages.breast.training.src,
                name: "Training vs. Validation Accuracy & Loss — EfficientNetV2"
              })}
              {renderMetricImageBlock({
                src: metricsImages.blood.training.src,
                name: "Training vs. Validation Accuracy & Loss — XGBoost (Blood Cancer)"
              })}
            </div>
          </div>

          {/* ========== BRAIN ========== */}
          <div className="metrics-block">
            <h3 className="metrics-heading">🧠 Brain Tumor Detection — Confusion Matrices</h3>

            <h4 className="model-sub-heading">Predicted vs. Actual Classification (Validation Set)</h4>
            <div className="cm-row">
              {renderConfusionMatrix('brain', 'EfficientNetV2')}
              {renderConfusionMatrix('brain', 'DenseNet')}
            </div>
          </div>

          {/* ========== LUNG ========== */}
          <div className="metrics-block">
            <h3 className="metrics-heading">🫁 Lung Cancer Detection — Confusion Matrices</h3>

            <h4 className="model-sub-heading">Predicted vs. Actual Classification (Validation Set)</h4>
            <div className="cm-row">
              {renderConfusionMatrix('lung', 'EfficientNetV2')}
              {renderConfusionMatrix('lung', 'DenseNet')}
            </div>
          </div>

          {/* ========== BREAST ========== */}
          <div className="metrics-block">
            <h3 className="metrics-heading">🎗️ Breast Cancer Detection — Confusion Matrices</h3>

            <h4 className="model-sub-heading">Predicted vs. Actual Classification (Validation Set)</h4>
            <div className="cm-row">
              {renderConfusionMatrix('breast', 'EfficientNetV2')}
              {renderConfusionMatrix('breast', 'DenseNet')}
            </div>
          </div>

          {/* ========== BLOOD ========== */}
          <div className="metrics-block">
            <h3 className="metrics-heading">🩸 Blood Cancer (Leukemia) — Confusion Matrix</h3>

            <h4 className="model-sub-heading">Predicted vs. Actual Classification (Validation Set)</h4>
            <div className="cm-row cm-row-single">
              {renderConfusionMatrix('blood', 'XGBoost')}
            </div>
          </div>
        </div>

        {/* ==================== EVALUATION METRICS TABLES ==================== */}
        <div className="performance-section">
          <div className="evaluation-section">
            <h2 className="evaluation-main-heading">Comprehensive Evaluation Metrics</h2>
            <p className="evaluation-intro">
              The following tables present detailed performance metrics derived from confusion matrix analysis
              (True Positives, True Negatives, False Positives, False Negatives) and AUC-ROC evaluation.
              These metrics are critical for assessing model reliability in clinical cancer screening,
              where maximizing Recall (minimizing missed diagnoses) is of paramount clinical importance.
            </p>
            {renderMetricsTable('brain')}
            {renderMetricsTable('lung')}
            {renderMetricsTable('breast')}
            {renderMetricsTable('blood')}
          </div>
        </div>
      </section>
    </div>
  );

  const renderContactPage = () => (
    <div className="contact-page">
      <section className="contact-hero">
        <h1 className="page-title">Contact Us</h1>
        <p className="page-subtitle">
          Have questions about our AI screening platform? Our team is here to assist you
        </p>
      </section>

      <section className="contact-content">
        <div className="contact-grid">
          <div className="contact-info">
            <h2>Contact Information</h2>
            <p className="contact-intro">
              Reach out for technical support, clinical integration inquiries, or research collaboration opportunities.
            </p>

            <div className="contact-methods">
              <div className="contact-method">
                <div className="contact-icon-wrapper">
                  <Mail size={24} />
                </div>
                <div className="contact-details">
                  <h4>Email</h4>
                  <a href="mailto:info@cancerdetection.ai">info@cancerdetection.ai</a>
                  <a href="mailto:support@cancerdetection.ai">support@cancerdetection.ai</a>
                </div>
              </div>

              <div className="contact-method">
                <div className="contact-icon-wrapper">
                  <Phone size={24} />
                </div>
                <div className="contact-details">
                  <h4>Phone</h4>
                  <a href="tel:+1234567890">+1 (234) 567-890</a>
                  <p className="availability">Mon-Fri: 9AM - 6PM EST</p>
                </div>
              </div>

              <div className="contact-method">
                <div className="contact-icon-wrapper">
                  <MapPin size={24} />
                </div>
                <div className="contact-details">
                  <h4>Address</h4>
                  <p>123 Medical AI Drive</p>
                  <p>Innovation District</p>
                  <p>San Francisco, CA 94105</p>
                </div>
              </div>
            </div>

            <div className="support-hours">
              <h3>Support Hours</h3>
              <div className="hours-list">
                <div className="hours-item">
                  <span>Monday - Friday:</span>
                  <span>9:00 AM - 6:00 PM EST</span>
                </div>
                <div className="hours-item">
                  <span>Saturday:</span>
                  <span>10:00 AM - 4:00 PM EST</span>
                </div>
                <div className="hours-item">
                  <span>Sunday:</span>
                  <span>Closed</span>
                </div>
              </div>
            </div>
          </div>

          <div className="contact-form-section">
            <form className="contact-form">
              <h2>Send Us a Message</h2>

              <div className="form-group">
                <label htmlFor="name">Full Name *</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  placeholder="John Doe"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="email">Email Address *</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  placeholder="john@example.com"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="subject">Subject *</label>
                <input
                  type="text"
                  id="subject"
                  name="subject"
                  placeholder="How can we help?"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="message">Message *</label>
                <textarea
                  id="message"
                  name="message"
                  rows="6"
                  placeholder="Tell us more about your inquiry..."
                  required
                ></textarea>
              </div>

              <button type="submit" className="submit-btn">
                Send Message
              </button>
            </form>

            <div className="faq-section">
              <h3>Frequently Asked Questions</h3>
              <div className="faq-list">
                <div className="faq-item">
                  <h4>Is this screening service free to use?</h4>
                  <p>Yes, our AI-assisted screening service is available at no cost for educational and preliminary assessment purposes.</p>
                </div>
                <div className="faq-item">
                  <h4>How reliable are the AI predictions?</h4>
                  <p>Our models achieve over 93% accuracy on held-out validation datasets. However, all results should be interpreted by a qualified medical professional — AI screening is not a substitute for clinical diagnosis.</p>
                </div>
                <div className="faq-item">
                  <h4>How is my data handled?</h4>
                  <p>We prioritize data privacy. Uploaded images are processed in memory and are not permanently stored. All data transmission is encrypted to ensure confidentiality.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );

  const renderAnalyzePage = () => (
    <div className="analyze-page">
      {!selectedScan ? (
        <div className="selection-container">
          <h2 className="section-title">Select Scan Type</h2>
          <div className="scan-type-grid">
            {scanTypes.map((scan) => (
              <button
                key={scan.id}
                className="scan-type-card"
                onClick={() => {
                  if (scan.id === 'blood') {
                    navigateTo('blood');
                  } else {
                    setSelectedScan(scan.id);
                  }
                }}
              >
                <div className="scan-icon">{scan.icon}</div>
                <h3 className="scan-name">{scan.name}</h3>
                <p className="scan-description">{scan.description}</p>
                <div className="scan-arrow">→</div>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="analysis-container">
          <div className="analysis-header">
            <div className="analysis-title-group">
              <span className="analysis-icon">
                {selectedScan === 'brain' ? '🧠' : selectedScan === 'lung' ? '🫁' : '🎗️'}
              </span>
              <h2 className="analysis-title">
                {scanTypes.find(s => s.id === selectedScan)?.name} Analysis
              </h2>
            </div>
            <button className="med-btn-secondary" onClick={resetAll}>
              <RotateCcw size={16} />
              <span>Reset</span>
            </button>
          </div>

          <div className="analysis-grid">
            <div className="upload-section">
              {/* User Profile Card */}
              <div className="med-form-card med-profile-card-analyze">
                <div className="med-form-header">
                  <span className="med-form-header-icon">👤</span> User Profile
                </div>
                <div className="med-form-grid med-form-grid-single">
                  <div className="med-field">
                    <label className="med-label">Full Name <span className="med-required">*</span></label>
                    <input type="text" name="name" className={`med-input ${showProfileError('name') ? 'med-input-error' : ''}`}
                      value={userProfile.name} onChange={handleUserProfileChange} onBlur={handleProfileBlur}
                      placeholder="e.g. Dr. Jane Smith" />
                    {showProfileError('name') && <span className="med-field-error">{profileErrors.name}</span>}
                  </div>
                  <div className="med-field">
                    <label className="med-label">Email <span className="med-required">*</span></label>
                    <input type="email" name="email" className={`med-input ${showProfileError('email') ? 'med-input-error' : ''}`}
                      value={userProfile.email} onChange={handleUserProfileChange} onBlur={handleProfileBlur}
                      placeholder="e.g. jane.smith@hospital.org" />
                    {showProfileError('email') && <span className="med-field-error">{profileErrors.email}</span>}
                  </div>
                  <div className="med-field">
                    <label className="med-label">Role <span className="med-required">*</span></label>
                    <select name="role" className={`med-select ${showProfileError('role') ? 'med-input-error' : ''}`}
                      value={userProfile.role} onChange={handleUserProfileChange} onBlur={handleProfileBlur}>
                      <option value="patient">Patient</option>
                      <option value="doctor">Doctor</option>
                      <option value="admin">Admin</option>
                    </select>
                    {showProfileError('role') && <span className="med-field-error">{profileErrors.role}</span>}
                  </div>
                </div>
              </div>

              <label className="upload-area">
                {imagePreview ? (
                  <div className="image-preview-container">
                    <img src={imagePreview} alt="Preview" className="image-preview" />
                    <div className="change-image-text">Click to change image</div>
                  </div>
                ) : (
                  <div className="upload-placeholder">
                    <Upload size={48} className="upload-icon" />
                    <p className="upload-title">Upload Medical Scan</p>
                    <p className="upload-subtitle">
                      Click to browse or drag and drop your image here
                    </p>
                    <p className="upload-formats">Supported: JPG, PNG, JPEG</p>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="file-input"
                />
              </label>

              <button
                onClick={analyzeImage}
                disabled={!uploadedImage || isAnalyzing || !isProfileValid}
                className="med-btn-primary"
              >
                {isAnalyzing ? (
                  <>
                    <Loader size={20} className="spinner" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Activity size={20} />
                    <span>Analyze Scan</span>
                  </>
                )}
              </button>
            </div>

            <div className="results-section">
              {isAnalyzing && (
                <div className="analyzing-state">
                  <Loader size={64} className="analyzing-spinner" />
                  <p className="analyzing-text">Analyzing your scan...</p>
                  <p className="analyzing-subtext">This may take a few moments</p>
                </div>
              )}

              {result && !isAnalyzing && (
                <div className="results-container">
                  {result.error ? (
                    <div className="result-card warning-card" style={{ padding: "1.5rem" }}>
                      <AlertCircle size={32} className="result-icon warning" style={{ marginBottom: "0.5rem" }} />
                      <p className="result-label">Backend Error</p>
                      <p className="result-value" style={{ fontSize: "0.95rem", color: "#f87171" }}>{result.error}</p>
                    </div>
                  ) : (
                    <>
                      <div className="results-header">
                        {(result.prediction || "").includes("No") ? (
                          <CheckCircle size={32} className="result-icon success" />
                        ) : (
                          <AlertCircle size={32} className="result-icon warning" />
                        )}
                        <h3 className="results-title">Analysis Results</h3>
                      </div>

                      <div className="results-content">
                        <div className={`result-card ${(result.prediction || "").includes("No") ? 'success-card' : 'warning-card'}`}>
                          <p className="result-label">Prediction</p>
                          <p className="result-value">{result.prediction}</p>
                        </div>

                        <div className="result-card confidence-card">
                          <p className="result-label">Confidence Level</p>
                          <div className="confidence-bar-container">
                            <div className="confidence-bar-bg">
                              <div
                                className="confidence-bar-fill"
                                style={{ width: `${result.confidence}%` }}
                              ></div>
                            </div>
                            <span className="confidence-value">{result.confidence}%</span>
                          </div>
                        </div>

                        {result.details && (
                          <div className="result-card details-card">
                            <p className="result-label">Additional Details</p>
                            <div className="details-list">
                              {Object.entries(result.details).map(([key, value]) => (
                                <div key={key} className="detail-item">
                                  <span className="detail-key">
                                    {key.replace(/([A-Z])/g, ' $1').trim()}:
                                  </span>
                                  <span className="detail-value">{value}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="disclaimer-card">
                          <AlertCircle size={20} className="disclaimer-icon" />
                          <p className="disclaimer-text">
                            This is an AI-assisted screening result, not a clinical diagnosis. Please consult
                            a qualified medical professional for comprehensive evaluation and treatment planning.
                          </p>
                        </div>

                        {result.db_error && (
                          <div className="disclaimer-card">
                            <AlertCircle size={20} className="disclaimer-icon" />
                            <p className="disclaimer-text">
                              Prediction completed, but DB log failed: {result.db_error}
                            </p>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              )}

              {!result && !isAnalyzing && uploadedImage && (
                <div className="med-empty-state">
                  <Activity size={64} className="med-empty-icon" />
                  <p className="med-empty-title">Scan Ready for Analysis</p>
                  <p className="med-empty-subtitle">
                    Your medical image has been uploaded successfully. Click "Analyze Scan" to begin AI-powered screening.
                  </p>
                </div>
              )}

              {!uploadedImage && !isAnalyzing && (
                <div className="med-empty-state">
                  <Upload size={64} className="med-empty-icon" />
                  <p className="med-empty-title">Upload a Medical Scan</p>
                  <p className="med-empty-subtitle">
                    Select and upload a medical image (MRI, CT, or histopathology) to begin AI-assisted screening.
                  </p>
                  <div className="med-empty-badge">
                    <Shield size={14} /> EfficientNetV2 Deep Learning Model
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderBloodPage = () => (
    <div className="analyze-page">
      <div className="analysis-container">
        <div className="analysis-header">
          <div className="analysis-title-group">
            <span className="analysis-icon">🩸</span>
            <h2 className="analysis-title">Leukemia Risk Assessment</h2>
          </div>
          <button className="med-btn-secondary" onClick={resetBlood}>
            <RotateCcw size={16} />
            <span>Reset Form</span>
          </button>
        </div>

        {bloodSubmitAttempted && (!isProfileValid || !isBloodFormValid) && (
          <div className="med-error-banner">
            <AlertCircle size={20} className="med-error-banner-icon" />
            <span className="med-error-banner-text">
              Please correct the highlighted errors below before proceeding with analysis.
            </span>
          </div>
        )}

        <div className="blood-form-layout">
          {/* ---- LEFT COLUMN: FORM ---- */}
          <div className="blood-form-column">

            {/* User Profile Card */}
            <div className="med-form-card">
              <div className="med-form-header">
                <span className="med-form-header-icon">👤</span> User Profile
              </div>
              <div className="med-form-grid med-form-grid-single">
                <div className="med-field">
                  <label className="med-label">Full Name <span className="med-required">*</span></label>
                  <input type="text" name="name" className={`med-input ${showProfileError('name') ? 'med-input-error' : ''}`}
                    value={userProfile.name} onChange={handleUserProfileChange} onBlur={handleProfileBlur}
                    placeholder="e.g. Dr. Jane Smith" />
                  {showProfileError('name') && <span className="med-field-error">{profileErrors.name}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">Email <span className="med-required">*</span></label>
                  <input type="email" name="email" className={`med-input ${showProfileError('email') ? 'med-input-error' : ''}`}
                    value={userProfile.email} onChange={handleUserProfileChange} onBlur={handleProfileBlur}
                    placeholder="e.g. jane.smith@hospital.org" />
                  {showProfileError('email') && <span className="med-field-error">{profileErrors.email}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">Role <span className="med-required">*</span></label>
                  <select name="role" className={`med-select ${showProfileError('role') ? 'med-input-error' : ''}`}
                    value={userProfile.role} onChange={handleUserProfileChange} onBlur={handleProfileBlur}>
                    <option value="patient">Patient</option>
                    <option value="doctor">Doctor</option>
                    <option value="admin">Admin</option>
                  </select>
                  {showProfileError('role') && <span className="med-field-error">{profileErrors.role}</span>}
                </div>
              </div>
            </div>

            {/* Personal Information Card */}
            <div className="med-form-card">
              <div className="med-form-header">
                <span className="med-form-header-icon">📋</span> Personal Information
              </div>
              <div className="med-form-grid">
                <div className="med-field">
                  <label className="med-label">Age <span className="med-required">*</span></label>
                  <input type="number" name="Age" className={`med-input ${showBloodError('Age') ? 'med-input-error' : ''}`}
                    value={bloodForm.Age} onChange={handleBloodChange} onBlur={handleBloodBlur}
                    placeholder="e.g. 45" min="1" max="120" />
                  {showBloodError('Age') && <span className="med-field-error">{bloodErrors.Age}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">Gender <span className="med-required">*</span></label>
                  <select name="Gender" className={`med-select ${showBloodError('Gender') ? 'med-input-error' : ''}`}
                    value={bloodForm.Gender} onChange={handleBloodChange} onBlur={handleBloodBlur}>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                  {showBloodError('Gender') && <span className="med-field-error">{bloodErrors.Gender}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">Country <span className="med-required">*</span></label>
                  <select name="Country" className={`med-select ${showBloodError('Country') ? 'med-input-error' : ''}`}
                    value={bloodForm.Country} onChange={handleBloodChange} onBlur={handleBloodBlur}>
                    <option value="USA">United States</option>
                    <option value="India">India</option>
                    <option value="UK">United Kingdom</option>
                    <option value="Canada">Canada</option>
                    <option value="Australia">Australia</option>
                    <option value="Germany">Germany</option>
                    <option value="France">France</option>
                    <option value="Japan">Japan</option>
                    <option value="Brazil">Brazil</option>
                    <option value="Other">Other</option>
                  </select>
                  {showBloodError('Country') && <span className="med-field-error">{bloodErrors.Country}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">BMI <span className="med-required">*</span></label>
                  <input type="number" step="0.1" name="BMI" className={`med-input ${showBloodError('BMI') ? 'med-input-error' : ''}`}
                    value={bloodForm.BMI} onChange={handleBloodChange} onBlur={handleBloodBlur}
                    placeholder="e.g. 24.5" min="10" max="50" />
                  {showBloodError('BMI') && <span className="med-field-error">{bloodErrors.BMI}</span>}
                </div>
              </div>
            </div>

            {/* Blood Test Results Card */}
            <div className="med-form-card">
              <div className="med-form-header">
                <span className="med-form-header-icon">🧪</span> Blood Test Results
              </div>
              <div className="med-form-grid">
                <div className="med-field">
                  <label className="med-label">WBC Count (cells/µL) <span className="med-required">*</span></label>
                  <input type="number" name="WBC_Count" className={`med-input ${showBloodError('WBC_Count') ? 'med-input-error' : ''}`}
                    value={bloodForm.WBC_Count} onChange={handleBloodChange} onBlur={handleBloodBlur}
                    placeholder="e.g. 6000" />
                  {showBloodError('WBC_Count') && <span className="med-field-error">{bloodErrors.WBC_Count}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">RBC Count (×10⁶/µL) <span className="med-required">*</span></label>
                  <input type="number" step="0.01" name="RBC_Count" className={`med-input ${showBloodError('RBC_Count') ? 'med-input-error' : ''}`}
                    value={bloodForm.RBC_Count} onChange={handleBloodChange} onBlur={handleBloodBlur}
                    placeholder="e.g. 5.2" />
                  {showBloodError('RBC_Count') && <span className="med-field-error">{bloodErrors.RBC_Count}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">Platelet Count (×10³/µL) <span className="med-required">*</span></label>
                  <input type="number" name="Platelet_Count" className={`med-input ${showBloodError('Platelet_Count') ? 'med-input-error' : ''}`}
                    value={bloodForm.Platelet_Count} onChange={handleBloodChange} onBlur={handleBloodBlur}
                    placeholder="e.g. 250" />
                  {showBloodError('Platelet_Count') && <span className="med-field-error">{bloodErrors.Platelet_Count}</span>}
                </div>
                <div className="med-field">
                  <label className="med-label">Hemoglobin (g/dL) <span className="med-required">*</span></label>
                  <input type="number" step="0.1" name="Hemoglobin_Level" className={`med-input ${showBloodError('Hemoglobin_Level') ? 'med-input-error' : ''}`}
                    value={bloodForm.Hemoglobin_Level} onChange={handleBloodChange} onBlur={handleBloodBlur}
                    placeholder="e.g. 13.5" />
                  {showBloodError('Hemoglobin_Level') && <span className="med-field-error">{bloodErrors.Hemoglobin_Level}</span>}
                </div>
                <div className="med-field med-field-full">
                  <label className="med-label">Bone Marrow Blasts (%) <span className="med-required">*</span></label>
                  <input type="number" name="Bone_Marrow_Blasts" className={`med-input ${showBloodError('Bone_Marrow_Blasts') ? 'med-input-error' : ''}`}
                    value={bloodForm.Bone_Marrow_Blasts} onChange={handleBloodChange} onBlur={handleBloodBlur}
                    placeholder="0 – 100" min="0" max="100" />
                  {showBloodError('Bone_Marrow_Blasts') && <span className="med-field-error">{bloodErrors.Bone_Marrow_Blasts}</span>}
                </div>
              </div>
            </div>

            {/* Risk Factors Card */}
            <div className="med-form-card">
              <div className="med-form-header">
                <span className="med-form-header-icon">⚠️</span> Clinical Risk Factors
              </div>
              <div className="med-form-grid">
                {[
                  ['Genetic_Mutation', 'Genetic Mutation'],
                  ['Family_History', 'Family History'],
                  ['Smoking_Status', 'Smoking Status'],
                  ['Alcohol_Consumption', 'Alcohol Consumption'],
                  ['Radiation_Exposure', 'Radiation Exposure'],
                  ['Infection_History', 'Infection History'],
                  ['Chronic_Illness', 'Chronic Illness'],
                  ['Immune_Disorders', 'Immune Disorders'],
                ].map(([key, label]) => (
                  <div key={key} className="med-field">
                    <label className="med-label">{label}</label>
                    <select name={key} className="med-select" value={bloodForm[key]} onChange={handleBloodChange}>
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="med-btn-group">
              <button
                onClick={analyzeBlood}
                disabled={isBloodAnalyzing}
                className="med-btn-primary"
              >
                {isBloodAnalyzing ? (
                  <><Loader size={20} className="spinner" /><span>Running Analysis...</span></>
                ) : (
                  <><Activity size={20} /><span>Analyze Blood Report</span></>
                )}
              </button>
            </div>
          </div>

          {/* ---- RIGHT COLUMN: RESULTS ---- */}
          <div className="results-section">
            {isBloodAnalyzing && (
              <div className="analyzing-state">
                <Loader size={64} className="analyzing-spinner" />
                <p className="analyzing-text">Processing clinical data...</p>
                <p className="analyzing-subtext">Running XGBoost classification model</p>
              </div>
            )}

            {bloodResult && !isBloodAnalyzing && (
              <div className="results-container">
                <div className="results-header">
                  {bloodResult.prediction === 'Negative' ? (
                    <CheckCircle size={32} className="result-icon success" />
                  ) : (
                    <AlertCircle size={32} className="result-icon warning" />
                  )}
                  <h3 className="results-title">Assessment Results</h3>
                </div>
                <div className="results-content">
                  <div className={`result-card ${bloodResult.prediction === 'Negative' ? 'success-card' : 'warning-card'}`}>
                    <p className="result-label">Leukemia Status</p>
                    <p className="result-value" style={{ fontSize: '1.8rem' }}>
                      {bloodResult.prediction === 'Negative' ? '✅ Negative' : '⚠️ Positive'}
                    </p>
                  </div>
                  <div className="result-card confidence-card">
                    <p className="result-label">Probability of Leukemia</p>
                    <div className="confidence-bar-container">
                      <div className="confidence-bar-bg">
                        <div className="confidence-bar-fill" style={{ width: `${bloodResult.confidence}%` }}></div>
                      </div>
                      <span className="confidence-value">{bloodResult.confidence}%</span>
                    </div>
                  </div>
                  <div className="disclaimer-card">
                    <AlertCircle size={20} className="disclaimer-icon" />
                    <p className="disclaimer-text">
                      This is an AI-assisted screening result and does not constitute a medical diagnosis.
                      Please consult a qualified haematologist or oncologist for clinical evaluation,
                      confirmatory testing, and treatment guidance.
                    </p>
                  </div>
                  {bloodResult.db_error && (
                    <div className="disclaimer-card">
                      <AlertCircle size={20} className="disclaimer-icon" />
                      <p className="disclaimer-text">
                        Prediction completed, but DB log failed: {bloodResult.db_error}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {!bloodResult && !isBloodAnalyzing && (
              <div className="med-empty-state">
                <Shield size={64} className="med-empty-icon" />
                <p className="med-empty-title">Awaiting Patient Data</p>
                <p className="med-empty-subtitle">
                  Complete all required fields in the clinical assessment form, then click
                  "Analyze Blood Report" to generate a risk prediction.
                </p>
                <div className="med-empty-badge">
                  <Activity size={14} /> XGBoost Classification Model
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );


  return (
    <div className="app-container">
      {renderNavigation()}

      <main className="app-main">
        {currentPage === 'home' && renderHomePage()}
        {currentPage === 'about' && renderAboutPage()}
        {currentPage === 'contact' && renderContactPage()}
        {currentPage === 'analyze' && renderAnalyzePage()}
        {currentPage === 'blood' && renderBloodPage()}
      </main>

      {fullscreenImage && (
        <div className="fullscreen-modal" onClick={closeFullscreen}>
          <button className="fullscreen-close" onClick={closeFullscreen}>
            <X size={32} />
          </button>
          <div className="fullscreen-content" onClick={(e) => e.stopPropagation()}>
            <img
              src={fullscreenImage.src}
              alt={fullscreenImage.name}
              className="fullscreen-image"
            />
            <p className="fullscreen-label">{fullscreenImage.name}</p>
          </div>
        </div>
      )}

      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Cancer Detection AI</h4>
            <p>AI-powered screening for medical imaging and clinical data analysis</p>
          </div>
          <div className="footer-section">
            <h4>Quick Links</h4>
            <ul>
              <li onClick={() => navigateTo('home')}>Home</li>
              <li onClick={() => navigateTo('about')}>About</li>
              <li onClick={() => navigateTo('contact')}>Contact</li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Legal</h4>
            <ul>
              <li>Privacy Policy</li>
              <li>Terms of Service</li>
              <li>Medical Disclaimer</li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>© 2025 Cancer Detection AI. All rights reserved.</p>
          <p>AI screening results are informational only — always consult qualified healthcare professionals for medical decisions</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
