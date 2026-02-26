# 🧠 ML Analyzer – Desktop Machine Learning Application

## 📌 Overview

**ML Analyzer** is a modular desktop machine learning application that allows users to:

- Upload any CSV dataset  
- Automatically preprocess the data  
- Select the ML task (Classification / Regression / Clustering)  
- Choose an algorithm dynamically  
- Train the model  
- Evaluate performance using metrics  
- Visualize results  
- Generate predictions  

This project demonstrates the design of an end-to-end machine learning system integrated with a graphical user interface.

---

# 🎯 The Problem

Many machine learning projects focus only on training models inside notebooks.  
However, real-world systems require:

- Structured architecture  
- Data validation  
- Dynamic model selection  
- Error handling  
- Reusable pipelines  
- User interaction  

The goal of this project was to build a complete ML workflow inside a desktop application — not just a model script.

---

# 🏗 System Architecture

The project is structured into separate layers:

```
ML-Analyzer/
│
├── GUI/
│   ├── page_1.py       # Landing page
│   ├── page_2.py       # Dataset configuration & model selection
│   ├── page_3.py       # Training, evaluation & visualization
│
├── ML/
│   ├── preprocess.py   # Data preprocessing pipeline
│   ├── model.py        # Model training logic
│   ├── predict.py      # Prediction & evaluation utilities
│
└── main.py             # Application entry point
```

### Design Principles Used

- Separation of Concerns  
- Modular ML pipeline  
- Dynamic frame switching  
- Reusable preprocessing artifacts  
- Custom exception handling  
- Defensive input validation  

---

# 🔄 Workflow

1. User uploads CSV dataset  
2. System detects feature columns  
3. User selects:
   - Target column
   - Task type
   - Algorithm
4. Data is automatically:
   - Cleaned
   - Encoded
   - Scaled
5. Model is trained
6. Evaluation metrics are displayed
7. Visual performance plots are generated
8. Predictions can be made on new inputs

---

# 🤖 Supported Machine Learning Tasks

## 📊 Classification
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📈 Regression
- Linear Regression

Metrics:
- MAE
- RMSE
- R² Score
- Predicted vs Actual Plot

---

## 🔍 Clustering
- K-Means

Output:
- Cluster assignments
- Cluster visualization

---

# 🛠 Preprocessing Pipeline

The system automatically handles:

- Missing value imputation  
- Label encoding / One-hot encoding  
- Feature scaling using StandardScaler  
- Target encoding  
- Artifact preservation (encoders & scaler for inference)  

Custom `PreprocessingError` and `PredictionError` exceptions ensure robustness.

---

# 📊 Visualization

The application generates:

- Confusion Matrix (Classification)
- Actual vs Predicted scatter plot (Regression)
- Cluster visualization (Clustering)

All plots are integrated directly inside the GUI using Matplotlib.

---

# 🧩 Key Technical Features

- Modular multi-page GUI (CustomTkinter)
- Dynamic algorithm switching
- Runtime model selection
- Validation of user inputs
- Structured preprocessing artifacts
- Integrated evaluation system
- Clean frame navigation logic
- Exception-safe operations

---

# 🚀 Technologies Used

- Python
- CustomTkinter
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

# 💡 Challenges Solved

- Synchronizing preprocessing between training and prediction
- Handling dynamic feature selection
- Managing scaling and encoding artifacts
- Integrating ML logic into GUI environment
- Creating reusable and structured ML components

---

# 🔮 Future Improvements

- Hyperparameter customization
- Cross-validation support
- Model saving/loading
- More algorithms (Random Forest, XGBoost, Logistic Regression)
- Automatic model comparison
- Feature importance visualization

---

# 👨‍💻 Author

**Samir Elhadad**  
AI & Data Science Student  
Machine Learning & Python Developer  

---

## 📎 Project Type

End-to-End Machine Learning Desktop System  
Designed to demonstrate practical ML engineering and application architecture.
