# Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent transactions in a credit card dataset using multiple **supervised** and **unsupervised** machine learning algorithms. It handles real-world challenges like **class imbalance**, **anomaly detection**, and **model evaluation** using appropriate metrics.

---

## Models Used

### Supervised Learning
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Unsupervised Learning
- Isolation Forest
- K-Means Clustering

---

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File**: `creditcard.csv`
- **Description**: Contains anonymized features of credit card transactions along with `Amount`, `Time`, and `Class` (1 = Fraud, 0 = Legitimate).

---

## Workflow

1. Loaded and preprocessed the dataset
2. Scaled features (`StandardScaler`)
3. Handled class imbalance using undersampling
4. Trained models (3 supervised + 2 unsupervised)
5. Evaluated each model using precision, recall, F1, ROC-AUC, and PR-AUC
6. Visualized performance comparison of all models

---

## Evaluation Metrics

Each model is evaluated on:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

## Key Highlights

- **XGBoost** achieved the best results among supervised models in recall and AUC.
- **Isolation Forest** successfully detected fraud cases without labeled data.
- **K-Means** demonstrated basic unsupervised anomaly detection using clustering and majority mapping.

---

## Technologies Used

- **Python**
  - `pandas`, `numpy` – data manipulation
  - `matplotlib`, `seaborn` – visualizations
  - `scikit-learn` – ML models, preprocessing, metrics
  - `xgboost` – for gradient boosting
  - `imbalanced-learn` – handling class imbalance
- **Machine Learning Concepts**
  - Supervised vs. Unsupervised learning
  - Anomaly Detection
  - Model Evaluation Metrics
  - Data Scaling
  - Class Imbalance Handling
