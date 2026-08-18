# Telecom Customer Churn Classification

## 1. Problem Statement

**Business Challenge**: In the highly competitive telecom industry, customer churn (the rate at which customers leave a service) is a critical metric. Predicting which customers are likely to churn enables proactive retention strategies, reducing revenue loss and improving customer lifetime value.

**Objective**: Build a machine learning classification model to predict whether a customer will churn (cancel their subscription) based on their demographic, service, and billing characteristics.

---

## 2. Dataset Description

**Dataset Name**: IBM Telco Customer Churn

**Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Dataset Characteristics**:
- **Instances**: 7,043 customer records
- **Features**: 21 input features
- **Target Variable**: Churn (Binary: Yes = 1, No = 0)
- **Class Distribution**: Imbalanced dataset requiring attention to recall metrics

**Feature Categories**:

1. **Customer Demographics**:
   - Senior Citizen, Partner, Dependents, Tenure

2. **Service Information**:
   - Phone Service, Multiple Lines, Internet Service Type
   - Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies

3. **Billing Information**:
   - Contract, Billing Cycle, Paperless Billing, Payment Method
   - Monthly Charges, Total Charges

---

## 3. Data Preprocessing

**Preprocessing Pipeline Overview**:

1. **Load and Inspect**:
   - Load CSV using pandas
   - Check shape, columns, data types, missing values, duplicates
   - Analyze target distribution

2. **Feature Engineering**:
   - Remove `customerID` (identifier, not predictive)
   - Convert `TotalCharges` to numeric (original may contain blanks)
   - Handle missing/invalid values in `TotalCharges`

3. **Target Encoding**:
   - Encode binary target: No → 0, Yes → 1

4. **Categorical & Numerical Feature Separation**:
   - Identify categorical features (binary and multi-class)
   - Identify numerical features (continuous)

5. **Feature Transformation**:
   - **Categorical**: OneHotEncoder (handles unseen categories)
   - **Numerical**: StandardScaler (essential for distance-based models like KNN)
   - **Sparse Matrix Handling**: Convert to dense for GaussianNB compatibility

6. **Pipeline Construction**:
   - Use `ColumnTransformer` + `Pipeline` to prevent data leakage
   - Apply preprocessing only to training set; transform test set using fitted pipeline

7. **Data Splitting**:
   - Stratified train/test split (80/20)
   - `random_state=42` for reproducibility
   - Same split used for all five models

---

## 4. Models Used

### 1. Logistic Regression
A linear model for binary classification that estimates the probability of class membership using a logistic function.
- **Strengths**: Interpretable, fast, good baseline
- **Configuration**: max_iter=1000 for convergence

### 2. Decision Tree Classifier
A tree-based model that recursively splits features to minimize impurity and make hierarchical decisions.
- **Strengths**: Interpretable, non-parametric, handles non-linear relationships
- **Configuration**: max_depth=10 to prevent overfitting

### 3. K-Nearest Neighbors (KNN)
A distance-based classifier that assigns class based on majority vote of k nearest neighbors.
- **Strengths**: Simple, non-parametric, no training phase
- **Important**: Requires scaled features; distance metrics are scale-sensitive
- **Configuration**: n_neighbors=5 (reasonable default)

### 4. Gaussian Naive Bayes
A probabilistic classifier based on Bayes theorem assuming feature independence and Gaussian distribution.
- **Strengths**: Fast, low variance, works well with small datasets
- **Note**: Input must be dense; preprocessing converts sparse matrices accordingly

### 5. Random Forest Classifier
An ensemble of decision trees that bootstrap samples and feature subsets for robust predictions.
- **Strengths**: Reduces overfitting, handles non-linear relationships, provides feature importance
- **Configuration**: n_estimators=100, n_jobs=-1 (parallel processing)

---

## 5. Evaluation Metrics

### 1. **Accuracy**
Proportion of correct predictions among total predictions.
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **When to use**: Balanced datasets
- **Limitation**: Misleading with imbalanced classes

### 2. **AUC (Area Under the ROC Curve)**
Measures trade-off between true positive rate and false positive rate across classification thresholds.
$$\text{AUC} = \int_{0}^{1} TPR(t) \, dFPR(t)$$
- **Range**: 0 to 1 (1 = perfect classifier)
- **Advantage**: Threshold-independent, handles class imbalance

### 3. **Precision**
Proportion of positive predictions that are actually correct.
$$\text{Precision} = \frac{TP}{TP + FP}$$
- **Use case**: Minimize false positives (e.g., avoid wasting retention offers)

### 4. **Recall (Sensitivity)**
Proportion of actual positives correctly identified by the model.
$$\text{Recall} = \frac{TP}{TP + FN}$$
- **Critical for churn**: Failing to identify churning customers is costly in retention scenarios
- **Focus metric**: High recall ensures we don't miss at-risk customers

### 5. **F1 Score**
Harmonic mean of precision and recall, balancing both metrics.
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Use case**: When both false positives and false negatives are costly

### 6. **Matthews Correlation Coefficient (MCC)**
Correlation coefficient between predicted and actual values; balanced measure for binary classification.
$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
- **Range**: -1 to 1 (-1 = inverse, 0 = random, 1 = perfect)
- **Advantage**: Considers all four confusion matrix components

---

## 6. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |

**Instructions**: Fill this table using actual model output from the Jupyter notebook. Do not fabricate results.

---

## 7. Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Random Forest | |
| Overall Winner | |

**Guidelines for observations** (based on actual results):
- Discuss each metric and its interpretation
- Identify strengths and weaknesses
- Assess overfitting/underfitting tendency
- Evaluate suitability for customer churn prediction
- **Special Focus on Recall**: Explain how well each model catches churning customers

---

## 8. Streamlit Application

The interactive Streamlit app (`app.py`) provides:

1. **CSV Upload**: Users can upload their customer data
2. **Model Selection**: Choose from 5 trained models
3. **Predictions**: Get churn predictions for uploaded customers
4. **Evaluation Metrics**: Display accuracy, AUC, precision, recall, F1, MCC
5. **Confusion Matrix**: Visualize TP, FP, TN, FN
6. **Classification Report**: (Optional) Detailed classification performance

**Features**:
- Clean, customized UI
- Validates uploaded CSV contains required feature columns
- If Churn column present: Calculate and display metrics
- If Churn column absent: Display predictions only with explanation

---

## 9. GitHub Repository

[Add repository URL here after pushing to GitHub]

---

## 10. Installation

**Step 1: Create Virtual Environment**
```bash
python -m venv venv
```

**Step 2: Activate Virtual Environment**
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 11. Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## Project Structure

```
telco-customer-churn/
│
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── test_data.csv              # Test subset for demo
│
├── model/
│   ├── logistic_regression.py  # Logistic Regression implementation
│   ├── decision_tree.py        # Decision Tree implementation
│   ├── knn.py                  # KNN implementation
│   ├── naive_bayes.py          # Gaussian Naive Bayes implementation
│   └── random_forest.py        # Random Forest implementation
│
├── notebooks/
│   └── classification_analysis.ipynb  # Main analysis & model training
│
└── data/
    └── README.md              # Data directory documentation
```

---

**Dataset Source**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
