# Fraud Detection Pipeline 

Objectives, methods, results, and recommendations for an end-to-end fraud detection pipeline.

---

## 📑 Table of Contents

1. [Objective & Scope](#objective--scope)
2. [Methodology](#methodology)
3. [Results & Analysis](#results--analysis)
4. [Conclusions & Recommendations](#conclusions--recommendations)

---

## 🧐 Objective & Scope

- **Problem:** Identify fraudulent transactions among 1,048,575 records.
- **Classes:**
  - **Fraudulent (positive):** 1,142 cases (0.1%).
  - **Legitimate (negative):** All other transactions.
- **Goal:** Maximize recall (detect as many frauds as possible) while controlling false positives.

---

## 🛠️ Methodology

1. **Data Loading & Cleaning**
   - Import CSV data with `pandas`.
   - Handle missing values and normalize numerical features.

2. **Train/Validation/Test Split**
   - Stratified split: 80% train, 10% validation, 10% test.

3. **Class Imbalance Handling**
   - **SMOTE Oversampling:** Generate synthetic minority samples.
   - **NearMiss Undersampling:** Select nearest majority samples to balance classes.

4. **Model Training**
   - **Random Forest Classifier** (100 trees).
   - **Logistic Regression** (L2 penalty, class weights balanced).
   - Trained under three scenarios: original data, SMOTE, NearMiss.

5. **Evaluation Metrics**
   - **Accuracy**, **Precision**, **Recall**, **F1 Score**.
   - **ROC Curve** & **AUC**.
   - **Confusion Matrix** visualizations.

---

## 📊 Results & Analysis

| Scenario             | Model             | Accuracy | Recall  | Precision | F1 Score |
|----------------------|-------------------|----------|---------|-----------|----------|
| No Resampling        | Random Forest     | 99.97%   | 75.6%   | 84.2%     | 0.854    |
|                      | Logistic Reg.     | 99.93%   | 92.3%   | 2.8%      | 0.055    |
| SMOTE Oversampling   | Random Forest     | 99.95%   | 83.4%   | 53.5%     | 0.647    |
|                      | Logistic Reg.     | 99.90%   | 95.1%   | 3.4%      | 0.065    |
| NearMiss Undersampling | Random Forest   | 99.30%   | 100%    | 0.7%      | 0.014    |
|                      | Logistic Reg.     | 99.10%   | 100%    | 0.4%      | 0.008    |

- **ROC AUC:** RF > 0.99; LR ~ 0.85.
- **Confusion Matrices:** SMOTE reduces false negatives; NearMiss eliminates false negatives but floods false positives.

---

## 💡 Conclusions & Recommendations

- **Recommended Model:** Random Forest + SMOTE—best balance of recall (~83%) and precision (~53%).
- **Logistic Regression:** High recall but unacceptable precision (<5%).
- **NearMiss Undersampling:** Eliminates missed fraud but unmanageable false positives.

### Next Steps

- **Threshold Optimization:** Adjust classification thresholds to tune precision/recall.
- **Ensemble Methods:** Combine multiple models for robust predictions.
- **Feature Engineering:** Derive new features to improve separability.
- **Pipeline Packaging:** Modularize script, add configuration files, logging, and unit tests.

---

_End of report contents._

