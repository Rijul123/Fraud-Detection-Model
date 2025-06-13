# Fraud-Detection-Analysis


Objective & ScopeThe goal of this project is to detect fraudulent transactions within a large dataset of financial records. We define the problem as a binary classification task:
Positive class (fraud): Transactions labeled as fraudulent.
Negative class (legitimate): All other transactions.
The dataset consists of 1,048,575 transactions, with only 1,142 fraud cases (0.1% prevalence). We must address severe class imbalance and aim to maximize detection (recall) while controlling false positive rates.
MethodologyData Loading & Cleaning
Read raw data from CSV using pandas.
Handle missing values and standardize numeric features.
Train/Test Split
Stratified split: 80% training, 10% validation, 10% testing (train_test_split with stratify).
Class Imbalance Handling
SMOTE Oversampling: Synthetic Minority Over-sampling Technique to boost minority class samples in training data.
NearMiss Undersampling: Reduce majority class samples by selecting those closest to minority instances.
Model Selection & Training
Random Forest Classifier (100 trees, default settings).
Logistic Regression (L2 penalty, balanced class weights).
Each model trained on (a) original data, (b) SMOTE-oversampled data, (c) NearMiss-undersampled data.
Evaluation Metrics
Accuracy, Precision, Recall, F1 Score.
ROC Curve & AUC.
Confusion Matrix visualization.
Results & AnalysisBaseline (No Resampling)
Random Forest: Accuracy 99.97%, Recall 75.6%, Precision 84.2%, F1 0.854.
Logistic Regression: Accuracy 99.93%, Recall 92.3%, Precision 2.8%, F1 0.055.
SMOTE Oversampling
Random Forest: Recall improved to 83.4%, Precision dropped to 53.5% (more false positives).
Logistic Regression: Recall 95.1%, Precision 3.4%, F1 0.065.
NearMiss Undersampling
Random Forest: Recall 100%, Precision 0.7%, F1 0.014 (too many false positives).
Logistic Regression: Recall 100%, Precision 0.4%, F1 0.008.
ROC AUC
Random Forest consistently achieved AUC > 0.99 across all scenarios.
Logistic Regression AUC ~0.85 without resampling, dropping slightly when resampled.
Confusion Matrices
SMOTE reduced false negatives but increased false positives notably.
NearMiss eliminated false negatives but produced overwhelming false positives.
Conclusions & RecommendationsBest Model: Random Forest with SMOTE oversampling balances detection ability and manageable false positives. It achieves a good tradeoff with Recall ~83% and Precision ~53%.
Logistic Regression performs poorly on precision despite high recall, making it unsuitable for deployment.
Undersampling leads to impractical false positive rates.
Recommended Next Steps:
Integrate threshold tuning (e.g., adjust classification threshold) to further balance precision/recall.
Explore ensemble methods combining multiple classifiers.
Investigate feature engineering to improve separability.
Deploy pipeline as a modular package, incorporate parameter configuration and logging.
