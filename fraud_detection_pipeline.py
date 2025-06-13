import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( precision_score, 
                             recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score,  ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/rijulregmi/Downloads/Dataset_project.csv'  )
# removing any null values, if they exxist
data_cleaned = data.dropna()

# splitting data into 60%, 20%, 20%
train_data, temp = train_test_split(
    data_cleaned, test_size=0.4, stratify=data_cleaned['isFraud'], random_state=42)
validation_data, test_data = train_test_split(
    temp, test_size=0.5, stratify=temp['isFraud'], random_state=42)

features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X_train = train_data[features]
y_train = train_data['isFraud']
X_val = validation_data[features]
y_val = validation_data['isFraud']

# encoding
X_train = pd.get_dummies(X_train, columns=['type'])
X_val = pd.get_dummies(X_val, columns=['type'])
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

# adding weights
classes = np.array([0, 1])
weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

# Define Models
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    false_negative_rate = 1 - recall
    false_alert_rate = 1 - precision
    f1 = f1_score(y_val, y_pred)
 
    
    print(f"\n{model_name} Performance on Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Negative Rate: {false_negative_rate:.4f}")
    print(f"False Alert Rate: {false_alert_rate:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Confusion Matrix Visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name}  Confusion Matrix")
    plt.show()
    
    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_val, y_val)
    plt.title(f"{model_name} ROC Curve")
    plt.show()
    
    # Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(model, X_val, y_val)
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.show()


print("Training Without Resampling")
forest_model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
train_and_evaluate_model(forest_model, X_train, y_train, X_val, y_val, "Random Forest")

log_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
train_and_evaluate_model(log_model, X_train, y_train, X_val, y_val, "Logistic Regression")

print("\n Training With Resampling")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

nearmiss = NearMiss()
X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(X_train, y_train)

print("Random Forest with SMOTE:")
train_and_evaluate_model(forest_model, X_train_smote, y_train_smote, X_val, y_val, "Random Forest (SMOTE)")

print("Random Forest with NearMiss:")
train_and_evaluate_model(forest_model, X_train_nearmiss, y_train_nearmiss, X_val, y_val, "Random Forest (NearMiss)")

print("Logistic Regression with SMOTE:")
train_and_evaluate_model(log_model, X_train_smote, y_train_smote, X_val, y_val, "Logistic Regression (SMOTE)")

print("Logistic Regression with NearMiss:")
train_and_evaluate_model(log_model, X_train_nearmiss, y_train_nearmiss, X_val, y_val, "Logistic Regression (NearMiss)")

print("\n Done!")
