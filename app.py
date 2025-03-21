# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")
# Load the dataset
df = pd.read_csv('TechBlitz DataScience Dataset.csv')

# Print column names to verify
print("Columns in dataset:", df.columns.tolist())

# Handling missing values - Assuming no missing values here as per your dataset
# If there were missing values, we would use: df.fillna(df.mean()) for numerical features

# Outlier Detection & Treatment using IQR (Interquartile Range)
def cap_outliers(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

print("Handling outliers...")
df = cap_outliers(df)

# Feature Scaling using Standardization (Z-score)
print("Scaling features...")
scaler = StandardScaler()
# Get numerical columns automatically
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Air Quality' in numerical_features:
    numerical_features.remove('Air Quality')
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the scaler for future use
print("Saving scaler...")
joblib.dump(scaler, 'scaler.pkl')

# Encoding target variable (Air Quality) to numeric values if it's not already numeric
if not pd.api.types.is_numeric_dtype(df['Air Quality']):
    quality_mapping = {'Good': 0, 'Moderate': 1, 'Poor': 2, 'Hazardous': 3}
    df['Air Quality'] = df['Air Quality'].map(quality_mapping)

# Split into features and target
X = df.drop(columns=['Air Quality'])
y = df['Air Quality']

# Train-test split (80% training, 20% test)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training and Evaluation with Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model evaluation
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Save the trained model using joblib
print("Saving Random Forest model...")
joblib.dump(rf_model, 'air_quality_model.pkl')

# Hyperparameter tuning for XGBoost using GridSearchCV
print("Training XGBoost model...")
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Define hyperparameters for Grid Search
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.3, 0.5]
}

xgb_model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
print("Best parameters from GridSearchCV:", grid_search.best_params_)

# Evaluate the best model
best_xgb_model = grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy after tuning: {accuracy_xgb * 100:.2f}%")

# Save the optimized model
print("Saving XGBoost model...")
joblib.dump(best_xgb_model, 'optimized_air_quality_model.pkl')

print("Saving column names...")
# Save the column names for future reference
joblib.dump(numerical_features, 'feature_names.pkl')

print("Training and saving completed successfully!")

# Visualization of Results
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix for Random Forest model
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=["Good", "Moderate", "Poor", "Hazardous"],
            yticklabels=["Good", "Moderate", "Poor", "Hazardous"])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the target variable for multiclass ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_pred_prob = rf_model.predict_proba(X_test)

# Calculate ROC curve and AUC for each class
fpr = {}; tpr = {}; roc_auc = {}
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(8, 6))
for i in range(4):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Feature importance for Random Forest model
feature_importances = rf_model.feature_importances_
features = X.columns

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.title('Feature Importance for Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
