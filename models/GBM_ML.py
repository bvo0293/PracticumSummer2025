import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# DATA PREPARATION
# =======================
print("Starting data preparation for the enhanced GBM model...")

# ---- LOAD ALL DATA ----
try:
    grades_df = pd.read_csv("data/processed/Final_DF.csv" , low_memory=False)
    courses_df = pd.read_csv("data/raw/Courses.csv")
    print("Loaded Final_DF.csv and Courses.csv.")
except FileNotFoundError as e:
    print(f"Error: A required data file was not found. Please check your 'Data/' directory. Missing file: {e.filename}")
    exit()

grades_df['student_id'] = grades_df['student_id'].astype(str)
grades_df['Mark'] = pd.to_numeric(grades_df['Mark'], errors='coerce')
grades_df.dropna(subset=['Mark'], inplace=True)
grades_df['DepartmentDesc'] = grades_df['DepartmentDesc'].str.upper().str.strip()

# ---- DEFINE THE TARGET VARIABLE: "SUCCESS" ----
success_threshold = 80
grades_df['Success'] = np.where(grades_df['Mark'] >= success_threshold, 1, 0)
print(f"Defined target variable 'Success' using a threshold of {success_threshold}.")

# ---- ENHANCED FEATURE ENGINEERING---
print("Starting enhanced feature engineering...")

# -- Student-Level Features --
grades_df['student_avg_mark_overall'] = grades_df.groupby('student_id')['Mark'].transform('mean')
grades_df['student_avg_mark_in_subject'] = grades_df.groupby(['student_id', 'DepartmentDesc'])['Mark'].transform('mean')
print("- Calculated student-level features.")


# -- Granular Course-Level Feature --
def assign_granular_difficulty_rank(row):
    """
    Creates a more detailed difficulty rank by combining CourseLevelDesc and HonorsDesc.
    """
    base_rank = 1.0  # Default rank
    bonus_rank = 0.0  # Bonus for honors, AP, etc.

    # Determine the base rank from 'CourseLevelDesc'
    level_desc = row['CourseLevelDesc']
    if pd.notna(level_desc) and isinstance(level_desc, str):
        match = re.search(r'level (\d+)', level_desc.lower())
        if match:
            base_rank = float(match.group(1))

    # Determine the bonus rank from 'HonorsDesc'.
    honors_desc = row['HonorsDesc']
    if pd.notna(honors_desc) and isinstance(honors_desc, str):
        honors_desc = honors_desc.lower()
        if 'ib' in honors_desc:
            bonus_rank = 0.6
        elif 'ap' in honors_desc:
            bonus_rank = 0.6
        elif 'honors' in honors_desc or 'hr' in honors_desc:
            bonus_rank = 0.2
        elif 'dual' in honors_desc:
            bonus_rank = 0.4

    # The final rank is the sum of the base and the bonus
    return base_rank + bonus_rank


# Apply the new, more granular ranking function
courses_df['course_difficulty_rank'] = courses_df.apply(assign_granular_difficulty_rank, axis=1)
print("- Calculated **new granular course difficulty rank** using both CourseLevelDesc and HonorsDesc.")

# ---- MERGE DATASETS ----
print("Merging datasets...")
course_features_df = courses_df[['siscourseidentifier', 'course_difficulty_rank']]
training_df = pd.merge(
    grades_df,
    course_features_df,
    left_on='CourseNumber',
    right_on='siscourseidentifier',
    how='left'
)

# ---- FINAL CLEANING AND PREPARATION ----
print("Final cleaning...")
training_df['course_difficulty_rank'].fillna(1.0, inplace=True)
final_columns = [
    'student_avg_mark_overall',
    'student_avg_mark_in_subject',
    'course_difficulty_rank',  # Now using the more granular rank
    'DepartmentDesc',
    'Success'
]
training_df = training_df[final_columns].dropna()
training_df = pd.get_dummies(training_df, columns=['DepartmentDesc'], prefix='dept')

# ---- SAVE THE NEW TRAINING DATASET ----
output_filename = "data/processed/training_data_v2.csv"
training_df.to_csv(output_filename, index=False)
print(f"✅ Enhanced data preparation complete! New training data saved to: {output_filename}\n")

# ==============================================================================
# GBM MODEL TRAINING AND EVALUATION
# ==============================================================================
print("----------------------------------------------------")
print("Starting GBM model training with enhanced features...")

# ---- LOAD THE PREPARED DATA ----
data = pd.read_csv(output_filename)

# ---- DEFINE FEATURES (X) AND TARGET (y) ----
y = data['Success']
X = data.drop('Success', axis=1)

# ---- SPLIT DATA ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples.")

# ---- INITIALIZE AND TRAIN THE XGBOOST MODEL ----
print("\nTraining XGBoost model on new features...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# ---- EVALUATE THE MODEL ----
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# **NEW**: Full Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Successful (0)', 'Successful (1)']))

# **NEW**: Feature Importance Plot
print("\nGenerating Feature Importance Plot...")
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10) # Display top 10

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
plt.title('Top 10 Feature Importances for Success Prediction Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
# Save the plot to a file
plt.savefig('feature_importance.png')
print("✅ Feature importance plot saved as 'feature_importance.png'")

# ---- SAVE THE NEWLY TRAINED MODEL ----
model_path = 'models/student_success_model.pkl'

joblib.dump(model, model_path)
print("\n----------------------------------------------------")
print(f"✅ New model saved to: {model_path}")
