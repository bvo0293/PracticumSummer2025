import pandas as pd
import numpy as np
import re

print("Starting data preparation for GBM model...")

# ---- LOAD ALL NECESSARY DATA ----
try:
    # Load the main historical grades data
    grades_df = pd.read_csv("Data/Final_DF.csv", low_memory=False)
    print(f"Loaded Final_DF.csv with {len(grades_df)} rows.")

    # Load the course catalog
    courses_df = pd.read_csv("Data/Courses.csv")
    print(f"Loaded Courses.csv with {len(courses_df)} rows.")

except FileNotFoundError as e:
    print(f"Error: A required data file was not found. Please check your 'Data/' directory. Missing file: {e.filename}")
    exit()

# ---- INITIAL CLEANING AND PREPARATION ----

# Clean grades data
grades_df['student_id'] = grades_df['student_id'].astype(str)
grades_df['Mark'] = pd.to_numeric(grades_df['Mark'], errors='coerce')
# We need a valid mark to determine success and for feature calculation
grades_df.dropna(subset=['Mark'], inplace=True)
# Standardize department names
grades_df['DepartmentDesc'] = grades_df['DepartmentDesc'].str.upper().str.strip()


# ---- DEFINE THE TARGET VARIABLE: "SUCCESS" ----
# We define "success" in a course as achieving a mark of 80 or higher.
# This creates a binary target (1 for success, 0 for not).
success_threshold = 80
grades_df['Success'] = np.where(grades_df['Mark'] >= success_threshold, 1, 0)
print(f"Defined target variable 'Success' using a threshold of {success_threshold}.")


# ---- FEATURE ENGINEERING ----
print("Starting feature engineering...")

# -- Student-Level Features --
# These features describe the student's historical performance.

# Calculate each student's overall average mark across all courses
# We use transform('mean') to broadcast the result to all rows for each student
grades_df['student_avg_mark_overall'] = grades_df.groupby('student_id')['Mark'].transform('mean')
print("- Calculated student's overall average mark.")

# Calculate each student's average mark per subject (department)
grades_df['student_avg_mark_in_subject'] = grades_df.groupby(['student_id', 'DepartmentDesc'])['Mark'].transform('mean')
print("- Calculated student's average mark in each subject.")

# -- Course-Level Features --
# These features describe the course itself.

def assign_difficulty_rank(row):
    """Assigns a numerical difficulty rank based on HonorsDesc."""
    honors_desc = row['HonorsDesc']
    if pd.isna(honors_desc): return 1
    honors_desc = honors_desc.lower()
    if 'ib' in honors_desc: return 4
    if 'ap' in honors_desc or 'dual' in honors_desc: return 3
    if 'honors' in honors_desc or 'hr' in honors_desc: return 2
    return 1

# Apply the difficulty ranking to the course catalog
courses_df['course_difficulty_rank'] = courses_df.apply(assign_difficulty_rank, axis=1)
print("- Calculated course difficulty rank from HonorsDesc.")


# ---- MERGE DATASETS INTO A SINGLE TRAINING DATAFRAME ----
print("Merging datasets...")

# Select the features we need from the course catalog
course_features_df = courses_df[['siscourseidentifier', 'course_difficulty_rank']]

# Merge the course features into the main grades dataframe
# 'CourseNumber' in grades_df corresponds to 'siscourseidentifier' in courses_df
training_df = pd.merge(
    grades_df,
    course_features_df,
    left_on='CourseNumber',
    right_on='siscourseidentifier',
    how='left'
)
print(f"Merged course features. Shape after merge: {training_df.shape}")


# ---- FINAL CLEANING AND PREPARATION FOR MODEL ----
print("Final cleaning and preparation...")

# Handle potential missing values from the merge
# If a course had no difficulty rank, we'll assign the default of 1 (Standard)
training_df['course_difficulty_rank'].fillna(1, inplace=True)

# Select only the columns needed for training
# These are our predictors (features) and the target variable
final_columns = [
    'student_avg_mark_overall',
    'student_avg_mark_in_subject',
    'course_difficulty_rank',
    'DepartmentDesc', # This will be one-hot encoded next
    'Success' # This is our target
]
training_df = training_df[final_columns]
training_df.dropna(inplace=True) # Drop any rows that still have missing values
print("- Selected final columns for the model.")

# --- One-Hot Encoding for Categorical Features ---
# ML models require all input to be numeric. We convert 'DepartmentDesc'
# into multiple columns of 0s and 1s.
training_df = pd.get_dummies(training_df, columns=['DepartmentDesc'], prefix='dept')
print("- Performed one-hot encoding on 'DepartmentDesc'.")


# ---- SAVE THE DATASET ----
output_filename = "training_data.csv"
training_df.to_csv(output_filename, index=False)

print("\n----------------------------------------------------")
print(f"✅ Data preparation complete!")
print(f"Final training data saved to: {output_filename}")
print(training_df.head())

