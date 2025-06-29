import pandas as pd
import numpy as np
import re

print("Starting enhanced data preparation for Illuminate data...")

# ---- LOAD THE RAW DATA ----
try:
    illuminate_df = pd.read_csv("Data/IlluminateCombined.csv", encoding='cp1252', low_memory=False)
    print(f"Loaded raw IlluminateData with {len(illuminate_df)} rows.")

except FileNotFoundError:
    print("Error: 'Data/IlluminateCombined.csv' not found. Please ensure the file is in the 'Data' directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# ---- STANDARDIZE AND CLEAN DATA ----

# Clean column names immediately
illuminate_df.columns = illuminate_df.columns.str.strip()
print("- Cleaned column names.")

# Rename the student identifier column for consistency
if 'Mask_StudentPersonkey' in illuminate_df.columns:
    illuminate_df.rename(columns={'Mask_StudentPersonkey': 'student_id'}, inplace=True)
    print("- Renamed student identifier column to 'student_id'.")

# Standardize text columns to uppercase for consistent grouping and joining
for col in ['title', 'Department', 'Standard_Subject']:
    if col in illuminate_df.columns:
        illuminate_df[col] = illuminate_df[col].str.upper().str.strip()
print("- Standardized 'title', 'Department', and 'Standard_Subject' columns to uppercase.")

# Use 'Department' as the primary subject column, and fill missing values with 'title'
if 'Department' in illuminate_df.columns and 'title' in illuminate_df.columns:
    illuminate_df['Department'].fillna(illuminate_df['title'], inplace=True)
    print("- Consolidated subject information into the 'Department' column.")


# ---- FEATURE ENGINEERING: CREATE A STANDARD GRADE LEVEL ----
def extract_numeric_grade(grade_text):
    """Extracts a numeric grade from various text formats."""
    if pd.isna(grade_text):
        return None
    # Look for patterns like 'Grade 6', '6th Grade', or just a number '09'
    match = re.search(r'\d+', str(grade_text))
    if match:
        return int(match.group(0))
    return None

# Apply the function to create a unified, numeric grade level column
if 'categorytitle' in illuminate_df.columns:
    illuminate_df['StandardGradeLevel'] = illuminate_df['categorytitle'].apply(extract_numeric_grade)
    print("- Created a standardized 'StandardGradeLevel' column from 'categorytitle'.")
else:
    illuminate_df['StandardGradeLevel'] = np.nan


# ---- FILTER DATA ----

# Filter for High School Grade Levels
high_school_levels = [9, 10, 11, 12] # Use numeric grades now
original_rows = len(illuminate_df)
illuminate_df_filtered = illuminate_df[illuminate_df['StandardGradeLevel'].isin(high_school_levels)].copy()
print(f"- Filtered for high school grade levels (9-12). Kept {len(illuminate_df_filtered)} rows from {original_rows}.")


# ---- FINALIZE DATATYPES AND DROP NULLS ----

# Ensure data types are correct for key columns
illuminate_df_filtered['student_id'] = illuminate_df_filtered['student_id'].astype(str)
illuminate_df_filtered['Response_percent_correct'] = pd.to_numeric(illuminate_df_filtered['Response_percent_correct'], errors='coerce')
illuminate_df_filtered['responsedatevalue'] = pd.to_datetime(illuminate_df_filtered['responsedatevalue'], errors='coerce')

# Drop rows where essential information is missing AFTER filtering
illuminate_df_filtered.dropna(
    subset=['student_id', 'Response_percent_correct', 'Department', 'responsedatevalue', 'StandardGradeLevel'],
    inplace=True
)
print("- Dropped rows with missing essential data.")


# ---- SELECT AND SAVE THE CLEANED DATA ----
# Keep only the columns that will be needed in the Streamlit app
final_columns = [
    'student_id',
    'Department',
    'StandardGradeLevel',
    'Response_percent_correct',
    'responsedatevalue',
    'condition',
    'StandardStateNumber'
]
# Ensure all selected columns exist before trying to select them
final_columns_exist = [col for col in final_columns if col in illuminate_df_filtered.columns]
cleaned_df = illuminate_df_filtered[final_columns_exist]

output_filename = "Data/cleaned_illuminate.csv"
cleaned_df.to_csv(output_filename, index=False)

print("\n----------------------------------------------------")
print(f"✅ Enhanced data preparation complete!")
print(f"Cleaned Illuminate data saved to: {output_filename}")
print(f"The final dataset has {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns.")
print("\nFinal Data Preview:")
print(cleaned_df.head())

