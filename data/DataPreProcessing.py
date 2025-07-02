# Load Libraries
import pandas as pd
import numpy as np
import re
pd.set_option('display.max_columns', None) 

# Load data
dtype_fix = {
    'GradeLevel': 'category',
    'CourseNumber': 'str',
    'DOECourseNumber': 'str',
    'StartPeriodCode': 'str',
    'EndPeriodCode': 'str',
    'AttemptedCreditOverrideReason': 'str',
    'EarnedCreditOverrideReason': 'str',
    'mask_studentpersonkey': 'str'
}

print("Creating StudentTeacherGradeCombined.csv !!!")
sg2025 = pd.read_csv("data/raw/Student Teacher Grade 2025.csv", dtype=dtype_fix, low_memory=False)
sg2024 = pd.read_csv('data/raw/Student Teacher Grade 2024.csv', dtype=dtype_fix, low_memory=False)
sg2023 = pd.read_csv('data/raw/Student Teacher Grade 2023.csv', dtype=dtype_fix, low_memory=False)
sg2022 = pd.read_csv('data/raw/Student Teacher Grade 2022.csv', dtype=dtype_fix, low_memory=False)
# Combine files
sg = pd.concat([sg2022, sg2023, sg2024, sg2025], ignore_index=True)

# Remove unused columns
colums_to_remove = ['RoomNumber','SectionIdentifier','SubjectAreaCreditCode',
                    'SubjectAreaCreditDesc','AttemptedCreditOverrideFlag',
                    'AttemptedCreditOverrideReason','EarnedCreditOverrideFlag',
                    'EarnedCreditOverrideReason','InstructionalSettingCode']
sg = sg.drop(columns=[col for col in colums_to_remove if col in sg.columns])

# Data cleaning

def remove_html_tags(text):
    if isinstance(text, str):
        return re.sub(r'<.*?>', '', text)
    return text

def truncate_text(text, length=40):
    if isinstance(text, str) and len(text) > length:
        return text[:length-3] + "..."
    return text

sg['CourseDesc'] = sg['CourseDesc'].apply(remove_html_tags)
sg['CourseDesc'] = sg['CourseDesc'].apply(truncate_text)

sg.to_csv('data\processed\StudentTeacherGradeCombined.csv', index=False)
print("StudentTeacherGradeCombined.csv created")

print("Creating IlluminateCombined.csv!!!")
illuminate2022 = pd.read_csv('data/raw/IlluminateData2022.csv',encoding="cp1252", low_memory=False)
illuminate2023 = pd.read_csv('data/raw/IlluminateData2023.csv',encoding="cp1252", low_memory=False)
illuminate2024 = pd.read_csv('data/raw/IlluminateData2024.csv',encoding="cp1252", low_memory=False)
illuminate2025 = pd.read_csv('data/raw/IlluminateData2025.csv',encoding="cp1252", low_memory=False)
illuminate = pd.concat([illuminate2022, illuminate2023, illuminate2024, illuminate2025], ignore_index=True)

def clean_title_column(df):
    # Define a mapping of common variations to standardized titles
    title_mapping = {
        "language arts": "Language Arts",
        "Language arts": "Language Arts",
        "LANGUAGE ARTS": "Language Arts",
        "Language Arts": "Language Arts",
        "math": "Math",
        "MATH": "Math",
        "Math": "Math",
        "social studies": "Social Studies",
        "Social Studies": "Social Studies",
        "SCIENCE": "Science",
        "science": "Science",
        "Science": "Science",
        # Add more mappings as needed
    }

    # Normalize by stripping whitespace and applying lowercase
    df['title'] = df['title'].str.strip()
    df['Department'] = df['Department'].str.strip()
    # Replace using mapping (case-insensitive)
    df['title'] = df['title'].apply(lambda x: title_mapping.get(x, x.title()))
    df['Department'] = df['Department'].apply(lambda x: title_mapping.get(x, x.title()))
    return df
illuminate = clean_title_column(illuminate)


def clean_grade_levels(df):
    def format_grade(x):
        # Convert to string first
        x_str = str(x).strip()

        # Handle Kindergarten labels
        if x_str.upper() in ['KK', 'KINDERGARTEN', 'K']:
            return 'K'

        # If float or numeric string, convert to int then format as two-digit string
        try:
            x_float = float(x_str)
            x_int = int(x_float)
            return f"{x_int:02d}"
        except ValueError:
            # If cannot convert, return original string (or you can return None)
            return x_str

    df['GradeLevelDuringUnitTest'] = df['GradeLevelDuringUnitTest'].apply(format_grade)
    df['AssessmentGradeLevel'] = df['AssessmentGradeLevel'].apply(format_grade)
    return df
illuminate = clean_grade_levels(illuminate)
illuminate.to_csv('data/processed/IlluminateCombined.csv', index=False)
print("IlluminateCombined.csv created")
# --------------------------------------------

print("Creating Final_DF.csv !!!")
grad_summary = pd.read_csv("data/raw/GraduationAreaSummary.csv", low_memory=False)

# Courses.csv (static course catalog)
courses = pd.read_csv("data/raw/Courses.csv", low_memory=False)

all_grades = pd.read_csv("data/processed/StudentTeacherGradeCombined.csv", low_memory=False)
def categorize_grade_level(grade):
    if grade in ['K', '01', '02', '03', '04', '05']:
        return 'Elementary'
    elif grade in ['06', '07', '08']:
        return 'Middle'
    elif grade in ['09', '10', '11', '12']:
        return 'High'
    else:
        return 'Unknown'

all_grades['SchoolLevel'] = all_grades['GradeLevel'].apply(categorize_grade_level)

all_grades = all_grades.rename(columns={"mask_studentpersonkey": "student_id"})

# Filter only High School students with credit data
hs_grades = all_grades[
    (all_grades["SchoolLevel"] == "High") &
    (all_grades["AttemptedCredit"].notna()) &
    (all_grades["EarnedCredit"] > 0)
]

# Make cleaned copies before modifying
hs_grades_cleaned = hs_grades.copy()
grad_summary_cleaned = grad_summary.copy()
hs_grades_cleaned.loc[:, "student_id"] = hs_grades_cleaned["student_id"].astype(str).str.replace(r"\.0$", "", regex=True)

# Rename and normalize student ID in the graduation summary
grad_summary_cleaned['student_id'] = grad_summary_cleaned['mask_studentpersonkey'].astype(str)
grad_summary_cleaned['student_id'] = grad_summary_cleaned['student_id'].str.replace(r"\\.0$", "", regex=True)
grad_summary_cleaned = grad_summary_cleaned.drop(columns=["mask_studentpersonkey"])


# Normalize student ID in the grades data
# The column is already named 'student_id'
hs_grades_cleaned['student_id'] = hs_grades_cleaned['student_id'].astype(str)
hs_grades_cleaned['student_id'] = hs_grades_cleaned['student_id'].str.replace(r"\\.0$", "", regex=True)


summary_ids = set(grad_summary_cleaned["student_id"])

# Get student IDs with course records in 2024-2025
active_2024_ids = set(
    hs_grades_cleaned[hs_grades_cleaned["SchoolYear"] == "2024-2025"]["student_id"]
)

# Find which summary students are active this year
active_summary_students = summary_ids & active_2024_ids
inactive_summary_students = summary_ids - active_2024_ids

grad_summary_filtered = grad_summary_cleaned[
    grad_summary_cleaned["student_id"].isin(active_2024_ids)
]

merged = grad_summary_filtered.merge(
    hs_grades_cleaned,
    on="student_id",
    how="left"
)

#  Keep relevant columns
credit_data = merged[["student_id", "SubjectArea", "AreaCreditStillNeeded"]].drop_duplicates()

# Pivot to get SubjectArea as columns and values as AreaCreditStillNeeded
credit_pivot = credit_data.pivot_table(
    index="student_id",
    columns="SubjectArea",
    values="AreaCreditStillNeeded",
    aggfunc="last"
).fillna(0)

# Clean column names
credit_pivot.columns = [f"CredStill_{col}" for col in credit_pivot.columns]
credit_pivot.reset_index(inplace=True)

# Merge back with hs_grades_cleaned to enrich further
final_df = hs_grades_cleaned.merge(credit_pivot, on="student_id", how="left")

final_df.to_csv("data/processed/Final_DF.csv"  , index=False)
print("Final_DF.csv !!! created")