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

sg2025 = pd.read_csv("data\raw\Student Teacher Grade 2025.csv", dtype=dtype_fix, low_memory=False)
sg2024 = pd.read_csv('data\raw\Student Teacher Grade 2024.csv', dtype=dtype_fix, low_memory=False)
sg2023 = pd.read_csv('data\raw\Student Teacher Grade 2023.csv', dtype=dtype_fix, low_memory=False)
sg2022 = pd.read_csv('data\raw\Student Teacher Grade 2022.csv', dtype=dtype_fix, low_memory=False)
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

illuminate2022 = pd.read_csv('data\raw\IlluminateData2022.csv',encoding="cp1252", low_memory=False)
illuminate2023 = pd.read_csv('data\raw\IlluminateData2023.csv',encoding="cp1252", low_memory=False)
illuminate2024 = pd.read_csv('data\raw\IlluminateData2024.csv',encoding="cp1252", low_memory=False)
illuminate2025 = pd.read_csv('data\raw\IlluminateData2025.csv',encoding="cp1252", low_memory=False)
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
illuminate.to_csv('data\processed\IlluminateCombined.csv', index=False)