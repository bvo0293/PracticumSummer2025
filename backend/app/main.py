from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

import joblib

from backend.services.helper import assign_granular_difficulty_rank, create_student_vectors, get_student_profile, generate_ml_recommendations, get_collaborative_recommendations

app = FastAPI()

# Frontend access for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentData(BaseModel):
    studentId: str
    department: str | None = None
    subjectArea: str | None = None

@app.on_event("startup")
def load_data():
    global grade_cache, courses_cache, grad_cache, model, model_features
    print("Loading CSV")
# -----------------------------------------------------
    grade_cache = pd.read_csv("data/processed/Final_DF.csv", low_memory=False)
    grade_cache.columns = grade_cache.columns.str.strip()
    grade_cache['Mark'] = pd.to_numeric(grade_cache['Mark'], errors='coerce')
    grade_cache['DepartmentDesc'] = grade_cache['DepartmentDesc'].str.upper().str.strip()

    courses_cache = pd.read_csv("data/raw/Courses.csv", low_memory=False)

    grad_cache = pd.read_csv("data/raw/GraduationAreaSummary.csv", low_memory=False)
    grad_cache = grad_cache.rename(columns={'mask_studentpersonkey': 'student_id'})

    model = joblib.load('models/student_success_model.pkl')
    training_cols_df = pd.read_csv("data/processed/training_data_v2.csv")
    model_features = training_cols_df.drop('Success', axis=1).columns.tolist()
# -----------------------------------------------------
    print("CSV loaded and cached.")

@app.post("/api/submit")
async def submit_student_data(data: StudentData):
    student_id = data.studentId
    global grade_cache, courses_cache, grad_cache, model, model_features

# -----------------------------------------------------
    student_data = grade_cache[grade_cache["student_id"] == int(student_id)]
    student_data['DepartmentDesc'] = student_data['DepartmentDesc'].str.upper().str.strip()

    student_data = student_data[
        ['SchoolYear', 'GradeLevel', 'CourseDesc', 'MarkingPeriodCode', 'Mark',
         'EarnedCredit', 'SchoolDetailFCSId', 'DepartmentDesc']
    ]
    student_data = student_data.rename(columns={
        'SchoolYear': 'Year', 'GradeLevel': 'Grade Level', 'CourseDesc': 'Course',
        'MarkingPeriodCode': 'Period', 'Mark': 'Mark', 'EarnedCredit': 'Credit',
        'SchoolDetailFCSId': 'School ID', 'DepartmentDesc': 'Department'
    })
    student_data = student_data.where(pd.notnull(student_data), None)

# -----------------------------------------------------
    total_credits = grade_cache[grade_cache["student_id"] == int(student_id)]
    total_credits = total_credits.groupby('DepartmentDesc')['EarnedCredit'].sum().sort_values(ascending=False)
    total_credits = total_credits.to_frame().T

# -----------------------------------------------------
    student_data['Timeline'] = student_data['Year'].astype(str) + '-P' + student_data['Period'].astype(str)
    avg_mark = student_data.groupby(['Department', 'Timeline'])['Mark'].mean().unstack(level=0)
    if not avg_mark.empty:
        try:
            avg_mark.index = pd.Categorical(avg_mark.index, categories=sorted(avg_mark.index.unique(),
                                                                                key=lambda x: (int(x.split('-P')[0]),
                                                                                                x.split('-P')[1])),
                                            ordered=True)
            avg_mark = avg_mark.sort_index()
        except Exception:
            avg_mark = avg_mark.sort_index()
        avg_mark_filled = avg_mark.ffill().bfill()
    else:
        avg_mark_filled = pd.DataFrame()
    
    if not avg_mark_filled.empty:
        
        fig, ax = plt.subplots(figsize=(22, 8))

        columns_to_plot = avg_mark_filled.columns.tolist()
        avg_mark_filled[columns_to_plot].plot(kind='line', marker='o', ax=ax)
        
        ax.set_ylabel("Average Mark")
        ax.set_xlabel("Timeline (Year-Period)")
        plt.xticks(rotation=45, ha="right")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Department", loc="upper right", bbox_to_anchor=(1.13, 1), fontsize="x-small")
        ax.set_title(f"Student ID: {student_id}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
# -----------------------------------------------------    
    departments_for_trend_analysis = avg_mark_filled.columns.tolist()

    trend_recommendations = {}
    if not avg_mark_filled.empty and departments_for_trend_analysis:
        for dept_to_analyze in departments_for_trend_analysis:
            marks = avg_mark_filled[dept_to_analyze].dropna()
            if len(marks) < 2:
                trend_recommendations[dept_to_analyze] = f"Not enough data points to analyze trend."
                continue

            x = np.arange(len(marks))
            y = marks.values
            slope = np.polyfit(x, y, 1)[0] if np.any(y != y[0]) else 0
            threshold = 0.05

            if slope > threshold:
                trend_recommendations[dept_to_analyze] =  f"Marks are improving (trend: {slope:.2f})."
            elif slope < -threshold:
                trend_recommendations[dept_to_analyze] =  f"Marks show a downward trend (trend: {slope:.2f})."
            else:
                trend_recommendations[dept_to_analyze] =  f"Marks are consistent (trend: {slope:.2f})."

# -----------------------------------------------------  
    student_vectors_df = create_student_vectors(grade_cache, courses_cache)
    target_student_id = int(student_id)
    collab_recs_df = get_collaborative_recommendations(target_student_id, student_vectors_df, grade_cache, courses_cache)

# -----------------------------------------------------  
    ml_recommendations_df = generate_ml_recommendations(student_id, grad_cache, grade_cache, courses_cache, model, model_features )
    ml_recommendations_df = ml_recommendations_df.where(pd.notnull(ml_recommendations_df), None)
    student_credit_needs = grad_cache[grad_cache['student_id'] == int(student_id)][['AreaCreditStillNeeded', 'SubjectArea']]
    student_credit_needs = student_credit_needs.where(pd.notnull(student_credit_needs), None)
# -----------------------------------------------------  
    return {
    "student_id" : data.studentId,
    "student_data": student_data.to_dict(orient="records"),
    "total_credits": total_credits.to_dict(orient="records"),
    "image" : img_base64,
    "trend" : [trend_recommendations],
    "collab_rec" : collab_recs_df.to_dict(orient="records"),
    "ml_rec": ml_recommendations_df.to_dict(orient="records"),
    "needed_credits": student_credit_needs.to_dict(orient="records")
    }




