from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

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

# Cache for the DataFrame
data_cache: pd.DataFrame | None = None

@app.on_event("startup")
def load_data():
    global data_cache
    print("Loading CSV")
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
    data_cache = pd.read_csv('../data/processed/StudentTeacherGradeCombined.csv', dtype=dtype_fix, low_memory=False)
    data_cache = data_cache[data_cache['AttemptedCredit'].notnull()]
    data_cache = data_cache[data_cache['Mark'].notnull()]
    print("CSV loaded and cached.")

@app.post("/api/submit")
async def submit_student_data(data: dict):
    global data_cache
    student_id = data.get("studentId")

    student_data = data_cache[data_cache["mask_studentpersonkey"] == str(student_id)]
    student_data = student_data[
        ['SchoolYear', 'GradeLevel', 'CourseDesc', 'MarkingPeriodCode', 'Mark',
         'EarnedCredit', 'SchoolDetailFCSId', 'DepartmentDesc']
    ]
    print(student_data)

    return {
        "student_data": student_data.to_dict(orient="records")
    }