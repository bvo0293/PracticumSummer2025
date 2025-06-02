import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Load Data ----
@st.cache_data
def load_data():
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
    sg = pd.read_csv('Data/StudentTeacherGradeCombined.csv', dtype=dtype_fix, low_memory=False)
    # remove rows
    sg = sg[sg['AttemptedCredit'].notnull()]
    sg = sg[sg['Mark'].notnull()]
    return sg

# ---- Display Student Performance Function ----
def display_student_performance(df, student_id, department=None):
    student_df = df[df['mask_studentpersonkey'] == str(student_id)].copy()

    if student_df.empty:
        st.warning(f"No data found for student ID: {student_id}")
        return

    if department:
        student_df = student_df[student_df['DepartmentDesc'].str.lower() == department.lower()]
        if student_df.empty:
            st.warning(f"No data found for student ID: {student_id} in department: {department}")
            return

    student_df = student_df[
        ['SchoolYear', 'GradeLevel', 'CourseDesc', 'MarkingPeriodCode', 'Mark',
         'EarnedCredit', 'SchoolDetailFCSId', 'DepartmentDesc']
    ].sort_values(by=['SchoolYear', 'MarkingPeriodCode'])

    student_df = student_df.rename(columns={
        'SchoolYear': 'Year',
        'GradeLevel': 'GradeLevel',
        'CourseDesc': 'Course',
        'MarkingPeriodCode': 'Period',
        'Mark': 'Mark',
        'EarnedCredit': 'Credit',
        'SchoolDetailFCSId': 'School ID',
        'DepartmentDesc': 'Department'
    })

    student_df['Mark'] = pd.to_numeric(student_df['Mark'], errors='coerce')
    student_df['Timeline'] = student_df['Year'].astype(str) + '-' + student_df['Period'].astype(str)

    # Remove Timeline column and reset index before display
    display_df = student_df.drop(columns=['Timeline']).reset_index(drop=True)

    st.subheader("Student Performance Table")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    avg_mark = (
        student_df.groupby(['Department', 'Timeline'])['Mark']
        .mean()
        .unstack(level=0)
        .sort_index()
    )

    avg_mark_filled = avg_mark.ffill()

    total_credits = (
        student_df.groupby('Department')['Credit']
        .sum()
        .sort_values(ascending=False)
    )

    st.subheader("Total Credits Earned by Department")
    for dept, credits in total_credits.items():
        st.write(f"- **{dept}**: {credits:.2f} credits")

    focus_departments = ['LANGUAGE ARTS', 'MATH', 'SCIENCE']
    columns_to_plot = [dept for dept in focus_departments if dept in avg_mark_filled.columns]

    if not columns_to_plot:
        st.info("No data available for Language Arts, Math, or Science departments.")
        return

    recommendations = []
    for dept in columns_to_plot:
        marks = avg_mark_filled[dept].dropna()
        if len(marks) < 2:
            recommendations.append(f"Not enough data to analyze trend for {dept}.")
            continue

        x = np.arange(len(marks))
        y = marks.values
        slope = np.polyfit(x, y, 1)[0]
        threshold = 0.05

        if slope > threshold:
            recommendations.append(
                f"Your {dept} marks have been improving over time. "
                "Please consider enrolling in advanced courses listed below to continue your growth."
                "\n- Recommended Course 1: [Placeholder]"
                "\n- Recommended Course 2: [Placeholder]"
            )
        elif slope < -threshold:
            recommendations.append(
                f"Your {dept} marks have shown a downward trend. "
                "We recommend seeking extra help or tutoring and focusing on foundational courses listed below."
                "\n- Recommended Course 1: [Placeholder]"
                "\n- Recommended Course 2: [Placeholder]"
            )
        else:
            recommendations.append(
                f"Your {dept} marks have been consistent. "
                "Maintain your progress by exploring enrichment opportunities or elective courses listed below."
                "\n- Recommended Course 1: [Placeholder]"
                "\n- Recommended Course 2: [Placeholder]"
            )

    st.subheader("Performance Summary")
    for rec in recommendations:
        st.markdown("- " + rec)

    st.subheader("Average Mark in Core Subjects Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_mark_filled[columns_to_plot].plot(kind='line', marker='o', ax=ax)
    ax.set_ylabel("Average Mark")
    ax.set_xlabel("Timeline (Year-Semester)")
    plt.xticks(rotation=45)
    ax.grid(True)
    ax.set_title(f"Student ID: {student_id}")
    plt.tight_layout()
    st.pyplot(fig)

# ---- Streamlit UI ----
st.title("📊 Student Academic Performance Dashboard")

# Load data
df = load_data()

# User inputs
student_id = st.text_input("Enter Student ID:")
department = st.text_input("Filter by Department (optional):")

# Display results if ID is entered
if student_id:
    display_student_performance(df, student_id, department if department else None)
