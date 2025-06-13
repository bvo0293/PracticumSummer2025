import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ---- Configuration for Subject Mapping ----
SUBJECT_TO_DEPT_MAP = {
    'Math': ['MATH', 'MATHEMATICS'],
    'ELA': ['LANGUAGE ARTS', 'ENGLISH', 'ELA'],
    'Science': ['SCIENCE'],
    'Social Studies': ['SOCIAL STUDIES', 'HISTORY', 'SOCIAL SCIENCES'],
    'Health/ PersonalFitness': ['HEALTH', 'PHYSICAL EDUCATION', 'PERSONAL FITNESS', 'HEALTH EDUCATION'],
    'World Language/ FineArts/ CareerTech': [
        'WORLD LANGUAGES', 'SPANISH', 'FRENCH', 'GERMAN',
        'FINE ARTS', 'ART', 'MUSIC', 'THEATRE', 'DANCE',
        'CAREER TECHNICAL AND AGRICULTURAL EDUCATION', 'CAREER AND TECHNICAL EDUCATION', 'CTE', 'BUSINESS', 'TECHNOLOGY'
    ],
    'Electives': ['ELECTIVE COURSES']
}


# ---- Load Data ----
@st.cache_data
def load_data():
    """Loads all necessary dataframes for the application."""
    data_files = {
        "grades": "Final_DF.csv",
        "courses": "Data/Courses.csv",
        "grad_summary": "Data/GraduationAreaSummary.csv"
    }
    data = {}
    try:
        # Load main student grades data
        grades_df = pd.read_csv(data_files["grades"], low_memory=False)
        grades_df['student_id'] = grades_df['student_id'].astype(str)
        grades_df['Mark'] = pd.to_numeric(grades_df['Mark'], errors='coerce')
        if 'DepartmentDesc' in grades_df.columns:
            grades_df['DepartmentDesc'] = grades_df['DepartmentDesc'].str.upper().str.strip()
        data['grades'] = grades_df

        # Load courses data
        data['courses'] = pd.read_csv(data_files["courses"])

        # Load graduation summary data
        grad_summary_df = pd.read_csv(data_files["grad_summary"])
        grad_summary_df = grad_summary_df.rename(columns={'mask_studentpersonkey': 'student_id'})
        grad_summary_df['student_id'] = grad_summary_df['student_id'].astype(str)
        data['grad_summary'] = grad_summary_df

        return data

    except FileNotFoundError as e:
        st.error(
            f"Error: A required data file was not found. Please check your 'Data/' directory. Missing file: {e.filename}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None


# ---- Recommendation Engine Logic (from Jupyter Notebook) ----

def extract_grade_level_from_course(course_name):
    if not isinstance(course_name, str):
        return 0
    roman_map = {'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'I': 9, 'II': 10, 'III': 11}
    for numeral, grade in roman_map.items():
        if re.search(r'\\b' + numeral + r'\\b', course_name, re.IGNORECASE):
            return grade
    match = re.search(r'\\b(G|Grade\\s*)?([9]|10|11|12)\\b', course_name, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return 0


def assign_difficulty_rank(row):
    honors_desc = row['HonorsDesc']
    if pd.isna(honors_desc):
        return 1
    honors_desc = honors_desc.lower()
    if 'ib' in honors_desc:
        return 4
    if 'ap' in honors_desc or 'dual' in honors_desc:
        return 3
    if 'honors' in honors_desc or 'hr' in honors_desc:
        return 2
    return 1


def get_student_profile(student_id, grades_df):
    student_grades = grades_df[grades_df['student_id'] == student_id].copy()
    if student_grades.empty:
        return None
    student_grades['GradeLevel'] = pd.to_numeric(student_grades['GradeLevel'], errors='coerce')
    current_grade = student_grades.sort_values(by='SchoolYear', ascending=False)['GradeLevel'].iloc[0]
    valid_marks = student_grades[(student_grades['Mark'].notna()) & (student_grades['Mark'] > 0)]
    average_mark = valid_marks['Mark'].mean() if not valid_marks.empty else 75
    return {'grade_level': current_grade, 'average_mark': average_mark}


def generate_recommendations(student_id, summary_df, grades_df, courses_df):
    profile = get_student_profile(student_id, grades_df)
    if not profile:
        return pd.DataFrame()

    student_grade = profile['grade_level']
    student_avg_mark = profile['average_mark']

    credit_gaps = summary_df[(summary_df['student_id'] == student_id) & (summary_df['AreaCreditStillNeeded'] > 0)]
    if credit_gaps.empty:
        return pd.DataFrame()

    min_difficulty = 2 if student_avg_mark >= 90 else 1

    # Enhance the course catalog inside the function
    courses_enhanced = courses_df.copy()
    courses_enhanced['course_grade_level'] = courses_enhanced['coursename'].apply(extract_grade_level_from_course)
    courses_enhanced['difficulty_rank'] = courses_enhanced.apply(assign_difficulty_rank, axis=1)

    recommendations = []
    for _, gap in credit_gaps.iterrows():
        subject_area = gap['SubjectArea']
        # Find the department from our subject map
        department_list = SUBJECT_TO_DEPT_MAP.get(subject_area, [])
        if not department_list:
            continue

        possible_courses = courses_enhanced[courses_enhanced['DepartmentDesc'].isin(department_list)]
        taken_course_numbers = set(grades_df[grades_df['student_id'] == student_id]['CourseNumber'])

        for _, course in possible_courses.iterrows():
            if course['siscourseidentifier'] in taken_course_numbers:
                continue
            course_grade = course['course_grade_level']
            if course_grade != 0 and course_grade < student_grade:
                continue
            if course['difficulty_rank'] < min_difficulty:
                continue

            recommendations.append({
                'student_id': student_id,
                'SubjectArea': subject_area,
                'coursename': course['coursename'],
                'CourseId': course['siscourseidentifier']
            })

    return pd.DataFrame(recommendations)


# ---- Display Student Performance Function ----
def display_student_performance(all_data, student_id, department_filter=None, subject_of_interest=None):
    if not all_data:
        st.warning("Data could not be loaded. Cannot display performance.")
        return

    grades_df = all_data['grades']
    courses_df = all_data['courses']
    grad_summary_df = all_data['grad_summary']

    student_df_orig = grades_df[grades_df['student_id'] == str(student_id)].copy()

    if student_df_orig.empty:
        st.warning(f"No data found for student ID: {student_id}")
        return

    # ---- Generate Live Recommendations ----
    live_recommendations_df = generate_recommendations(student_id, grad_summary_df, grades_df, courses_df)

    # ---- Display Graduation Credit Requirements & Recommendations ----
    st.subheader("🎓 Graduation Credit Requirements & Course Recommendations")
    st.markdown("_Click on a subject area to see recommended courses._")

    student_credit_needs = grad_summary_df[grad_summary_df['student_id'] == str(student_id)]

    if not student_credit_needs.empty:
        for _, need in student_credit_needs.iterrows():
            credits_needed = need['AreaCreditStillNeeded']
            display_name = need['SubjectArea']
            if credits_needed > 0:
                with st.expander(f"**{display_name}**: {credits_needed:.1f} Credits Needed"):
                    if not live_recommendations_df.empty:
                        subject_recs = live_recommendations_df[live_recommendations_df['SubjectArea'] == display_name]
                        if not subject_recs.empty:
                            st.write("Recommended Courses:")
                            for _, row in subject_recs.iterrows():
                                st.markdown(f"- **{row['coursename']}** (Course ID: {row['CourseId']})")
                        else:
                            st.write("No specific course recommendations generated for this subject.")
                    else:
                        st.write("No recommendations generated for this student.")
            else:
                st.metric(label=f"{display_name} Needed", value="Complete ✔️")
    else:
        st.write("Could not retrieve graduation credit requirement data for this student.")

    st.markdown("---")

    # --- Performance Analysis Section ---
    student_df = student_df_orig.copy()

    # Apply filters
    current_filter_message = ""
    if department_filter:
        student_df = student_df[student_df['DepartmentDesc'].str.lower() == department_filter.lower()]
        current_filter_message = f"Filtering performance by Department: **{department_filter.upper()}**"
    elif subject_of_interest:
        # Filtering logic for subject_of_interest
        current_filter_message = f"Focusing on Subject: {subject_of_interest}"
        target_depts_upper = []
        if subject_of_interest in SUBJECT_TO_DEPT_MAP:
            target_depts_upper = [d.upper() for d in SUBJECT_TO_DEPT_MAP[subject_of_interest]]
            student_df = student_df[student_df['DepartmentDesc'].isin(target_depts_upper)]
        elif subject_of_interest == 'Electives':
            all_mapped_core_depts = set(d.upper() for d_list in SUBJECT_TO_DEPT_MAP.values() for d in d_list)
            all_mapped_core_depts.update([d.upper() for d in CORE_DEPARTMENTS_FOR_ELECTIVE_CALC])
            student_df = student_df[~student_df['DepartmentDesc'].isin(all_mapped_core_depts)]
            current_filter_message = "Focusing on Subject: Electives"

    if current_filter_message:
        st.info(current_filter_message)

    # Ensure required columns for performance analysis are present
    required_perf_cols = ['SchoolYear', 'GradeLevel', 'CourseDesc', 'MarkingPeriodCode', 'Mark', 'EarnedCredit',
                          'SchoolDetailFCSId', 'DepartmentDesc']
    if not all(col in student_df.columns for col in required_perf_cols):
        st.error("One or more required columns for performance analysis are missing.")
        return

    student_df = student_df[required_perf_cols].sort_values(by=['SchoolYear', 'MarkingPeriodCode'])
    student_df = student_df.rename(columns={
        'SchoolYear': 'Year', 'GradeLevel': 'Grade Level', 'CourseDesc': 'Course',
        'MarkingPeriodCode': 'Period', 'Mark': 'Mark', 'EarnedCredit': 'Credit',
        'SchoolDetailFCSId': 'School ID', 'DepartmentDesc': 'Department'
    })

    student_df['Mark'] = pd.to_numeric(student_df['Mark'], errors='coerce')
    student_df = student_df.dropna(subset=['Mark'])

    if student_df.empty:
        st.warning(f"No valid academic mark data found for the current filter to display performance details.")
        return

    student_df['Timeline'] = student_df['Year'].astype(str) + '-P' + student_df['Period'].astype(str)

    # ---- Display Table ----
    with st.expander("📖 Student Performance Table (Filtered)", expanded=False):
        st.dataframe(student_df.drop(columns=['Timeline']), use_container_width=True, hide_index=True)

    avg_mark = (
        student_df.groupby(['Department', 'Timeline'])['Mark']
        .mean()
        .unstack(level=0)
    )
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

    st.subheader("📊 Total Credits Earned by Department (Filtered)")
    total_credits = student_df.groupby('Department')['Credit'].sum().sort_values(ascending=False)
    if not total_credits.empty:
        cols = st.columns(3)
        for i, (dept, credits_val) in enumerate(total_credits.items()):
            cols[i % 3].metric(label=dept, value=f"{credits_val:.2f} credits")
    else:
        st.write("No credit data available for the current filter.")

    # ---- Performance Trend Summary ----
    st.subheader("💡 Performance Trend Summary (Filtered)")

    departments_for_trend_analysis = []
    if not avg_mark_filled.empty:
        if department_filter:
            departments_for_trend_analysis = [dept.upper() for dept in avg_mark_filled.columns if
                                              dept.lower() == department_filter.lower()]
        elif subject_of_interest:
            if subject_of_interest in SUBJECT_TO_DEPT_MAP:
                departments_for_trend_analysis = [
                    dept.upper() for dept in SUBJECT_TO_DEPT_MAP[subject_of_interest]
                    if dept.upper() in avg_mark_filled.columns
                ]
            elif subject_of_interest == 'Electives':
                all_mapped_core_depts = set(d.upper() for d_list in SUBJECT_TO_DEPT_MAP.values() for d in d_list)
                all_mapped_core_depts.update([d.upper() for d in CORE_DEPARTMENTS_FOR_ELECTIVE_CALC])
                departments_for_trend_analysis = [
                    dept.upper() for dept in avg_mark_filled.columns
                    if dept.upper() not in all_mapped_core_depts
                ]

        # **MODIFIED LOGIC**: If no specific filter is active, analyze all available departments.
        if not departments_for_trend_analysis:
            departments_for_trend_analysis = avg_mark_filled.columns.tolist()

    recommendations = []
    if not avg_mark_filled.empty and not departments_for_trend_analysis:
        st.markdown("- Could not determine specific departments for trend analysis based on selection.")
    elif avg_mark_filled.empty:
        st.markdown("- No mark data available for the current filter to generate trends or recommendations.")

    for dept_to_analyze in departments_for_trend_analysis:
        if dept_to_analyze in avg_mark_filled.columns:
            marks = avg_mark_filled[dept_to_analyze].dropna()
            if len(marks) < 2:
                recommendations.append(
                    f"**{dept_to_analyze}**: Not enough data points ({len(marks)}) to analyze trend reliably.")
                continue
            x = np.arange(len(marks))
            y = marks.values
            if np.all(y == y[0]):
                slope = 0
            else:
                try:
                    slope = np.polyfit(x, y, 1)[0]
                except (np.linalg.LinAlgError, ValueError) as e:
                    recommendations.append(
                        f"**{dept_to_analyze}**: Could not calculate trend due to data issues ({e}). Marks: {y.tolist()}")
                    continue
            threshold = 0.05
            if slope > threshold:
                recommendations.append(
                    f"**{dept_to_analyze}**: Marks have been improving (trend: {slope:.2f}). "
                    "Consider advanced courses to continue this growth."
                )
            elif slope < -threshold:
                recommendations.append(
                    f"**{dept_to_analyze}**: Marks show a downward trend (trend: {slope:.2f}). "
                    "We recommend seeking extra help or tutoring to strengthen understanding."
                )
            else:
                recommendations.append(
                    f"**{dept_to_analyze}**: Marks have been consistent (trend: {slope:.2f}). "
                    "Maintain this progress and consider exploring enrichment opportunities."
                )
        else:
            recommendations.append(f"**{dept_to_analyze}**: No mark data in current filter to analyze trend.")

    if recommendations:
        for rec in recommendations:
            st.markdown("- " + rec)
    elif departments_for_trend_analysis and not avg_mark_filled.empty:
        st.markdown(
            "- No specific trend-based recommendations generated for selected departments in filter. Ensure at least two marking periods with data.")
    elif avg_mark_filled.empty:
        pass
    else:
        st.markdown("- No specific departments identified for trend analysis in current filter.")

    st.subheader("📈 Average Mark Over Time (Filtered)")
    if not avg_mark_filled.empty:
        columns_to_plot = departments_for_trend_analysis if departments_for_trend_analysis else avg_mark_filled.columns.tolist()
        plot_title_suffix = "Selected Subjects" if (department_filter or subject_of_interest) else "All Subjects"
        fig, ax = plt.subplots(figsize=(12, 6))

        # Check if columns_to_plot has any valid columns that exist in avg_mark_filled
        valid_columns_to_plot = [col for col in columns_to_plot if col in avg_mark_filled.columns]

        if valid_columns_to_plot:
            avg_mark_filled[valid_columns_to_plot].plot(kind='line', marker='o', ax=ax)
            ax.set_ylabel("Average Mark")
            ax.set_xlabel("Timeline (Year-Period)")
            plt.xticks(rotation=45, ha="right")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title="Department")
            ax.set_title(f"Student ID: {student_id} - Avg Marks in {plot_title_suffix}")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No data available to plot for the current filter.")
    else:
        st.info("No data available to plot for the current filter.")


# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("📊 Student Academic Performance Dashboard")

# Load all data at the beginning
all_data_loaded = load_data()

# --- Sidebar for User Inputs ---
st.sidebar.header("🔍 Filters")
student_id_input = st.sidebar.text_input("Enter Student ID:", key="student_id_input")

department_options = [""]
if all_data_loaded and 'DepartmentDesc' in all_data_loaded['grades'].columns:
    department_options.extend(sorted(all_data_loaded['grades']['DepartmentDesc'].dropna().unique()))

department_filter_input = st.sidebar.selectbox(
    "Filter Performance by Specific Department (Optional):",
    options=department_options,
    index=0,
    key="department_filter"
)

subject_options = ['All Subjects', 'Math', 'ELA', 'Science', 'Social Studies', 'Health/ PersonalFitness',
                   'World Language/ FineArts/ CareerTech', 'Electives']
subject_of_interest_input = st.sidebar.selectbox(
    "Filter Performance by Subject Area (Optional):",
    options=subject_options,
    key="subject_of_interest"
)

if st.sidebar.button("View Performance", key="view_performance_button"):
    if student_id_input:
        st.header(f"🚀 Performance Overview for Student ID: {student_id_input}")
        display_student_performance(
            all_data_loaded,
            student_id_input,
            department_filter=department_filter_input if department_filter_input else None,
            subject_of_interest=subject_of_interest_input if subject_of_interest_input != 'All Subjects' else None
        )
    else:
        st.sidebar.warning("Please enter a Student ID.")
else:
    st.info("Enter a Student ID and click 'View Performance' in the sidebar to see the dashboard.")

