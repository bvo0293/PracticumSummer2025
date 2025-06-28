import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---- Configuration for Subject Mapping ----
SUBJECT_TO_DEPT_MAP = {
    'MATH': ['MATH', 'MATHEMATICS'],
    'LANGUAGE ARTS': ['LANGUAGE ARTS', 'ENGLISH', 'ELA'],
    'SCIENCE': ['SCIENCE'],
    'SOCIAL SCIENCES': ['SOCIAL STUDIES', 'HISTORY', 'SOCIAL SCIENCES'],
    'HEALTH/ PERSONALFITNESS': ['HEALTH', 'PHYSICAL EDUCATION', 'PERSONAL FITNESS', 'HEALTH EDUCATION'],
    'WORLD LANGUAGE/ FINEARTS/ CAREERTECH': [
        'WORLD LANGUAGES', 'SPANISH', 'FRENCH', 'GERMAN',
        'FINE ARTS', 'ART', 'MUSIC', 'THEATRE', 'DANCE',
        'CAREER TECHNICAL AND AGRICULTURAL EDUCATION', 'CAREER AND TECHNICAL EDUCATION', 'CTE', 'BUSINESS', 'TECHNOLOGY'
    ],
    'ELECTIVES': ['ELECTIVE COURSES']
}


# ---- Feature Engineering & Collaborative Filtering Logic ----

def assign_granular_difficulty_rank(row):
    """Creates a more detailed difficulty rank by combining CourseLevelDesc and HonorsDesc."""
    base_rank = 1.0
    bonus_rank = 0.0
    level_desc = row.get('CourseLevelDesc')
    if pd.notna(level_desc) and isinstance(level_desc, str):
        match = re.search(r'level (\d+)', level_desc.lower())
        if match:
            base_rank = float(match.group(1))
    honors_desc = row.get('HonorsDesc')
    if pd.notna(honors_desc) and isinstance(honors_desc, str):
        honors_desc = honors_desc.lower()
        if 'ib' in honors_desc:
            bonus_rank = 0.6
        elif 'ap' in honors_desc:
            bonus_rank = 0.4
        elif 'dual' in honors_desc:
            bonus_rank = 0.3
        elif 'honors' in honors_desc or 'hr' in honors_desc:
            bonus_rank = 0.2
    return base_rank + bonus_rank


@st.cache_resource
def create_student_vectors(_grades_df, _courses_df):
    """Creates a feature vector for each student to represent their academic profile."""
    print("Creating student profile vectors for collaborative filtering...")
    if 'DepartmentDesc' not in _grades_df.columns:
        return pd.DataFrame()
    student_subject_marks = _grades_df.pivot_table(index='student_id', columns='DepartmentDesc', values='Mark',
                                                   aggfunc='mean').fillna(0)
    student_overall_avg = _grades_df.groupby('student_id')['Mark'].mean().to_frame('overall_avg_mark')
    courses_with_difficulty = _courses_df.copy()
    courses_with_difficulty['difficulty_rank'] = courses_with_difficulty.apply(assign_granular_difficulty_rank, axis=1)
    grades_with_difficulty = pd.merge(_grades_df, courses_with_difficulty[['siscourseidentifier', 'difficulty_rank']],
                                      left_on='CourseNumber', right_on='siscourseidentifier', how='left')
    grades_with_difficulty['difficulty_rank'].fillna(1.0, inplace=True)
    advanced_courses_count = grades_with_difficulty[grades_with_difficulty['difficulty_rank'] >= 2].groupby(
        'student_id').size().to_frame('advanced_course_count')
    student_vectors = student_subject_marks.join(student_overall_avg, how='left').join(advanced_courses_count,
                                                                                       how='left').fillna(0)
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(student_vectors)
    scaled_student_vectors_df = pd.DataFrame(scaled_vectors, index=student_vectors.index,
                                             columns=student_vectors.columns)
    print("Student profile vectors created successfully.")
    return scaled_student_vectors_df


# ---- Load All Assets ----
@st.cache_resource
def load_all_assets():
    """Loads all dataframes, the ML model, and pre-computes student vectors."""
    assets = {}
    try:
        # Load data
        assets['grades'] = pd.read_csv("Data/Final_DF.csv", low_memory=False)
        assets['courses'] = pd.read_csv("Data/Courses.csv")
        assets['grad_summary'] = pd.read_csv("Data/GraduationAreaSummary.csv")
        assets['illuminate'] = pd.read_csv("Data/cleaned_illuminate.csv", low_memory=False)

        # Clean column names for all loaded dataframes
        for df_name in ['grades', 'courses', 'grad_summary', 'illuminate']:
            assets[df_name].columns = assets[df_name].columns.str.strip()

        # Rename columns after cleaning
        assets['grad_summary'] = assets['grad_summary'].rename(columns={'mask_studentpersonkey': 'student_id'})

        # Clean data types and values
        for df_name in ['grades', 'grad_summary', 'illuminate']:
            assets[df_name]['student_id'] = assets[df_name]['student_id'].astype(str)
        assets['grades']['Mark'] = pd.to_numeric(assets['grades']['Mark'], errors='coerce')
        if 'DepartmentDesc' in assets['grades'].columns:
            assets['grades']['DepartmentDesc'] = assets['grades']['DepartmentDesc'].str.upper().str.strip()
        if 'Department' in assets['illuminate'].columns:
            assets['illuminate']['Department'] = assets['illuminate']['Department'].str.upper().str.strip()

        # Load ML model and features
        assets['model'] = joblib.load('student_success_model.pkl')
        training_cols_df = pd.read_csv("training_data_v2.csv")
        assets['model_features'] = training_cols_df.drop('Success', axis=1).columns.tolist()

        # Pre-compute student vectors for collaborative filtering
        assets['student_vectors'] = create_student_vectors(assets['grades'], assets['courses'])

        print("All assets loaded and prepared successfully.")
        return assets
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please check setup. Missing file: {e.filename}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        return None


# ---- Recommendation Engines ----

def get_student_profile(student_id, grades_df):
    student_grades = grades_df[grades_df['student_id'] == student_id].copy()
    if student_grades.empty: return None
    profile = {}
    student_grades['GradeLevel'] = pd.to_numeric(student_grades['GradeLevel'], errors='coerce')
    profile['grade_level'] = student_grades.sort_values(by='SchoolYear', ascending=False)['GradeLevel'].iloc[0]
    valid_marks = student_grades[student_grades['Mark'].notna()]
    profile['average_mark'] = valid_marks['Mark'].mean() if not valid_marks.empty else 75
    profile['subject_avg_marks'] = valid_marks.groupby('DepartmentDesc')['Mark'].mean().to_dict()
    return profile


def generate_ml_recommendations(student_id, assets):
    summary_df, grades_df, courses_df, model, model_features = assets['grad_summary'], assets['grades'], assets[
        'courses'], assets['model'], assets['model_features']
    profile = get_student_profile(student_id, grades_df)
    if not profile: return pd.DataFrame()
    credit_gaps = summary_df[(summary_df['student_id'] == student_id) & (summary_df['AreaCreditStillNeeded'] > 0)]
    if credit_gaps.empty: return pd.DataFrame()
    courses_enhanced = courses_df.copy()
    courses_enhanced['course_difficulty_rank'] = courses_enhanced.apply(assign_granular_difficulty_rank, axis=1)
    recommendations = []
    for _, gap in credit_gaps.iterrows():
        subject_area = gap['SubjectArea']
        department_list = SUBJECT_TO_DEPT_MAP.get(subject_area.upper(), [])
        if not department_list: continue
        possible_courses = courses_enhanced[courses_enhanced['DepartmentDesc'].isin(department_list)]
        taken_course_numbers = set(grades_df[grades_df['student_id'] == student_id]['CourseNumber'])
        for _, course in possible_courses.iterrows():
            if course['siscourseidentifier'] in taken_course_numbers: continue
            avg_mark_in_subject = profile['subject_avg_marks'].get(course['DepartmentDesc'], profile['average_mark'])
            dept_feature_name = f"dept_{course['DepartmentDesc']}"
            features_for_pred = pd.DataFrame(columns=model_features)
            features_for_pred.loc[0, 'student_avg_mark_overall'] = profile['average_mark']
            features_for_pred.loc[0, 'student_avg_mark_in_subject'] = avg_mark_in_subject
            features_for_pred.loc[0, 'course_difficulty_rank'] = course['course_difficulty_rank']
            if dept_feature_name in features_for_pred.columns:
                features_for_pred.loc[0, dept_feature_name] = 1
            features_for_pred = features_for_pred[model_features].fillna(0)
            success_prob = model.predict_proba(features_for_pred)[0][1]
            recommendations.append({'SubjectArea': subject_area, 'coursename': course['coursename'],
                                    'CourseId': course['siscourseidentifier'], 'success_prob': success_prob,
                                    'HonorsDesc': course['HonorsDesc']})
    return pd.DataFrame(recommendations)


def get_collaborative_recommendations(target_student_id, assets, k=20, success_threshold=80):
    student_vectors_df = assets['student_vectors']
    if target_student_id not in student_vectors_df.index: return pd.DataFrame()
    target_vector = student_vectors_df.loc[[target_student_id]]
    similarities = cosine_similarity(target_vector, student_vectors_df)[0]
    sim_series = pd.Series(similarities, index=student_vectors_df.index)
    similar_students = sim_series.drop(target_student_id).nlargest(k).index.tolist()
    if not similar_students: return pd.DataFrame()
    grades_df, courses_df = assets['grades'], assets['courses']
    neighbor_grades = grades_df[grades_df['student_id'].isin(similar_students)]
    successful_neighbor_courses = neighbor_grades[neighbor_grades['Mark'] >= success_threshold]
    target_student_courses = set(grades_df[grades_df['student_id'] == target_student_id]['CourseNumber'])
    potential_recs = successful_neighbor_courses[
        ~successful_neighbor_courses['CourseNumber'].isin(target_student_courses)]
    if potential_recs.empty: return pd.DataFrame()

    # Make the merge and groupby more robust to missing columns
    detail_cols_to_merge = ['siscourseidentifier', 'coursename', 'HonorsDesc', 'CourseLevelDesc']
    existing_detail_cols = [col for col in detail_cols_to_merge if col in courses_df.columns]

    potential_recs_with_details = pd.merge(potential_recs, courses_df[existing_detail_cols], left_on='CourseNumber',
                                           right_on='siscourseidentifier', how='left')

    group_cols = ['CourseNumber', 'coursename']
    if 'HonorsDesc' in potential_recs_with_details.columns:
        group_cols.append('HonorsDesc')
    if 'CourseLevelDesc' in potential_recs_with_details.columns:
        group_cols.append('CourseLevelDesc')

    course_counts = potential_recs_with_details.groupby(group_cols).size().reset_index(name='peer_count')
    course_counts = course_counts.sort_values(by='peer_count', ascending=False)

    return course_counts.dropna().head(10)


# ---- Main Display Function ----
def display_student_performance(all_data, student_id, department_filter=None, subject_of_interest=None):
    if not all_data: return

    grades_df = all_data['grades']
    grad_summary_df = all_data['grad_summary']
    illuminate_df = all_data['illuminate']

    student_df_orig = grades_df[grades_df['student_id'] == str(student_id)].copy()
    if student_df_orig.empty:
        st.warning(f"No data found for student ID: {student_id}")
        return

    # ---- Current Assessment Snapshot Section ----
    st.subheader("🔔 Current Assessment Snapshot (Illuminate 2025)")
    student_illuminate_data = illuminate_df[illuminate_df['student_id'] == str(student_id)].copy()
    student_illuminate_data['Department'] = student_illuminate_data['Department'].str.upper().str.strip()

    if not student_illuminate_data.empty:
        student_profile = get_student_profile(student_id, grades_df)
        historical_averages = student_profile.get('subject_avg_marks', {})

        # Ensure 'responsedatevalue' is datetime and handle NaNs robustly
        student_illuminate_data['responsedatevalue'] = pd.to_datetime(student_illuminate_data['responsedatevalue'],
                                                                      errors='coerce')
        student_illuminate_data.dropna(subset=['responsedatevalue', 'Department', 'Response_percent_correct'],
                                       inplace=True)

        if not student_illuminate_data.empty:
            # **REVISED LOGIC**: Get the max date for each department
            latest_dates = student_illuminate_data.groupby('Department')['responsedatevalue'].max().reset_index()
            latest_dates = latest_dates.rename(columns={'responsedatevalue': 'max_date'})

            # Merge to find all assessments on the latest date for each department
            latest_assessments = pd.merge(student_illuminate_data, latest_dates, on='Department')
            latest_assessments = latest_assessments[
                latest_assessments['responsedatevalue'] == latest_assessments['max_date']]

            # If multiple assessments on the same day, average their scores.
            final_display_scores = latest_assessments.groupby('Department')[
                'Response_percent_correct'].mean().reset_index()

            cols = st.columns(len(final_display_scores) if len(final_display_scores) > 0 else 1)
            col_idx = 0
            for _, row in final_display_scores.iterrows():
                dept = row['Department']
                recent_score = row['Response_percent_correct']
                historical_avg = historical_averages.get(dept, None)
                with cols[col_idx]:
                    st.metric(label=f"Most Recent {dept} Assessment", value=f"{recent_score:.1f}%")
                    if historical_avg is not None:
                        delta = recent_score - historical_avg
                        st.metric(label="vs. Historical Average", value=f"{historical_avg:.1f}%", delta=f"{delta:.1f}%")
                col_idx += 1
        else:
            st.info("No recent Illuminate assessment data with valid dates found for this student.")
    else:
        st.info("No recent Illuminate assessment data found for this student.")

    st.markdown("---")

    # ---- Generate All Recommendations ----
    ml_recommendations_df = generate_ml_recommendations(student_id, all_data)
    collab_recs_df = get_collaborative_recommendations(student_id, all_data)

    # ---- Display ML-Based Recommendations ----
    st.subheader("🎓 Graduation Credit & AI-Powered Course Recommendations")
    st.markdown("_Recommendations to fill credit gaps, with a predicted success score._")
    student_credit_needs = grad_summary_df[grad_summary_df['student_id'] == str(student_id)]
    if not student_credit_needs.empty:
        for _, need in student_credit_needs.iterrows():
            credits_needed = need['AreaCreditStillNeeded']
            display_name = need['SubjectArea']
            if credits_needed > 0:
                with st.expander(f"**{display_name}**: {credits_needed:.1f} Credits Needed"):
                    if not ml_recommendations_df.empty:
                        subject_recs = ml_recommendations_df[ml_recommendations_df['SubjectArea'] == display_name]
                        if not subject_recs.empty:
                            subject_recs = subject_recs.sort_values(by='success_prob', ascending=False)
                            for _, row in subject_recs.iterrows():
                                honor_desc = f"({row['HonorsDesc']})" if pd.notna(row['HonorsDesc']) else ""
                                st.markdown(
                                    f"- **{row['coursename']}** {honor_desc} (ID: `{row['CourseId']}`) - Predicted Success: **{row['success_prob'] * 100:.0f}%**")
                        else:
                            st.write("No specific course recommendations generated for this subject.")
                    else:
                        st.write("Recommendation engine did not produce results.")
            else:
                st.metric(label=f"{display_name} Needed", value="Complete ✔️")
    else:
        st.write("Could not retrieve graduation credit requirement data.")

    st.markdown("---")

    # ---- Display Collaborative Filtering Recommendations ----
    st.subheader("💡 Courses Popular Among Academic Peers")
    st.markdown("_Based on courses that students with a similar academic profile to yours have succeeded in._")
    if not collab_recs_df.empty:
        for _, row in collab_recs_df.iterrows():
            honor_desc = f"({row['HonorsDesc']})" if 'HonorsDesc' in row and pd.notna(row['HonorsDesc']) else ""
            level_desc = f"{row['CourseLevelDesc']}" if 'CourseLevelDesc' in row and pd.notna(
                row['CourseLevelDesc']) else ""
            st.markdown(
                f"- **{row['coursename']}** {honor_desc} {level_desc} (ID: `{row['CourseNumber']}`) - Taken by {row['peer_count']} peers")
    else:
        st.write("No peer-based recommendations could be generated at this time.")

    st.markdown("---")

    # --- Performance Analysis Section ---
    st.subheader("📈 Student Performance Analysis")
    student_df = student_df_orig.copy()

    if department_filter:
        student_df = student_df[student_df['DepartmentDesc'].str.upper() == department_filter.upper()]
        st.info(f"Filtering performance analysis by Department: **{department_filter.upper()}**")
    elif subject_of_interest:
        department_list = SUBJECT_TO_DEPT_MAP.get(subject_of_interest.upper(), [])
        student_df = student_df[student_df['DepartmentDesc'].isin(department_list)]
        st.info(f"Filtering performance analysis by Subject Area: **{subject_of_interest}**")

    if student_df.empty:
        st.warning("No performance data found for the selected filter.")
        return

    student_df = student_df.rename(columns={
        'SchoolYear': 'Year', 'GradeLevel': 'Grade Level', 'CourseDesc': 'Course',
        'MarkingPeriodCode': 'Period', 'Mark': 'Mark', 'EarnedCredit': 'Credit',
        'SchoolDetailFCSId': 'School ID', 'DepartmentDesc': 'Department'
    })

    with st.expander("📖 View Performance Table (Filtered)", expanded=False):
        display_cols = ['Year', 'Grade Level', 'Course', 'Period', 'Mark', 'Credit', 'Department']
        st.dataframe(student_df[display_cols], use_container_width=True, hide_index=True)

    student_df['Timeline'] = student_df['Year'].astype(str) + '-P' + student_df['Period'].astype(str)

    avg_mark = student_df.groupby(['Department', 'Timeline'])['Mark'].mean().unstack(level=0)
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

    st.subheader("💡 Performance Trend Summary (Filtered)")

    departments_for_trend_analysis = []
    if not avg_mark_filled.empty:
        if department_filter:
            departments_for_trend_analysis = [dept.upper() for dept in avg_mark_filled.columns if
                                              dept.upper() == department_filter.upper()]
        elif subject_of_interest:
            departments_for_trend_analysis = [dept.upper() for dept in
                                              SUBJECT_TO_DEPT_MAP.get(subject_of_interest.upper(), []) if
                                              dept.upper() in avg_mark_filled.columns]
        else:
            departments_for_trend_analysis = avg_mark_filled.columns.tolist()

    trend_recommendations = []
    if not avg_mark_filled.empty and departments_for_trend_analysis:
        for dept_to_analyze in departments_for_trend_analysis:
            marks = avg_mark_filled[dept_to_analyze].dropna()
            if len(marks) < 2:
                trend_recommendations.append(f"**{dept_to_analyze}**: Not enough data points to analyze trend.")
                continue

            x = np.arange(len(marks))
            y = marks.values
            slope = np.polyfit(x, y, 1)[0] if np.any(y != y[0]) else 0
            threshold = 0.05

            if slope > threshold:
                trend_recommendations.append(f"**{dept_to_analyze}**: Marks are improving (trend: {slope:.2f}).")
            elif slope < -threshold:
                trend_recommendations.append(
                    f"**{dept_to_analyze}**: Marks show a downward trend (trend: {slope:.2f}).")
            else:
                trend_recommendations.append(f"**{dept_to_analyze}**: Marks are consistent (trend: {slope:.2f}).")

    if trend_recommendations:
        for rec in trend_recommendations:
            st.markdown("- " + rec)
    else:
        st.info("No performance trend data to display for the current filter.")

    st.subheader("📈 Average Mark Over Time (Filtered)")
    if not avg_mark_filled.empty:
        columns_to_plot = avg_mark_filled.columns.tolist()
        plot_title_suffix = "Selected Subjects" if (department_filter or subject_of_interest) else "All Subjects"
        fig, ax = plt.subplots(figsize=(12, 6))
        avg_mark_filled[columns_to_plot].plot(kind='line', marker='o', ax=ax)
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


# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("📊 Student Academic Performance Dashboard")

all_assets_loaded = load_all_assets()

st.sidebar.header("🔍 Filters")
student_id_input = st.sidebar.text_input("Enter Student ID:", key="student_id_input")

department_filter_value = None
subject_filter_value = None
if all_assets_loaded:
    dept_list = [""] + sorted(all_assets_loaded['grades']['DepartmentDesc'].dropna().unique())
    department_filter_value = st.sidebar.selectbox("Filter Performance by Department:", options=dept_list, index=0)
    if not department_filter_value:
        # Use uppercase keys for the dropdown options
        subject_list = [""] + list(SUBJECT_TO_DEPT_MAP.keys())
        subject_filter_value = st.sidebar.selectbox("Filter Performance by Subject Area:", options=subject_list,
                                                    index=0)

if st.sidebar.button("View Performance", key="view_performance_button"):
    if all_assets_loaded:
        if student_id_input:
            st.header(f"🚀 Performance Overview for Student ID: {student_id_input}")
            display_student_performance(
                all_assets_loaded,
                student_id_input,
                department_filter=department_filter_value if department_filter_value else None,
                subject_of_interest=subject_filter_value if subject_filter_value else None
            )
        else:
            st.sidebar.warning("Please enter a Student ID.")
    else:
        st.error("Application assets could not be loaded. Please check file paths and try again.")
else:
    st.info("Enter a Student ID and click 'View Performance' in the sidebar to see the dashboard.")
