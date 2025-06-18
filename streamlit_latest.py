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
        elif 'ap' in honors_desc or 'dual' in honors_desc:
            bonus_rank = 0.4
        elif 'honors' in honors_desc or 'hr' in honors_desc:
            bonus_rank = 0.2
    return base_rank + bonus_rank


@st.cache_resource
def create_student_vectors(_grades_df, _courses_df):
    """Creates a feature vector for each student to represent their academic profile."""
    print("Creating student profile vectors for collaborative filtering...")
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
        assets['grades'] = pd.read_csv("Final_DF.csv", low_memory=False)
        assets['courses'] = pd.read_csv("Data/Courses.csv")
        grad_summary_df = pd.read_csv("Data/GraduationAreaSummary.csv")
        grad_summary_df = grad_summary_df.rename(columns={'mask_studentpersonkey': 'student_id'})
        assets['grad_summary'] = grad_summary_df

        # Clean data
        for df_name in ['grades', 'grad_summary']:
            assets[df_name]['student_id'] = assets[df_name]['student_id'].astype(str)
        assets['grades']['Mark'] = pd.to_numeric(assets['grades']['Mark'], errors='coerce')
        if 'DepartmentDesc' in assets['grades'].columns:
            assets['grades']['DepartmentDesc'] = assets['grades']['DepartmentDesc'].str.upper().str.strip()

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
        department_list = SUBJECT_TO_DEPT_MAP.get(subject_area, [])
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
                                    'CourseId': course['siscourseidentifier'], 'success_prob': success_prob})
    return pd.DataFrame(recommendations)


def get_collaborative_recommendations(target_student_id, assets, k=20, success_threshold=80):
    """Generates course recommendations based on successful academic peers."""
    student_vectors_df = assets['student_vectors']
    if target_student_id not in student_vectors_df.index: return pd.DataFrame()

    target_vector = student_vectors_df.loc[[target_student_id]]
    similarities = cosine_similarity(target_vector, student_vectors_df)[0]
    sim_series = pd.Series(similarities, index=student_vectors_df.index)
    similar_students = sim_series.drop(target_student_id).nlargest(k).index.tolist()
    if not similar_students: return pd.DataFrame()

    grades_df = assets['grades']
    courses_df = assets['courses']

    neighbor_grades = grades_df[grades_df['student_id'].isin(similar_students)]
    successful_neighbor_courses = neighbor_grades[neighbor_grades['Mark'] >= success_threshold]
    target_student_courses = set(grades_df[grades_df['student_id'] == target_student_id]['CourseNumber'])
    potential_recs = successful_neighbor_courses[
        ~successful_neighbor_courses['CourseNumber'].isin(target_student_courses)]
    if potential_recs.empty: return pd.DataFrame()

    potential_recs_with_names = pd.merge(
        potential_recs,
        courses_df[['siscourseidentifier', 'coursename']],
        left_on='CourseNumber',
        right_on='siscourseidentifier',
        how='left'
    )

    course_counts = potential_recs_with_names.groupby(['CourseNumber', 'coursename']).size().reset_index(
        name='peer_count')
    course_counts = course_counts.sort_values(by='peer_count', ascending=False)

    return course_counts[['coursename', 'CourseNumber', 'peer_count']].dropna().head(10)


# ---- Main Display Function ----
def display_student_performance(all_data, student_id, department_filter=None, subject_of_interest=None):
    if not all_data: return

    grades_df = all_data['grades']
    grad_summary_df = all_data['grad_summary']
    student_df_orig = grades_df[grades_df['student_id'] == str(student_id)].copy()
    if student_df_orig.empty:
        st.warning(f"No data found for student ID: {student_id}")
        return

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
                                st.markdown(
                                    f"- **{row['coursename']}** (ID: `{row['CourseId']}`) - Predicted Success: **{row['success_prob'] * 100:.0f}%**")
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
            st.markdown(f"- **{row['coursename']}** (ID: `{row['CourseNumber']}`) - Taken by {row['peer_count']} peers")
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
        department_list = SUBJECT_TO_DEPT_MAP.get(subject_of_interest, [])
        student_df = student_df[student_df['DepartmentDesc'].isin(department_list)]
        st.info(f"Filtering performance analysis by Subject Area: **{subject_of_interest}**")

    if student_df.empty:
        st.warning("No performance data found for the selected filter.")
        return

    # **FIX**: Rename columns *before* they are used for calculations or display.
    student_df = student_df.rename(columns={
        'SchoolYear': 'Year', 'GradeLevel': 'Grade Level', 'CourseDesc': 'Course',
        'MarkingPeriodCode': 'Period', 'Mark': 'Mark', 'EarnedCredit': 'Credit',
        'SchoolDetailFCSId': 'School ID', 'DepartmentDesc': 'Department'
    })

    # Display Table
    with st.expander("📖 View Performance Table (Filtered)", expanded=False):
        # Use the new, user-friendly column names for display
        display_cols = ['Year', 'Grade Level', 'Course', 'Period', 'Mark', 'Credit', 'Department']
        st.dataframe(student_df[display_cols], use_container_width=True, hide_index=True)

    # Create Timeline for charts
    student_df['Timeline'] = student_df['Year'].astype(str) + '-P' + student_df['Period'].astype(str)

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

    st.subheader("💡 Performance Trend Summary (Filtered)")
    # (Trend analysis logic here)

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
