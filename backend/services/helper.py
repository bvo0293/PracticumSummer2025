import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re

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

def get_student_profile(student_id, grades_df):
    student_grades = grades_df[grades_df['student_id'] == int(student_id)].copy()
    if student_grades.empty: return None
    profile = {}
    student_grades['GradeLevel'] = pd.to_numeric(student_grades['GradeLevel'], errors='coerce')
    profile['grade_level'] = student_grades.sort_values(by='SchoolYear', ascending=False)['GradeLevel'].iloc[0]
    valid_marks = student_grades[student_grades['Mark'].notna()]
    profile['average_mark'] = valid_marks['Mark'].mean() if not valid_marks.empty else 75
    profile['subject_avg_marks'] = valid_marks.groupby('DepartmentDesc')['Mark'].mean().to_dict()
    return profile

def generate_ml_recommendations(student_id, summary_df, grades_df, courses_df, model, model_features):
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

    profile = get_student_profile(student_id, grades_df)
    if not profile: return pd.DataFrame()
    credit_gaps = summary_df[(summary_df['student_id'] == int(student_id)) & (summary_df['AreaCreditStillNeeded'] > 0)]
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
                                    'CourseId': course['siscourseidentifier'], 'success_prob': success_prob, 'HonorsDesc': course["HonorsDesc"]})
    return pd.DataFrame(recommendations).groupby("SubjectArea", group_keys=False).apply(lambda x: x.nlargest(5, 'success_prob'))


def get_collaborative_recommendations(target_student_id, student_vectors_df, grade_cache, courses_cache , k=20, success_threshold=80):
    """Generates course recommendations based on successful academic peers."""
    if target_student_id not in student_vectors_df.index:
        return pd.DataFrame()

    target_vector = student_vectors_df.loc[[target_student_id]]
    similarities = cosine_similarity(target_vector, student_vectors_df)[0]
    sim_series = pd.Series(similarities, index=student_vectors_df.index)
    similar_students = sim_series.drop(target_student_id).nlargest(k).index.tolist()
    if not similar_students: return pd.DataFrame()

    grades_df, courses_df = grade_cache, courses_cache

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