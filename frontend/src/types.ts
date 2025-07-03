export interface InputStudentData {
  studentId: string;
  department?: string;
  subjectArea?: string;
}

export interface StudentData {
    CourseDesc: string;
    DepartmentDesc: string;
    EarnCredit: number;
    GradeLevel: string;
    Mark: string;
    MarkingPeriodCode: string;
    SchoolDetailFCSId: number;
    SchoolYear: string;
}

export interface TotalCredit {
  total_credits: Record<string, number>;
}

export interface Trend {
  trend: Record<string, string>;
}

export interface CollabRec {
  CourseNumber: string;
  coursename: string;
  HonorsDesc: string;
  peer_count: number;
}

export interface MLRec{
  SubjectArea: string;
  coursename: string;
  CourseId: string;
  success_prob: number;
  HonorsDesc: string;
}

export interface NeededCredits {
  AreaCreditStillNeeded: number;
  SubjectArea: string;
}

export interface ReturnedStudentData {
  student_id: string;
  student_data: StudentData[];
  total_credits: TotalCredit[];
  image: string;
  trend: Trend[];
  collab_rec: CollabRec[];
  ml_rec: MLRec[];
  needed_credits: NeededCredits[];
}

export interface FormProps {
  onSubmitSuccess: (data: ReturnedStudentData) => void;
}

export interface SideBarProps {
  onResponse: (data: ReturnedStudentData) => void;
}

export interface MainContentProps {
  data: ReturnedStudentData;
}

export interface TableProps {
  rows: StudentData[];
}

export interface CardProps {
  data: TotalCredit[];
}

export interface PlotImageProps {
  image: string;
}

export interface TrendProps {
  trend: Trend[];
}

export interface CollabRecProps {
  collab_rec: CollabRec[];
} 

export interface MLRecProps {
  ml_rec: MLRec[];
  needed_credits: NeededCredits[];
}