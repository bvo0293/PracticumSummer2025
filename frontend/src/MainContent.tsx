import type { MainContentProps } from "./types";
import StudentTable from "./StudentTable";
import Cards from "./Cards";
import Plot from "./Plot";
import Trend from "./Trend";
import CollabRec from "./CollabRec";
import MLRec from "./MLRec";

function MainContent({ data }: MainContentProps) {
  const student_id = data.student_id ? data.student_id : "";
  const student_data = data.student_data ? data.student_data : [];
  const total_credits = data.total_credits ? data.total_credits : [];
  const image: string = data?.image ?? "";
  const trend = data.trend ? data.trend : [];
  const collab_rec = data.collab_rec ? data.collab_rec : [];
  const ml_rec = data.ml_rec ? data.ml_rec : [];
  const needed_credits = data.needed_credits ? data.needed_credits : [];

  if (student_id === "") return <p></p>;

  return (
    <div style={{ padding: "16px" }}>
      <h1
        className="text-center m-0"
        style={{ color: "#00593c", height: "75px" }}
      >
        Student Academic Performance Dashboard
      </h1>
      <h2>I. Credit Summary</h2>
      <div className="d-flex align-items-center mb-4">
        <div className="flex-grow-1">
          <Cards data={total_credits} />
        </div>
        <div
          className="card"
          style={{
            width: "450px",
            height: "320px",
            backgroundColor: "#f8faff",
          }}
        >
          <div className="card-body">
            <h5 className="card-title">Student Profile</h5>
            <p className="card-text">Student ID: {student_id}</p>
            <p className="card-text">Name: XXXXX XXXXX</p>
            <p className="card-text">Date of Birth: XXXX/XX/XX</p>
            <p className="card-text">Sex: XXXX</p>
            <p className="card-text">School: XXXXX</p>
            <p className="card-text">Grade: XX</p>
            <p className="card-text">GPA: XX</p>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <Plot image={image} />
      </div>

      <div className="mb-4">
        <StudentTable rows={student_data} />
      </div>

      <div className="mb-4">
        <Trend trend={trend} />
      </div>
      <hr />

      <h2>II. Recommendations</h2>
      <div className="mb-4">
        <CollabRec collab_rec={collab_rec} />
      </div>

      <div className="mb-4">
        <MLRec ml_rec={ml_rec} needed_credits={needed_credits} />
      </div>
    </div>
  );
}

export default MainContent;
