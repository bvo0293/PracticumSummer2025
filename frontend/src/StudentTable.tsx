import { useEffect, useState } from "react";
import type { TableProps } from "./types";

function StudentTable({ rows }: TableProps) {
  if (rows.length === 0) return <p></p>;

  const [columns, setColumns] = useState<string[]>([]);
  const [records, setRecords] = useState<any[]>([]);
  

  useEffect(() => {
    if (rows.length > 0) {
      setColumns(Object.keys(rows[0]));
      setRecords(rows);
    }
  }, []);

  return (
    <div>
      <h3>🧮 Student Performance Analysis</h3>
      <p>
        This session explores student performance across courses and marking
        periods, analyzing grades, earned credits, and related school and
        department details to identify academic trends and outcomes.
      </p>
      <div
        className="table-responsive mt-4"
        style={{ height: "400px", overflowY: "auto" }}
      >
        <table className="table table-bordered table-striped">
          <thead className="table-dark">
            <tr>
              {columns.map((c, i) => (
                <th key={i}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {records.map((r, i) => (
              <tr key={i}>
                {columns.map((c, j) => (
                  <td key={j}>{r[c]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default StudentTable;
