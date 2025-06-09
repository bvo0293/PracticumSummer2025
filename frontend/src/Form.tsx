import { useState } from "react";
import type {FormEvent} from "react" ;

interface StudentData {
  studentId: string;
  department?: string;
}

function Form() {
  const [studentId, setStudentId] = useState("");
  const [department, setDepartment] = useState("");

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const data:StudentData = {
      studentId,
      department: department.trim() === "" ? undefined : department,
    };

    try {
      const response = await fetch("http://localhost:8000/api/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      alert(`Server response: ${result.message}`);
    } catch (error: any) {
      alert(`Failed to submit: ${error.message}`);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{ maxWidth: "400px", margin: "0 auto" }}
    >
      <div className="mb-3">
        <label htmlFor="studentId" className="form-label">
          Student ID:
        </label>
        <input
          id="studentId"
          type="text"
          value={studentId}
          onChange={(e) => setStudentId(e.target.value)}
          placeholder="Enter your student ID"
          className="form-control"
          required
          style={{ width: "100%" }}
        />
      </div>

      <div className="mb-3">
        <label htmlFor="department" className="form-label">
          Department (optional):
        </label>
        <input
          id="department"
          type="text"
          value={department}
          onChange={(e) => setDepartment(e.target.value)}
          placeholder="Enter your department"
          className="form-control"
          style={{ width: "100%" }}
        />
      </div>

      <button
        type="submit"
        disabled={studentId.trim() === ""}
        className="btn btn-primary w-100"
      >
        Apply
      </button>
    </form>
  );
}

export default Form;
