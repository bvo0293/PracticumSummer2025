import TopBar from "./TopBar";
import MainContent from "./MainContent";
import SideBar from "./SideBar";
import type { ReturnedStudentData } from "./types";
import { useState } from "react";
import "./App.css";

function App() {
  const [responseData, setResponseData] = useState<ReturnedStudentData[]>([]);

  const handleApiResponse = (dataFromForm: ReturnedStudentData[]) => {
    setResponseData(dataFromForm);
  };

  return (
    <div>
      <div style={{ position: "fixed", top: 0, left: 0, right: 0 }}>
        <TopBar />
      </div>

      <div className="d-flex" style={{ paddingTop: "35px", height: "100%" }}>
        <div
          style={{
            width: "300px",
            position: "fixed",
            top: "35px",
            bottom: 0,
            backgroundColor: "#f0f2f6",
            overflowY: "auto",
          }}
        >
          <SideBar onResponse={handleApiResponse} />
        </div>

        <div
          className="flex-grow-1 p-1"
          style={{
            marginLeft: "300px",
            overflowY: "auto",
            height: "calc(100vh - 35px)",
          }}
        >
          <MainContent data={responseData} />
        </div>
      </div>
    </div>
  );
}

export default App;
