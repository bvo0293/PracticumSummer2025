import logo from "./assets/logo.png";
import Form from "./Form";
import type { SideBarProps } from "./types";

function SideBar({ onResponse }: SideBarProps) {
  return (
    <div className="p-0" style={{ minWidth: "300px" }}>
      <div className="d-flex justify-content-center align-items-center p-2">
        <img src={logo} alt="Logo" style={{ height: "80px" }} />
      </div>

      <div className="justify-content-center align-items-center p-3">
        <Form onSubmitSuccess={onResponse} />
      </div>
    </div>
  );
}

export default SideBar;
