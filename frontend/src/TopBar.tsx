import logo from "./assets/logo.png";

function TopBar() {
  return (
    <div>
      <div
        className="d-flex align-items-center justify-content-end p-3"
        style={{ backgroundColor: "#242424", height: "35px" }}
      >
        <a
          href="https://www.facebook.com/FultonCountySchools/"
          className="social-icon"
          target="_blank"
          rel="noopener noreferrer"
        >
          <i className="bi bi-facebook"></i>
        </a>
        <a
          href="https://www.instagram.com/fultoncoschools/"
          className="social-icon"
          target="_blank"
          rel="noopener noreferrer"
        >
          <i className="bi bi-instagram"></i>
        </a>
        <a
          href="https://x.com/FultonCoSchools"
          className="social-icon"
          target="_blank"
          rel="noopener noreferrer"
        >
          <i className="bi bi-twitter-x"></i>
        </a>
        <a
          href="https://www.youtube.com/@FultonCountySchoolsTV"
          className="social-icon"
          target="_blank"
          rel="noopener noreferrer"
        >
          <i className="bi bi-youtube"></i>
        </a>
      </div>

      <nav
        className="container-fluid d-flex justify-content align-items-center p-5"
        style={{ height: "100px", backgroundColor: "white" }}
      >
        <img src={logo} alt="Logo" style={{ height: "100px" }} />
        <h1 className="flex-grow-1 text-center m-0" style={{ color: "#00593c"}}>Student Academic Performance Dashboard</h1>
      </nav>
    </div>
  );
}

export default TopBar;
