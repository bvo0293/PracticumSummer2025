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
    </div>
  );
}

export default TopBar;
