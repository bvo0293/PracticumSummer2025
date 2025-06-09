import TopBar from "./Topbar";
import MainContent from "./MainContent";
import './App.css'

function App() {
  return (
    <div className="h-screen flex flsex-col">
      <TopBar />
      <MainContent></MainContent>
    </div>
  );
}

export default App;
