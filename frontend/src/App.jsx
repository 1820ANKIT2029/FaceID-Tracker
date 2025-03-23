import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import FaceCapturePage from "./pages/FaceCapturePage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FaceCapturePage />} />
      </Routes>
    </Router>
  );
}

export default App;
