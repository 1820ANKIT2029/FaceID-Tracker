import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import FaceCapturePage from "./pages/FaceCapturePage";
import Home from "./pages/Home";
import React from "react";
import { Toaster } from "react-hot-toast";

function App() {
  return (
    <>
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/identify" element={<FaceCapturePage />} />
      </Routes>
    </Router>
    <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#1e3a8a", 
            color: "#e0f2fe",   
            border: "1px solid #22d3ee",
          },
          success: {
            iconTheme: {
              primary: "#22d3ee", 
              secondary: "#0f172a",
            },
          },
          error: {
            iconTheme: {
              primary: "#ef4444",
              secondary: "#0f172a",
            },
          },
        }}
      />
    </>
  );
}

export default App;
