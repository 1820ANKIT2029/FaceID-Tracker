import React, { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';
import '../App.css';

const App = () => {
  const videoRef = useRef(null);
  const overlayContainerRef = useRef(null);
  const [capturedImages, setCapturedImages] = useState([]);

  // Load face-api models and start the video stream when the component mounts
  useEffect(() => {
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
      faceapi.nets.faceExpressionNet.loadFromUri('/models')
    ])
      .then(startVideo)
      .catch(err => console.error('Failed to load models:', err));
  }, []);

  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: {} })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => console.error('Error accessing the camera:', err));
  };

  // Set up an overlay canvas for continuous face detection
  useEffect(() => {
    const video = videoRef.current;
    let detectionInterval;
    const handlePlay = () => {
      // Create and append the overlay canvas
      const canvas = faceapi.createCanvasFromMedia(video);
      if (overlayContainerRef.current) {
        overlayContainerRef.current.appendChild(canvas);
      }
      const displaySize = { width: video.width, height: video.height };
      faceapi.matchDimensions(canvas, displaySize);

      detectionInterval = setInterval(async () => {
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceExpressions();
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
      }, 100);
    };

    if (video) {
      video.addEventListener('play', handlePlay);
    }
    return () => {
      if (video) {
        video.removeEventListener('play', handlePlay);
      }
      if (detectionInterval) {
        clearInterval(detectionInterval);
      }
    };
  }, []);

  // Handle the "Click" button to capture the detected face as a PNG image
  const handleCaptureFace = async () => {
    const video = videoRef.current;
    if (!video) return;

    // Detect a single face from the current video frame
    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();

    if (detection) {
      // Extract the face region using the detection bounding box
      const faceCanvases = await faceapi.extractFaces(video, [detection.detection.box]);
      if (faceCanvases && faceCanvases.length > 0) {
        // Convert the canvas to a PNG data URL
        const dataUrl = faceCanvases[0].toDataURL('image/png');
        setCapturedImages(prev => [...prev, dataUrl]);
      }
    } else {
      alert('No face detected!');
    }
  };

  return (
    <div className="app-container">
      <div className="video-container" ref={overlayContainerRef}>
        <video
          ref={videoRef}
          width="720"
          height="560"
          autoPlay
          muted
          style={{ position: 'relative' }}
        />
        {/* "Click" button positioned at the bottom of the video */}
        <button className="capture-button" onClick={handleCaptureFace}>
          Click
        </button>
      </div>
      {/* Display captured images under the heading "Captured Images" */}
      <div className="captured-images-container">
        <h2>Captured Images</h2>
        <div className="images-grid">
          {capturedImages.map((imgSrc, index) => (
            <img key={index} src={imgSrc} alt={`Captured face ${index + 1}`} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
