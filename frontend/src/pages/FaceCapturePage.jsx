import React, { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';
import '../App.css';

const App = () => {
  const videoRef = useRef(null);
  const overlayContainerRef = useRef(null);
  const canvasRef = useRef(null);
  const [predictions, setPredictions] = useState([]);

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
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(err => console.error('Error accessing the camera:', err));
  };

  useEffect(() => {
    const video = videoRef.current;
    let detectionInterval;

    const handlePlay = async () => {
      if (!canvasRef.current) {
        canvasRef.current = faceapi.createCanvasFromMedia(video);
        if (overlayContainerRef.current) {
          overlayContainerRef.current.appendChild(canvasRef.current);
          canvasRef.current.style.position = 'absolute';
          canvasRef.current.style.top = '0';
          canvasRef.current.style.left = '0';
        }
      }

      detectionInterval = setInterval(async () => {
        const canvas = canvasRef.current;
        const displaySize = video.getBoundingClientRect();
        canvas.width = displaySize.width;
        canvas.height = displaySize.height;
        faceapi.matchDimensions(canvas, displaySize);
      
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks();
      
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
      
        if (detections.length > 0) {
          const predResults = await uploadImagesSequentially(video, detections.map(det => det.detection.box));
          setPredictions(predResults);
      
          resizedDetections.forEach((detection, index) => {
            const { x, y, width, height } = detection.detection.box;
            const pred = predResults[index];
            
            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);

            let labels = `${pred.name} (${(pred.probability * 100).toFixed(1)}%)`
            if(pred.name == "unknown"){
              labels = `${pred.name}`;
            }
            else{
                labels = `${pred.name} (${(pred.probability * 100).toFixed(1)}%)`;
            }
      
            if (pred) {
              const label = labels;
              ctx.fillStyle = 'blue';
              ctx.fillRect(x, y, width, 20);
      
              ctx.fillStyle = 'white';
              ctx.font = '14px Arial';
              ctx.fillText(label, x + 5, y + 15);
            }
          });
        } else {
          setPredictions([]);
        }
      }, 1000);
      
    };

    if (video) {
      video.addEventListener('play', handlePlay);
    }

    return () => {
      if (video) video.removeEventListener('play', handlePlay);
      if (detectionInterval) clearInterval(detectionInterval);
    };
  }, []);

  const uploadImagesSequentially = async (video, faceBoxes) => {
    const results = [];
    const targetSize = 128; // Resize target
  
    for (let i = 0; i < faceBoxes.length; i++) {
      const faceCanvas = document.createElement('canvas');
      const ctx = faceCanvas.getContext('2d');
      faceCanvas.width = targetSize;
      faceCanvas.height = targetSize;
  
      // Apply mild blur filter for denoising
      ctx.filter = 'blur(1px)';
      ctx.drawImage(
        video,
        faceBoxes[i].x, faceBoxes[i].y, faceBoxes[i].width, faceBoxes[i].height,
        0, 0, targetSize, targetSize
      );
      ctx.filter = 'none';
  
      // Optional: Convert to grayscale if your model supports it
      const imgData = ctx.getImageData(0, 0, targetSize, targetSize);
      for (let j = 0; j < imgData.data.length; j += 4) {
        const avg = (imgData.data[j] + imgData.data[j + 1] + imgData.data[j + 2]) / 3;
        imgData.data[j] = imgData.data[j + 1] = imgData.data[j + 2] = avg;
      }
      ctx.putImageData(imgData, 0, 0);
  
      const blob = await new Promise(resolve => faceCanvas.toBlob(resolve, 'image/png'));
      const formData = new FormData();
      formData.append('files', blob, `face_${i}.png`);
  
      try {
        const response = await fetch('http://127.0.0.1:8000/predict/', {
          method: 'POST',
          body: formData
        });
  
        if (!response.ok) throw new Error(`Error: ${response.status}`);
        const data = await response.json();
        results.push(data.prediction_result[0]);
      } catch (error) {
        console.error('Error in face prediction:', error);
        results.push({ name: 'Unknown', probability: 0 });
      }
    }
  
    return results;
  };
  

  return (
    <div className="app-container">
      <div className="video-container" ref={overlayContainerRef}>
        <video ref={videoRef} width="720" height="560" autoPlay muted style={{ position: 'relative' }} />
      </div>
    </div>
  );
};

export default App;
