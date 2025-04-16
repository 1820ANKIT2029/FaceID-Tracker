import React, { useRef, useEffect, useState } from 'react';
import * as faceapi from 'face-api.js';
import '../App.css';
import Table from './Table';

const App = () => {
  const videoRef = useRef(null);
  const overlayContainerRef = useRef(null);
  const canvasRef = useRef(null);
  const [predictions, setPredictions] = useState([]);
  const [data, setData] = useState([]);

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

            let labels = `${pred.name} (${(pred.probability * 100).toFixed(1)}%)`;
            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true });

            setData((prev) => {
              const exists = prev.some((entry) => entry.name === pred.name);
              return exists ? prev : [{ name: pred.name, time: time }, ...prev];
            });

            if (pred.name === "unknown") {
              labels = `${pred.name}`;
            }

            if (pred) {
              ctx.fillStyle = 'blue';
              ctx.fillRect(x, y, width, 20);

              ctx.fillStyle = 'white';
              ctx.font = '14px Arial';
              ctx.fillText(labels, x + 5, y + 15);
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
    const targetSize = 128;

    for (let i = 0; i < faceBoxes.length; i++) {
      const faceCanvas = document.createElement('canvas');
      const ctx = faceCanvas.getContext('2d');
      faceCanvas.width = targetSize;
      faceCanvas.height = targetSize;

      ctx.filter = 'blur(1px)';
      ctx.drawImage(
        video,
        faceBoxes[i].x, faceBoxes[i].y, faceBoxes[i].width, faceBoxes[i].height,
        0, 0, targetSize, targetSize
      );
      ctx.filter = 'none';

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
    <>
      <div className="min-h-screen w-full bg-gradient-to-br from-[#0a1e5e] to-[#071a3c] text-white">
        {/* Header */}
        <div className="flex justify-center p-6 border-b border-blue-600">
          <h1 className="text-4xl font-extrabold text-cyan-400 drop-shadow-md">
            FaceTrack<span className="text-white">.AI</span>
          </h1>
        </div>
  
        {/* Main Layout */}
        <div className="flex h-[calc(100vh-100px)] w-full">
          {/* Left - Video */}
          <div className="w-2/3 p-6 border-r border-blue-800 flex items-center justify-center">
            <div
              className="relative border-4 border-cyan-500 rounded-2xl shadow-xl h-full w-full"
              ref={overlayContainerRef}
            >
              <video
                ref={videoRef}
                autoPlay
                muted
                className="rounded-2xl w-full h-full object-cover"
              />
            </div>
          </div>
  
          {/* Right - Table */}
          <div className="w-1/3 p-6 overflow-y-auto text-white">
            <h2 className="text-2xl mb-4 font-semibold text-cyan-300 border-b border-cyan-500 pb-2">
              Detection Results
            </h2>
            <Table data={data} />
          </div>
        </div>
      </div>
    </>
  );
  

  
  
  
};

export default App;
