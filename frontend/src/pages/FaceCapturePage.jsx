import { useEffect, useRef } from "react";

const FaceCapturePage = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    };
    startCamera();
  }, []);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-10">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-4">Face Capture</h2>
        <video
          ref={videoRef}
          autoPlay
          className="border rounded-lg shadow-md w-[800px] h-[600px]"
        />
      </div>
    </div>
  );
};

export default FaceCapturePage;
