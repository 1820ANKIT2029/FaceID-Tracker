import React, { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "react-hot-toast";

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [capturedBlob, setCapturedBlob] = useState(null);
  const [data, setData] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const getCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Camera error:", err);
        toast.error("Unable to access camera.");
      }
    };
    getCamera();
  }, []);

  const capture = () => {
    const context = canvasRef.current.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, 300, 300);
    toast.success("Frame captured.");
  };

  const handleIdentify = () => {
    const toastId = toast.loading("Redirecting...");
    setTimeout(() => {
      toast.dismiss(toastId);
      navigate("/identify");
    }, 2000);
  };
  
  

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-950 via-blue-900 to-cyan-900 text-white px-4 py-10 md:px-16">
      <header className="mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-cyan-400 drop-shadow-lg">
          FaceTrack.AI
        </h1>
        <p className="mt-2 text-lg md:text-xl text-indigo-200">
          Real-time Criminal Detection System
        </p>
      </header>

      <div className="grid md:grid-cols-2 gap-12">
        {/* Camera Section */}
        <div className="space-y-4 bg-blue-800 bg-opacity-20 p-6 rounded-2xl shadow-lg">
          <video ref={videoRef} autoPlay className="rounded w-full border-2 border-blue-400" />
          <canvas ref={canvasRef} width="300" height="300" className="hidden" />

          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={capture}
              className="bg-indigo-600 hover:bg-indigo-700 py-2 rounded font-semibold"
            >
              Capture Frame
            </button>
            <button
              onClick={handleIdentify}
              className="bg-red-600 hover:bg-red-700 py-2 rounded font-semibold"
            >
              Identify Criminal
            </button>
          </div>
        </div>

        {/* Registration + Results Section */}
        <div className="space-y-6 bg-blue-800 bg-opacity-20 p-6 rounded-2xl shadow-lg">
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              const form = e.target;
              const nameInput = form.querySelector('input[name="name"]');
              const name = nameInput.value;

              if (!name || (!form.file.files[0] && !capturedBlob)) {
                toast.error("Please provide both name and image.");
                return;
              }

              const formData = new FormData();
              formData.append("name", name);
              formData.append("file", form.file.files[0] || capturedBlob, "captured.png");

              const toastId = toast.loading("Registering face...");

              try {
                const res = await fetch("http://127.0.0.1:8000/register/", {
                  method: "POST",
                  body: formData,
                });

                const result = await res.json();

                if (res.ok) {
                  toast.success(result.message || "Successfully registered.", { id: toastId });
                  form.reset();
                  setCapturedBlob(null);
                } else {
                  toast.error(result.message || "Registration failed.", { id: toastId });
                }
              } catch (err) {
                console.error(err);
                toast.error("Registration failed.", { id: toastId });
              }
            }}
            className="space-y-3"
          >
            <h2 className="text-2xl font-bold text-cyan-300">Register New Face</h2>

            <input
              type="text"
              name="name"
              placeholder="Enter full name"
              className="w-full p-2 rounded bg-white text-black"
              required
            />
            <label className="text-sm text-gray-300 block mb-1">Upload an image:</label>

            <div className="w-full">
            <label
                htmlFor="file-upload"
                className="block text-center bg-white text-black font-semibold py-2 px-4 rounded cursor-pointer hover:bg-gray-100 transition"
            >
                Choose File
            </label>
            <input
                id="file-upload"
                type="file"
                name="file"
                accept="image/*"
                className="hidden"
            />
            </div>


            <button
              type="button"
              onClick={() => {
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                const size = 128;
                canvas.width = size;
                canvas.height = size;
                ctx.drawImage(videoRef.current, 0, 0, size, size);
                canvas.toBlob((blob) => {
                  setCapturedBlob(blob);
                  toast.success("Face captured from webcam.");
                }, "image/png");
              }}
              className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded font-semibold w-full pointer"
            >
              Capture from Webcam
            </button>

            <button
              type="submit"
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold w-full"
            >
              Register Face
            </button>

            {capturedBlob && (
              <p className="text-green-400 text-sm font-mono">✓ Captured image ready.</p>
            )}
          </form>

          <Table data={data} />
        </div>
      </div>
    </main>
  );
}

function Table({ data }) {
  return (
    <div className="overflow-x-auto mt-6">
      <h3 className="text-xl font-semibold mb-2 text-cyan-300">Detection Results</h3>
      <table className="min-w-full bg-white text-black shadow-md rounded">
        <thead className="bg-indigo-100 text-indigo-900">
          <tr>
            <th className="py-2 px-4 border">Name</th>
            <th className="py-2 px-4 border">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td colSpan="2" className="text-center py-4 text-gray-500">
                No results yet.
              </td>
            </tr>
          ) : (
            data.map((item, index) => (
              <tr key={index} className="text-center">
                <td className="py-2 px-4 border">{item.name}</td>
                <td className="py-2 px-4 border">{item.confidence.toFixed(2)}%</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
