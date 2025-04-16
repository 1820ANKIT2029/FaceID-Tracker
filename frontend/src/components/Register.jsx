import React, { useRef, useState } from 'react';

const Register = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [name, setName] = useState('');
  const [message, setMessage] = useState('');

  const captureAndRegister = async () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    const formData = new FormData();
    formData.append('file', blob, `${name}.png`);
    formData.append('name', name);

    try {
      const response = await fetch('http://127.0.0.1:8000/register/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setMessage("Criminal Registered Successfully");
        setName('');
      } else {
        setMessage("Registration Failed");
      }
    } catch (err) {
      console.error(err);
      setMessage("Error occurred during registration");
    }
  };

  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        videoRef.current.srcObject = stream;
      });
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">Register Criminal</h2>
      <input
        type="text"
        value={name}
        onChange={e => setName(e.target.value)}
        placeholder="Enter Name"
        className="mb-2 p-2 text-black"
      />
      <div>
        <video ref={videoRef} autoPlay muted onCanPlay={startVideo} className="w-64 h-48" />
        <canvas ref={canvasRef} width="128" height="128" hidden />
      </div>
      <button onClick={captureAndRegister} className="bg-blue-600 px-4 py-2 mt-2">Register</button>
      {message && <p className="mt-2 text-sm text-green-400">{message}</p>}
    </div>
  );
};

export default Register;
