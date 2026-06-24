import React, { useState } from "react";
import axios from "axios";

function Home() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const uploadImage = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const response = await axios.post(
        process.env.REACT_APP_API_URL,
        formData
      );

      setPrediction(response.data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed");
    }

    setLoading(false);
  };

  return (
  <div style={{ textAlign: "center", marginTop: "50px" }}>
    <h1>Potato Disease Classifier</h1>

    <input
      type="file"
      accept="image/*"
      onChange={(e) => setFile(e.target.files[0])}
    />

    <br />
    <br />

    {file && (
      <img
        src={URL.createObjectURL(file)}
        alt="preview"
        width="300"
      />
    )}

    <br />
    <br />

   <button
  onClick={uploadImage}
  disabled={loading}
>
  {loading ? "Predicting..." : "Predict"}
</button>

    {loading && (
      <h3>Loading...</h3>
    )}

   {prediction && (
  <div>
    <h2>
      Disease: {prediction.pred_class}
    </h2>

    <h3>
      Confidence:
      {" "}
      {(prediction.confidence * 100).toFixed(2)}%
    </h3>

    <div
      style={{
        width: "300px",
        margin: "auto",
        border: "1px solid black"
      }}
    >
      <div
        style={{
          width: `${prediction.confidence * 100}%`,
          height: "20px",
          backgroundColor: "green"
        }}
      />
    </div>

  </div>
)}
  </div>
);
}

export default Home;