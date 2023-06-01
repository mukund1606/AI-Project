"use client";
import Canvas from "@/components/canvas";
import { useRef, useEffect, useState } from "react";

const numberNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];

export default function Home() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [isMobile, setIsMobile] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  async function processImage() {
    setPrediction(null);
    const data = canvasRef.current.toDataURL("image/png");
    const formDataToSubmit = new FormData();
    formDataToSubmit.append("data", data);
    try {
      const res = await fetch("http://localhost:5501/", {
        // const res = await fetch("https://ai-project-api.mukund.page/", {
        method: "POST",
        body: formDataToSubmit,
      });
      if (!res.ok) {
        throw new Error("Error");
      }
      const data = await res.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.log(err);
    }
  }
  useEffect(() => {
    if (typeof window !== "undefined") {
      if (window.innerWidth < 768) {
        setIsMobile(true);
      }
    }
    setIsLoading(false);
  }, []);
  return (
    !isLoading && (
      <main className="flex flex-col items-center min-h-screen gap-5 p-12">
        <div className="flex flex-col items-center gap-2 px-2 py-5 text-center bg-white rounded-xl">
          <h1 className="text-3xl font-bold text-black">Number Predictor</h1>
          <Canvas
            width={isMobile ? 300 : 500}
            height={isMobile ? 300 : 500}
            canvasRef={canvasRef}
            ctxRef={ctxRef}
            isMobile={isMobile}
          />
          <div className="flex gap-2">
            {/* Reset Button */}
            <button
              className="px-4 py-2 mt-4 text-white bg-black rounded"
              onClick={() => {
                setPrediction(null);
                ctxRef.current.clearRect(
                  0,
                  0,
                  canvasRef.current.width,
                  canvasRef.current.height
                );
              }}
            >
              Reset
            </button>
            <button
              className="px-4 py-2 mt-4 text-white bg-black rounded"
              onClick={processImage}
            >
              Predict
            </button>
          </div>
          <br className="p-2" />
          {prediction && (
            <div className="flex flex-col justify-center px-8 py-3 text-center bg-black rounded-xl">
              <h1 className="text-3xl font-extrabold text-white">Prediction</h1>
              <h1 className="text-[80px] font-medium text-white">
                {prediction}
              </h1>
              <h1 className="text-[50px] font-medium text-white">
                {numberNames[prediction]}
              </h1>
            </div>
          )}
        </div>
      </main>
    )
  );
}
