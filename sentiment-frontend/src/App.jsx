import React from "react";
import { useState } from "react";
import './App.css';
import axios from 'axios';
import img from '../assets/trans_vs_llm.png';

function App() {

  const [inputText,setInputText] = useState("");
  const [result,setResult] = useState(null);
  const [loading,SetLoading] = useState(false);

  const handleAnalyze = async () =>{
    SetLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/predict",{text:inputText});
      setResult(response.data);
    } catch (error) {
      console.error("Error analyzing sentiment: ",error)
    }finally{
      SetLoading(false)
    }
  }

  return (
    <>
    <div className="main-screen flex items-center justify-center bg-gray-300">
            <div className="bg-white p-8 rounded-2xl shadow-md w-full max-w-2xl">
                <div className="flex gap-5 items-center">
                    <img src={img} className="h-30"/>
                    <h1 className="text-2xl font-bold mb-4 text-black">Sentiment Analysis</h1>
                </div>
                <textarea
                    className="w-full p-4 border rounded-md mb-4 mt-5 border-black text-black"
                    rows="5"
                    placeholder="Enter text here..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                />
                <button
                    onClick={handleAnalyze}
                    className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
                    disabled={loading}
                >
                    {loading ? "Analyzing..." : "Analyze Sentiment"}
                </button>

                {result && (
                  <div className="mt-4">
                    <h2 className="text-5xl font-semibold text-black text-center">Results</h2>
                    <div className="mt-2 main-results flex flex-row gap-15">
                        <div className="mt-6">
                            <p className="text-black model-title text-2xl font-semibold"><h2>Transformer Model:</h2></p>
                            <p className="text-black mt-3"><b>Text</b>: {result.transformer_report.text}</p>
                            <p className="text-black mt-3"><b>Sentiment</b>: {result.transformer_report.sentiment}</p>
                            <p className="text-black mt-2"><b>Confidence</b>: {result.transformer_report.confidence}%</p>
                        </div>
                        <div className="mt-6">
                            <p className="text-black model-title text-2xl font-semibold"><h2>LLM Model:</h2></p>
                            <p className="text-black mt-3"><b>Text</b>: {result.llm_report.text}</p>
                            <p className="text-black mt-3"><b>Sentiment</b>: {result.llm_report.sentiment}</p>
                            <p className="text-black mt-2"><b>Confidence</b>: {result.llm_report.confidence}%</p>
                        </div>
                    </div>
                  </div>
                )}
            </div>
        </div>
    </>
  )
}

export default App
