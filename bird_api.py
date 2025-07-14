#!/usr/bin/env python3
"""
FastAPI server for the Bird Call Classifier
Usage: uvicorn bird_api:app --reload
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from transformers import Wav2Vec2Processor, AutoModelForAudioClassification
import pickle

# Initialize FastAPI app
app = FastAPI(
    title="🐦 Bird Call Classifier API",
    description="Classify bird species from audio recordings using Wav2Vec2",
    version="1.0.0"
)

# Global variables for the model
model = None
processor = None
label_encoder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionResponse(BaseModel):
    predicted_species: str
    confidence: float
    all_probabilities: Dict[str, float]
    success: bool
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    available_species: List[str]


def load_model(model_path: str = "./best_bird_classifier"):
    """Load the trained Wav2Vec2 model and processor."""
    global model, processor, label_encoder

    try:
        # Load the transformer model and processor
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = AutoModelForAudioClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        # Load the label encoder
        label_encoder_path = None
        for file in Path("./models").glob("transformer_label_encoder_*.pkl"):
            label_encoder_path = file
            break

        if label_encoder_path and label_encoder_path.exists():
            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
        else:
            raise FileNotFoundError("Label encoder not found!")

        print(f"✅ Model loaded successfully on {device}")
        print(f"📋 Available species: {list(label_encoder.classes_)}")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def predict_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
    """Predict bird species from audio data."""
    if model is None or processor is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Ensure audio is the right length and format
        target_length = 10 * sample_rate  # 10 seconds
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')

        # Process audio
        inputs = processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        # Get species name
        pred_species = label_encoder.inverse_transform([pred_class])[0]

        # Get all probabilities
        all_probs = {
            species: float(prob)
            for species, prob in zip(label_encoder.classes_, probs[0])
        }

        return {
            'predicted_species': pred_species,
            'confidence': confidence,
            'all_probabilities': all_probs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    success = load_model()
    if not success:
        print("⚠️  Server started but model failed to load!")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐦 Bird Call Classifier</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background: linear-gradient(135deg, #2d5016 0%, #4a7c59 50%, #5d8a66 100%);
                min-height: 100vh;
                color: #ffffff;
            }
            .container { 
                background: rgba(255, 255, 255, 0.95); 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                color: #2d5016;
                position: relative;
                overflow: hidden;
            }
            .container::before {
                content: "🌲🐦🌲";
                position: absolute;
                top: 10px;
                right: 20px;
                font-size: 24px;
                opacity: 0.6;
            }
            .title-section {
                text-align: center;
                margin-bottom: 30px;
                position: relative;
            }
            .title-section h1 {
                margin: 0;
                color: #2d5016;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .bird-animation {
                display: inline-block;
                animation: fly 3s ease-in-out infinite;
                margin-left: 10px;
            }
            @keyframes fly {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-10px) rotate(5deg); }
            }
            .upload-area { 
                border: 3px dashed #4a7c59; 
                padding: 30px; 
                text-align: center; 
                margin: 20px 0; 
                border-radius: 10px;
                background: linear-gradient(45deg, #f8fff8 0%, #e8f5e8 100%);
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #2d5016;
                background: linear-gradient(45deg, #f0fff0 0%, #e0f0e0 100%);
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(77, 124, 89, 0.2);
            }
            .result { 
                background: linear-gradient(135deg, #ffffff 0%, #f8fff8 100%); 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px; 
                border-left: 4px solid #4a7c59;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            button { 
                background: linear-gradient(135deg, #4a7c59 0%, #2d5016 100%); 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3);
            }
            button:hover { 
                background: linear-gradient(135deg, #2d5016 0%, #1a2f0d 100%); 
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(45, 80, 22, 0.4);
            }
            .species-list { 
                columns: 2; 
                margin: 10px 0; 
                background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #4a7c59;
            }
            .nature-accent {
                color: #4a7c59;
                font-weight: bold;
            }
            .forest-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                opacity: 0.1;
                font-size: 60px;
                overflow: hidden;
                pointer-events: none;
            }
            .floating-emoji {
                position: absolute;
                animation: float 6s ease-in-out infinite;
            }
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                33% { transform: translateY(-20px) rotate(5deg); }
                66% { transform: translateY(-10px) rotate(-3deg); }
            }
        </style>
    </head>
    <body>
        <div class="forest-bg">
            <div class="floating-emoji" style="top: 10%; left: 5%;">🌲</div>
            <div class="floating-emoji" style="top: 20%; left: 85%; animation-delay: 1s;">🐦</div>
            <div class="floating-emoji" style="top: 60%; left: 10%; animation-delay: 2s;">🍃</div>
            <div class="floating-emoji" style="top: 80%; left: 80%; animation-delay: 3s;">🌿</div>
            <div class="floating-emoji" style="top: 40%; left: 90%; animation-delay: 4s;">🦅</div>
            <div class="floating-emoji" style="top: 70%; left: 15%; animation-delay: 5s;">🌲</div>
        </div>

        <div class="container">
            <div class="title-section">
                <h1>🐦 Bird Call Classifier<span class="bird-animation">🦜</span></h1>
                <p class="nature-accent">Upload an audio file to identify the bird species! 🎵</p>
            </div>

            <div class="upload-area">
                <p>🎤 <strong>Choose your bird call audio file</strong> 🎤</p>
                <input type="file" id="audioFile" accept="audio/*">
                <br><br>
                <button onclick="classifyAudio()">🔍 Classify Bird Call 🔍</button>
            </div>

            <div id="result" style="display:none;"></div>

            <h3>🌿 Available Species 🌿</h3>
            <div id="species-list" class="species-list">🔄 Loading...</div>
        </div>

        <script>
            // Load available species on page load
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    const speciesList = data.available_species.map(s => `<div>🐦 ${s}</div>`).join('');
                    document.getElementById('species-list').innerHTML = speciesList;
                });

            async function classifyAudio() {
                const fileInput = document.getElementById('audioFile');
                const resultDiv = document.getElementById('result');

                if (!fileInput.files[0]) {
                    alert('Please select an audio file first!');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                resultDiv.innerHTML = '<p>🔄 Analyzing your bird call... 🎵</p>';
                resultDiv.style.display = 'block';

                try {
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (result.success) {
                        let html = `
                            <h3>🎯 Prediction Results 🎯</h3>
                            <div class="result">
                                <h4>🐦 <span class="nature-accent">${result.predicted_species}</span> 🐦</h4>
                                <p><strong>🎯 Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            </div>
                            <h4>🌿 All Probabilities 🌿</h4>
                        `;

                        // Sort probabilities and show top 5
                        const sorted = Object.entries(result.all_probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 5);

                        sorted.forEach(([species, prob]) => {
                            const emoji = prob > 0.1 ? '🐦' : '🐤';
                            html += `<div class="result">${emoji} ${species}: ${(prob * 100).toFixed(1)}%</div>`;
                        });

                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `<div class="result">❌ Error: ${result.message}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result">❌ Error: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and model is loaded."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=str(device),
        available_species=list(label_encoder.classes_) if label_encoder else []
    )


@app.post("/predict/", response_model=PredictionResponse)
async def predict_bird_call(file: UploadFile = File(...)):
    """Predict bird species from uploaded audio file."""

    if model is None:
        return PredictionResponse(
            predicted_species="",
            confidence=0.0,
            all_probabilities={},
            success=False,
            message="Model not loaded"
        )

    # Check file type
    if not file.content_type.startswith('audio/'):
        return PredictionResponse(
            predicted_species="",
            confidence=0.0,
            all_probabilities={},
            success=False,
            message="Please upload an audio file"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load audio with librosa
        try:
            audio_data, sample_rate = librosa.load(tmp_path, sr=16000, duration=10)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        # Make prediction
        result = predict_audio(audio_data, sample_rate)

        return PredictionResponse(
            predicted_species=result['predicted_species'],
            confidence=result['confidence'],
            all_probabilities=result['all_probabilities'],
            success=True
        )

    except Exception as e:
        return PredictionResponse(
            predicted_species="",
            confidence=0.0,
            all_probabilities={},
            success=False,
            message=f"Error processing audio: {str(e)}"
        )


@app.get("/species/")
async def get_available_species():
    """Get list of all species the model can classify."""
    if label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "species": list(label_encoder.classes_),
        "count": len(label_encoder.classes_)
    }


if __name__ == "__main__":
    print("🚀 Starting Bird Call Classifier API...")
    print("📖 Once running, visit http://localhost:8000 to use the web interface")
    print("📚 API docs available at http://localhost:8000/docs")

    uvicorn.run(
        "bird_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )