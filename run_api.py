#!/usr/bin/env python3
"""
This runs the API
"""

import subprocess
import sys
from pathlib import Path


def check_model_exists():
    """Check if the trained model exists."""
    model_path = Path("./best_bird_classifier")
    models_dir = Path("./models")

    if not model_path.exists():
        print("❌ Model not found at ./best_bird_classifier")
        print("🔧 Please train the transformer model first:")
        print("   python run_training.py")
        return False

    if not models_dir.exists() or not any(models_dir.glob("transformer_label_encoder_*.pkl")):
        print("❌ Label encoder not found in ./models/")
        print("🔧 Please train the transformer model first:")
        print("   python run_training.py")
        return False

    print("✅ Model and label encoder found!")
    return True


def install_requirements():
    """Install API requirements."""
    print("📦 Installing API requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_api.txt"])
        print("✅ Requirements installed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False


def run_api():
    """Run the FastAPI server."""
    print("🚀 Starting Bird Call Classifier API...")
    print("🌐 Web interface: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop")

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "bird_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 API server stopped!")


def main():
    print("🐦 Bird Call Classifier API Launcher")
    print("=" * 40)

    # Check if model exists
    if not check_model_exists():
        return

    # Install requirements
    if not install_requirements():
        return

    # Run the API
    run_api()


if __name__ == "__main__":
    main()