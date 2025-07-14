# birdcall_classifier
*This program builds classifiers to distinguish different bird calls, similar to the Merlin app. It supports both CNN and Wav2Vec2 transformer classifiers.*

---

## HOW TO USE

### 1. **Change the config settings to determine which classifiers you want to build**
The defaults are to run both the CNN and the Wav2Vec2 classifiers with grid search. Turn any of these false to run only one section.

### 2. **Simply run:**
```bash
python run_training.py
```
This will automatically download the DB.

### 3. **Start the web API (optional):**
```bash
python run_api.py
```
Then go to `http://localhost:8000` to upload audio files and classify bird calls through a web interface. Highly suggest to use the birds the app is trained on.

---

## REQUIREMENTS
Before using the PI, install requirements via this:
```bash
pip install -r requirements.txt
```
Here is the full list of req's:

Core ML & audio processing
torch>=2.7.0
torchaudio>=2.7.0
transformers>=4.51.0
librosa>=0.11.0
scikit-learn>=1.6.0

Data handling & utilities
numpy>=2.2.0
pandas>=2.2.0
requests>=2.32.0
tqdm>=4.67.0

Visualization (for training results)
matplotlib>=3.10.0
seaborn>=0.13.0

FastAPI & web server (for the API)
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

Optional: more viz
plotly>=6.2.0



## FEATURES
- Downloads bird call data from Xeno-Canto automatically
- Trains CNN and/or Wav2Vec2 models with grid search
- Web interface for testing model
- Builds a sqlite3 db for bird calls used in the model

## ABOUT THE MODELS WHEN RUN FROM THE BASE CONFIG
- The 3-layer CNN has about an 80% F1 and accuracy
- The WAV2VEC2 has about a 90% F1 and accuracy
- The CNN uses mel-spectrograms, Wav2Vec2 uses raw audio
- Both are trained on about 10 seconds per call (with a few thousand calls in total)
- Wav2Vec2 performs better but takes longer to train

