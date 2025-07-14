from __future__ import annotations
import json
import os
import pickle
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List
import librosa
import numpy as np
import requests

from config import data_dir, species_to_scrape, SPECTROGRAM_SHAPE

BASE_URL = "https://www.xeno-canto.org/api/2/recordings"


def normalize_name(s):
    """
     This makes the names of things similar to clarity
    """
    if not isinstance(s, str): return ""
    return s.strip().replace("-", " ").replace("_", " ").replace("  ", " ").lower()


class AudioFeatureExtractor:
    """
    This class extracts audio features and builds the spectrograms
    """

    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_path: str):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            return {
                "mfcc_mean": np.mean(mfcc, axis=1),
                "mfcc_std": np.std(mfcc, axis=1),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                "tempo": librosa.beat.beat_track(y=y, sr=sr)[0],
                "chroma_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1),
            }
        except Exception:
            return None

    def extract_spectrogram(self, audio_path: str, n_mels: int = 128):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
            return librosa.power_to_db(mel, ref=np.max)
        except Exception:
            return None


class BirdCallDatabase:
    """
    This class builds the sqlite database.
    """

    def __init__(self, db_path: str = "bird_calls.db"):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS recordings(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    xc_id TEXT UNIQUE NOT NULL,
                    english_name TEXT,
                    genus TEXT,
                    species TEXT,
                    country TEXT,
                    location TEXT,
                    length TEXT,
                    quality TEXT,
                    type TEXT,
                    local_file TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS embeddings(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id INTEGER,
                    embedding_type TEXT,
                    vector_data BLOB,
                    vector_shape TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(recording_id) REFERENCES recordings(id)
                );
                CREATE TABLE IF NOT EXISTS features(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id INTEGER,
                    feature_type TEXT,
                    feature_values TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(recording_id) REFERENCES recordings(id)
                );
                CREATE TABLE IF NOT EXISTS species_attempts(
                    species TEXT PRIMARY KEY,
                    last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    recording_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending'
                );
                CREATE INDEX IF NOT EXISTS idx_english_name ON recordings(english_name);
                CREATE INDEX IF NOT EXISTS idx_xc_id ON recordings(xc_id);
                CREATE INDEX IF NOT EXISTS idx_embedding_type ON embeddings(embedding_type);
                """
            )

    def get_all_embeddings(self, embedding_type: str):
        rows = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.vector_data, r.english_name
                FROM embeddings e
                JOIN recordings r ON r.id = e.recording_id
                WHERE e.embedding_type = ?
                """,
                (embedding_type,),
            ).fetchall()
        out = []
        for rec_id, blob, name in rows:
            arr = pickle.loads(blob)
            out.append((rec_id, arr, name))
        return out

    def get_ids_and_labels(self, embedding_type: str) -> list[tuple[int, str]]:
        """Fetches a list for a given embedding type."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """
                SELECT e.id, r.english_name
                FROM embeddings e
                JOIN recordings r ON r.id = e.recording_id
                WHERE e.embedding_type = ?
                """,
                (embedding_type,),
            ).fetchall()

    def get_embedding_by_id(self, embedding_id: int) -> np.ndarray | None:
        """Fetches a single embedding by its primary key."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT vector_data FROM embeddings WHERE id = ?", (embedding_id,)
            ).fetchone()
        if row:
            return pickle.loads(row[0])
        return None

    def recording_exists(self, xc_id: str):
        with sqlite3.connect(self.db_path) as conn:
            return (
                    conn.execute("SELECT 1 FROM recordings WHERE xc_id = ? LIMIT 1", (xc_id,)).fetchone()
                    is not None
            )

    def english_name_count(self, name: str):
        with sqlite3.connect(self.db_path) as conn:
            val = conn.execute(
                "SELECT COUNT(*) FROM recordings WHERE english_name = ?", (name,)
            ).fetchone()
        return val[0] if val else 0

    def mark_species_attempt(self, species: str, status: str, recording_count: int = 0):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO species_attempts
                       (species, last_attempt, recording_count, status)
                       VALUES(?, CURRENT_TIMESTAMP, ?, ?)""",
                (species, recording_count, status),
            )

    def insert_recording(self, data: Dict):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT OR REPLACE INTO recordings
                       (xc_id, english_name, genus, species, country, location,
                        length, quality, type, local_file, file_size)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data["xc_id"],
                    data["english_name"],
                    data["genus"],
                    data["species"],
                    data["country"],
                    data["location"],
                    data["length"],
                    data["quality"],
                    data["type"],
                    data["local_file"],
                    data.get("file_size"),
                ),
            )
            return cur.lastrowid

    def store_embedding(
            self,
            recording_id: int,
            arr: np.ndarray,
            embedding_type: str = "spectrogram",
            model_version: str = "v1.0",
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO embeddings
                       (recording_id, embedding_type, vector_data, vector_shape, model_version)
                       VALUES(?,?,?,?,?)""",
                (
                    recording_id,
                    embedding_type,
                    pickle.dumps(arr),
                    json.dumps(arr.shape),
                    model_version,
                ),
            )

    def store_features(self, recording_id: int, feats: Dict[str, List]):
        with sqlite3.connect(self.db_path) as conn:
            for k, v in feats.items():
                conn.execute(
                    """INSERT OR REPLACE INTO features
                           (recording_id, feature_type, feature_values)
                           VALUES(?,?,?)""",
                    (recording_id, k, json.dumps(v)),
                )


def _pad_or_trim(spec: np.ndarray, width: int = 432):
    """Pad or trim spectrogram to the correct width."""
    if spec.shape[1] < width:
        pad = width - spec.shape[1]
        return np.pad(spec, ((0, 0), (0, pad)), mode="constant")
    return spec[:, :width]


def process_and_store_audio(content: bytes, meta: Dict, db: BirdCallDatabase):
    fx = AudioFeatureExtractor()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        rec_id = db.insert_recording(meta)

        # Extract spectrogram with correct shape from the start
        spec = fx.extract_spectrogram(tmp_path, n_mels=SPECTROGRAM_SHAPE[0])
        if spec is not None:
            spec = _pad_or_trim(spec, width=SPECTROGRAM_SHAPE[1])
            db.store_embedding(rec_id, spec, "spectrogram")

        # Extract raw audio
        y, sr = librosa.load(tmp_path, sr=16000, duration=20)
        if len(y) >= sr:
            db.store_embedding(rec_id, y, "raw_audio")

        # Extract features
        feats = fx.extract_features(tmp_path)
        if feats is not None:
            nice = {k: (v.tolist() if isinstance(v, np.ndarray) else [v]) for k, v in feats.items()}
            db.store_features(rec_id, nice)
        return True
    except Exception:
        return False
    finally:
        os.unlink(tmp_path)


def download_and_process_recording(rec: Dict, name: str, folder: Path, db: BirdCallDatabase):
    if rec.get("q") == "D":
        return False
    xc_id = rec["id"]
    if db.recording_exists(xc_id):
        return False
    url = rec["file"]
    try:
        rsp = requests.get(url, timeout=20)
        rsp.raise_for_status()
    except Exception:
        return False
    file_name = f"{xc_id}_{rec.get('type', 'unknown').replace(' ', '_')}.mp3"
    meta = {
        "xc_id": xc_id,
        "english_name": name,
        "genus": rec.get("gen"),
        "species": rec.get("sp"),
        "country": rec.get("cnt"),
        "location": rec.get("loc"),
        "length": rec.get("length"),
        "quality": rec.get("q"),
        "type": rec.get("type"),
        "local_file": str(folder / file_name),
        "file_size": len(rsp.content),
    }
    return process_and_store_audio(rsp.content, meta, db)


def download_and_process_species():
    db = BirdCallDatabase()
    with sqlite3.connect(db.db_path) as conn:
        present_birds = set(
            normalize_name(row[0]) for row in conn.execute(
                "SELECT DISTINCT english_name FROM recordings WHERE english_name IS NOT NULL AND english_name != ''"
            )
        )

    print("DB species present:", sorted(present_birds))
    print("species_to_scrape:", sorted(normalize_name(s) for s in species_to_scrape))

    missing_species = []
    for species in species_to_scrape:
        if normalize_name(species) not in present_birds:
            missing_species.append(species)
        else:
            print(f"Skipping {species} – already flying around the db")
            db.mark_species_attempt(species, 'success', db.english_name_count(species))

    print(f"Will capture {len(missing_species)} missing species from the wild: {missing_species}")

    for species in missing_species:
        print(f"\nDownloading {species} from xeno-canto…")
        try:
            resp = requests.get(BASE_URL, params={"query": species}, timeout=15)
            data = resp.json()
            recordings = data.get("recordings", [])
            print(f"xeno-canto for {species}: got {len(recordings)} recordings")
        except Exception as e:
            print(f"API error for {species}: {e}")
            db.mark_species_attempt(species, 'error', 0)
            continue
        if not recordings:
            print(f"No recordings on xeno-canto for {species}. Typo?")
            db.mark_species_attempt(species, 'no_data', 0)
            continue
        folder = Path(data_dir) / species
        folder.mkdir(parents=True, exist_ok=True)
        processed = 0
        for rec in recordings:
            if download_and_process_recording(rec, species, folder, db):
                processed += 1
        status = 'success' if processed else 'error'
        db.mark_species_attempt(species, status, processed)
        print(f"{processed} recordings stored for {species}" if processed else
              f"Failed to store recordings for {species}")

    print("Done downloading species!")
    return db