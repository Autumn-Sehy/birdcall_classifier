import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from tqdm import tqdm
from download_to_db import BirdCallDatabase, AudioFeatureExtractor
import pandas as pd
from config import SPECTROGRAM_SHAPE


class BirdAudioDataset(Dataset):
    def __init__(self, audio_ids, labels, processor, label_encoder, db_path="bird_calls.db"):
        self.db_path = db_path
        self.audio_ids = audio_ids
        self.labels = [label_encoder.transform([label])[0] for label in labels]
        self.processor = processor
        self.db = None

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        if self.db is None:
            self.db = BirdCallDatabase(self.db_path)

        audio_id = self.audio_ids[idx]
        audio = self.db.get_embedding_by_id(audio_id)
        label = self.labels[idx]

        target_length = 160000
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

        audio = audio.astype(np.float32)
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt",
            padding=False, truncation=False, max_length=target_length
        )
        input_values = inputs.input_values.squeeze(0)
        if input_values.size(0) != target_length:
            if input_values.size(0) > target_length:
                input_values = input_values[:target_length]
            else:
                input_values = torch.nn.functional.pad(input_values, (0, target_length - input_values.size(0)))

        return {'input_values': input_values, 'labels': torch.tensor(label, dtype=torch.long)}


class BirdCallDataset(Dataset):
    def __init__(self, spectrograms: List[np.ndarray], labels: List[str], label_encoder: LabelEncoder):
        self.spectrograms = spectrograms
        self.labels = [label_encoder.transform([label])[0] for label in labels]

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spectrogram = torch.FloatTensor(self.spectrograms[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return spectrogram, label


def prepare_train_dev_test_splits(db_path="bird_calls.db", min_total_samples_per_class=30):
    db = BirdCallDatabase(db_path)
    spec_embs = db.get_all_embeddings('spectrogram')
    audio_info = db.get_ids_and_labels('raw_audio')

    spectrograms, spec_labels = zip(*[(s, l) for _, s, l in spec_embs]) if spec_embs else ([], [])
    audio_ids, audio_labels = zip(*[(rec_id, label) for rec_id, label in audio_info]) if audio_info else ([], [])

    if not spec_labels:
        raise ValueError("No spectrogram data found!")

    from collections import Counter
    counts = Counter(spec_labels)
    valid_classes = {label for label, c in counts.items() if c >= min_total_samples_per_class}

    Xs, Ys = zip(*[(s, l) for s, l in zip(spectrograms, spec_labels) if l in valid_classes])
    Xa_ids, Ya = zip(*[(a_id, l) for a_id, l in zip(audio_ids, audio_labels) if l in valid_classes])

    Xs, Ys, Xa_ids, Ya = list(Xs), list(Ys), list(Xa_ids), list(Ya)

    Xs_temp, Xs_test, Ys_temp, Ys_test = train_test_split(Xs, Ys, test_size=0.2, random_state=42, stratify=Ys)
    Xs_train, Xs_dev, Ys_train, Ys_dev = train_test_split(Xs_temp, Ys_temp, test_size=0.25, random_state=42,
                                                          stratify=Ys_temp)

    Xa_ids_temp, Xa_ids_test, Ya_temp, Ya_test = train_test_split(Xa_ids, Ya, test_size=0.2, random_state=42,
                                                                  stratify=Ya)
    Xa_ids_train, Xa_ids_dev, Ya_train, Ya_dev = train_test_split(Xa_ids_temp, Ya_temp, test_size=0.25, random_state=42,
                                                                  stratify=Ya_temp)

    return {
        'spectrograms': {
            'X_train': Xs_train, 'y_train': Ys_train,
            'X_dev': Xs_dev, 'y_dev': Ys_dev,
            'X_test': Xs_test, 'y_test': Ys_test
        },
        'raw_audio': {
            'X_train_ids': Xa_ids_train, 'y_train': Ya_train,
            'X_dev_ids': Xa_ids_dev, 'y_dev': Ya_dev,
            'X_test_ids': Xa_ids_test, 'y_test': Ya_test
        },
        'valid_classes': list(valid_classes)
    }


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, input_shape=SPECTROGRAM_SHAPE):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv_layers(x))


class BirdCallClassifier:
    def __init__(self, db_path="bird_calls.db"):
        self.db = BirdCallDatabase(db_path)
        self.label_encoder = LabelEncoder()
        self.models = {}

    def train_cnn_with_splits(self, data_splits: Dict, epochs: int = 20) -> Dict:
        data = data_splits['spectrograms']
        X_train, y_train = data['X_train'], data['y_train']
        X_dev, y_dev = data['X_dev'], data['y_dev']
        X_test, y_test = data['X_test'], data['y_test']

        all_labels = y_train + y_dev + y_test
        self.label_encoder.fit(all_labels)
        num_classes = len(self.label_encoder.classes_)

        train_dataset = BirdCallDataset(X_train, y_train, self.label_encoder)
        dev_dataset = BirdCallDataset(X_dev, y_dev, self.label_encoder)
        test_dataset = BirdCallDataset(X_test, y_test, self.label_encoder)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_losses, train_accuracies = [], []
        dev_losses, dev_accuracies = [], []

        for epoch in range(epochs):
            model.train()
            epoch_loss, correct, total = 0, 0, 0
            for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)
            train_losses.append(epoch_loss / len(train_loader))
            train_accuracies.append(correct / total)

            model.eval()
            dev_loss, dev_correct, dev_total = 0, 0, 0
            with torch.no_grad():
                for X, y in dev_loader:
                    X, y = X.to(device), y.to(device)
                    output = model(X)
                    loss = criterion(output, y)
                    dev_loss += loss.item()
                    dev_correct += (output.argmax(1) == y).sum().item()
                    dev_total += y.size(0)
            dev_losses.append(dev_loss / len(dev_loader))
            dev_accuracies.append(dev_correct / dev_total)

            print(f"Epoch {epoch + 1}: Train Acc {train_accuracies[-1]:.4f} | Dev Acc {dev_accuracies[-1]:.4f}")

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        test_acc = np.mean(np.array(all_preds) == np.array(all_targets))
        test_f1 = f1_score(all_targets, all_preds, average='weighted')
        print("\n--- CNN Test Set Results ---")
        print(classification_report(all_targets, all_preds, target_names=self.label_encoder.classes_))

        # Pretty confusion matrix for CNN
        import pandas as pd
        cm = confusion_matrix(all_targets, all_preds)
        cm_df = pd.DataFrame(
            cm,
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print(cm_df)

        self.models['cnn'] = model
        return {
            'model': model,
            'dev_accuracy': dev_accuracies[-1],
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'dev_losses': dev_losses,
            'dev_accuracies': dev_accuracies,
            'predictions': all_preds,
            'true_labels': all_targets,
            'classification_report': classification_report(all_targets, all_preds,
                                                           target_names=self.label_encoder.classes_, output_dict=True)
        }

    def predict(self, audio_path: str, model_type: str = 'cnn') -> Dict:
        if model_type != 'cnn' or 'cnn' not in self.models:
            raise ValueError(f"Model {model_type} not trained!")
        feature_extractor = AudioFeatureExtractor()
        spec = feature_extractor.extract_spectrogram(audio_path)
        if spec is None: return None

        target_shape = SPECTROGRAM_SHAPE
        if spec.shape[1] < target_shape[1]:
            pad = target_shape[1] - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')
        else:
            spec = spec[:, :target_shape[1]]

        X = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(next(self.models['cnn'].parameters()).device)
        self.models['cnn'].eval()
        with torch.no_grad():
            probs = torch.softmax(self.models['cnn'](X), dim=1)[0]
            pred_class = torch.argmax(probs).item()
        return {
            'predicted_species': self.label_encoder.inverse_transform([pred_class])[0],
            'confidence': probs[pred_class].item(),
            'all_probabilities': dict(zip(self.label_encoder.classes_, probs.cpu().numpy()))
        }


# ========== Suppress TQDM Context Manager ==========

from contextlib import contextmanager
import tqdm as tqdm_module

@contextmanager
def suppress_tqdm():
    """
    Context manager to temporarily suppress tqdm bars for cleaner eval/predict output.
    """
    orig = tqdm_module.tqdm
    tqdm_module.tqdm = lambda *a, **k: orig(*a, **{**k, "disable": True})
    yield
    tqdm_module.tqdm = orig

# ========== TransformerBirdClassifier ==========

class TransformerBirdClassifier:
    def __init__(self, db_path="bird_calls.db"):
        self.db_path = db_path
        self.label_encoder = LabelEncoder()
        self.processor = None
        self.model = None

    def setup_transformer_model(self, model_name="facebook/wav2vec2-base"):
        from transformers import Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def train_transformer_with_splits(self, data_splits: Dict, epochs: int = 10, batch_size: int = 8) -> Dict:
        from transformers import (
            AutoModelForAudioClassification, TrainingArguments, Trainer, EarlyStoppingCallback, AutoConfig
        )

        audio_data = data_splits['raw_audio']
        X_train_ids, y_train = audio_data['X_train_ids'], audio_data['y_train']
        X_dev_ids, y_dev = audio_data['X_dev_ids'], audio_data['y_dev']
        X_test_ids, y_test = audio_data['X_test_ids'], audio_data['y_test']

        all_labels = y_train + y_dev + y_test
        self.label_encoder.fit(all_labels)
        num_classes = len(self.label_encoder.classes_)

        train_dataset = BirdAudioDataset(X_train_ids, y_train, self.processor, self.label_encoder, self.db_path)
        dev_dataset = BirdAudioDataset(X_dev_ids, y_dev, self.processor, self.label_encoder, self.db_path)
        test_dataset = BirdAudioDataset(X_test_ids, y_test, self.processor, self.label_encoder, self.db_path)

        model_config = AutoConfig.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_classes
        )
        model_config.gradient_checkpointing = False

        self.model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base",
            config=model_config,
            ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir='./bird_classifier_checkpoints',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=2,
            fp16=torch.cuda.is_available(),
            report_to=None,
            disable_tqdm=False,
            gradient_checkpointing=True,
            save_total_limit=2,
        )

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            return {'eval_accuracy': np.mean(preds == labels), 'eval_f1': f1_score(labels, preds, average='weighted')}

        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset,
            eval_dataset=dev_dataset, compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()

        print("✔ Training complete. Evaluating on dev set...")
        with suppress_tqdm():
            eval_results = trainer.evaluate()

        print("\nEvaluating on final test set...")
        with suppress_tqdm():
            test_predictions = trainer.predict(test_dataset)
        y_true = test_predictions.label_ids
        y_pred = np.argmax(test_predictions.predictions, axis=1)

        print("\n--- Transformer Test Set Results ---")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))

        # Pretty confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print(cm_df)

        print("-" * 30)

        self.model.save_pretrained('./best_bird_classifier')
        self.processor.save_pretrained('./best_bird_classifier')
        return {
            'model': self.model,
            'dev_results': eval_results,
            'trainer': trainer
        }

    def predict_transformer(self, audio_path: str) -> Dict:
        if self.model is None:
            raise ValueError("Transformer model not trained yet!")
        y, _ = librosa.load(audio_path, sr=16000, duration=10)
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()
        pred_species = self.label_encoder.inverse_transform([pred_class])[0]
        all_probs = {s: float(p) for s, p in zip(self.label_encoder.classes_, probs[0])}
        return {
            'predicted_species': pred_species,
            'confidence': confidence,
            'all_probabilities': all_probs
        }