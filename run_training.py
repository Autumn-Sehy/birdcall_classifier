import torch
from itertools import product
from train_classifiers import BirdCallClassifier, TransformerBirdClassifier, prepare_train_dev_test_splits
from config import (
    TRAIN_CNN, TRAIN_TRANSFORMER, USE_GRID_SEARCH,
    CNN_EPOCHS, TRANSFORMER_EPOCHS, TRANSFORMER_BATCH_SIZE, TRANSFORMER_MODEL,
    CNN_GRID, TRANSFORMER_GRID
)
import json, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import os
from download_to_db import download_and_process_species


def check_and_build_db(db_path="bird_calls.db"):
    """
    Ensures the database exists. If not, download and build it.
    """
    db_exists = Path(db_path).is_file()

    if not db_exists:
        print(f"No database found. Building it - this will take some time!")
        download_and_process_species()
    return True


def save_model(classifier, model_type, config, results, out_dir="./models"):
    """
    This saves the best and most recent models as whatever time they were made.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == "cnn":
        torch.save(classifier.models['cnn'].state_dict(), f"{out_dir}/cnn_model_{ts}.pth")
        with open(f"{out_dir}/label_encoder_{ts}.pkl", "wb") as f:
            pickle.dump(classifier.label_encoder, f)
    elif model_type == "transformer":
        classifier.model.save_pretrained(f"{out_dir}/transformer_model_{ts}")
        classifier.processor.save_pretrained(f"{out_dir}/transformer_model_{ts}")
        with open(f"{out_dir}/transformer_label_encoder_{ts}.pkl", "wb") as f:
            pickle.dump(classifier.label_encoder, f)
    with open(f"{out_dir}/{model_type}_results_{ts}.json", "w") as f:
        json.dump({
            "config": config,
            "results": {k: v for k, v in results.items() if k not in ["model", "trainer"]},
            "timestamp": ts
        }, f, indent=2, default=str)


def run_cnn(config, data_splits):
    """
    This trains the CNN!
    """
    print(f"Training CNN on tweets - the bird kind, not the app kind.")
    classifier = BirdCallClassifier()
    results = classifier.train_cnn_with_splits(data_splits, epochs=config['epochs'])
    return classifier, results


def run_transformer(config, data_splits):
    print(f"Training Wav2Vec2! This is slower than the CNN but is more accurate.")
    classifier = TransformerBirdClassifier()
    classifier.setup_transformer_model(config['model_name'])
    results = classifier.train_transformer_with_splits(
        data_splits, epochs=config['epochs'], batch_size=config['batch_size']
    )
    return classifier, results


def grid_search(train_fn, param_grid, data_splits):
    """
    This uses grid search on the CNN/Wav2vec2 to test out different combinations of parameters.
    """
    keys, values = zip(*param_grid.items())
    combos = list(product(*values))
    best_metric, best_config, best_classifier, all_results = None, None, None, []
    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        print(f"\n🔍 [Grid {i + 1}/{len(combos)}] Training with config: {config}\n")
        classifier, results = train_fn(config, data_splits)
        metric = results.get('dev_accuracy') or results.get('dev_results', {}).get('eval_accuracy')
        all_results.append({"config": config, "metric": metric})
        if best_metric is None or (metric is not None and metric > best_metric):
            best_metric, best_config, best_classifier = metric, config, classifier
    return best_classifier, best_config, all_results


def main():
    print("🐦 Training on Bird Calls 🐦")

    # Check if database exists, build if needed
    check_and_build_db(db_path="bird_calls.db")

    # Split data
    data_splits = prepare_train_dev_test_splits()
    os.makedirs("./models", exist_ok=True)

    # Train according to settings in the CONFIG file
    if USE_GRID_SEARCH:
        if TRAIN_CNN:
            cnn, cnn_config, cnn_results = grid_search(run_cnn, CNN_GRID, data_splits)
            save_model(cnn, "cnn", cnn_config, {"grid_search_results": cnn_results})
        if TRAIN_TRANSFORMER:
            transformer, t_config, t_results = grid_search(run_transformer, TRANSFORMER_GRID, data_splits)
            save_model(transformer, "transformer", t_config, {"grid_search_results": t_results})
    else:
        if TRAIN_CNN:
            config = {"epochs": CNN_EPOCHS}
            cnn, results = run_cnn(config, data_splits)
            save_model(cnn, "cnn", config, results)
        if TRAIN_TRANSFORMER:
            config = {
                "epochs": TRANSFORMER_EPOCHS,
                "batch_size": TRANSFORMER_BATCH_SIZE,
                "model_name": TRANSFORMER_MODEL
            }
            transformer, results = run_transformer(config, data_splits)
            save_model(transformer, "transformer", config, results)
    print("Training done!")


if __name__ == "__main__":
    main()