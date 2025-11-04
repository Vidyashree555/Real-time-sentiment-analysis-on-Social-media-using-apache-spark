import time
import json
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "sklearn"
VEC_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
CSV_PATH = ROOT / "data" / "cleaned_tweets.csv"

BATCH_SIZE = 5
INTERVAL = 1.0  # seconds between batches


def load_artifacts():
    if not VEC_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing sklearn artifacts in {MODEL_DIR}")
    vectorizer = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def run_once(batch_size=BATCH_SIZE):
    vectorizer, model = load_artifacts()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV data not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    text_col = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)
    if text_col is None:
        raise ValueError("CSV must contain 'text' or 'clean_text' column")
    X = df[text_col].astype(str).fillna("")
    # just take first batch
    sample = X.iloc[:batch_size]
    X_vec = vectorizer.transform(sample)
    preds = model.predict(X_vec)
    proba = model.predict_proba(X_vec) if hasattr(model, "predict_proba") else None
    for i, text in enumerate(sample):
        p = preds[i]
        label = "Positive" if p == 1 else "Negative"
        if proba is not None:
            conf = proba[i]
            print(f"{i+1}. {text[:80].replace(chr(10),' ')} -> {label} (conf: {conf[0]:.2f}/{conf[1]:.2f})")
        else:
            print(f"{i+1}. {text[:80].replace(chr(10),' ')} -> {label}")


if __name__ == '__main__':
    print("Starting local sklearn streaming simulation (one batch)...")
    run_once()
