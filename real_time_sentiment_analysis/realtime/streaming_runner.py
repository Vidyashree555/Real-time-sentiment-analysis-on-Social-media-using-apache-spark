"""Single-file clean runner for local/kafka using sklearn artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CSV = os.path.join(ROOT, "data", "cleaned_tweets.csv")
SKLEARN_DIR = os.path.join(ROOT, "models", "sklearn")
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "tweets"


def load_models(sklearn_dir=SKLEARN_DIR):
    vec = os.path.join(sklearn_dir, "tfidf_vectorizer.pkl")
    mod = os.path.join(sklearn_dir, "sentiment_model.pkl")
    if not (os.path.exists(vec) and os.path.exists(mod)):
        logging.error("Sklearn artifacts missing in %s", sklearn_dir)
        return None, None
    try:
        v = joblib.load(vec)
        m = joblib.load(mod)
        logging.info("Loaded sklearn artifacts from %s", sklearn_dir)
        return v, m
    except Exception:
        logging.exception("Failed to load sklearn artifacts")
        return None, None


def run_local(csv_path, vectorizer, model, batch_size=5, interval=1.0, loop=False):
    try:
        import pandas as pd
    except Exception:
        logging.error("pandas is required for local mode")
        sys.exit(1)

    if not os.path.exists(csv_path):
        logging.error("CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    text_col = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)
    if text_col is None:
        logging.error("CSV needs 'text' or 'clean_text' column")
        sys.exit(1)

    texts = df[text_col].astype(str).fillna("").tolist()
    if not texts:
        logging.error("No rows in CSV")
        sys.exit(1)

    i = 0
    while True:
        batch = texts[i:i + batch_size]
        if not batch:
            if loop:
                i = 0
                continue
            break

        X = vectorizer.transform(batch)
        preds = model.predict(X)
        probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        for j, txt in enumerate(batch):
            p = int(preds[j])
            label = "Positive" if p == 1 else "Negative"
            if probs is not None:
                conf = probs[j]
                logging.info("%s -> %s (conf: %.2f/%.2f)", txt[:120].replace("\n", " "), label, conf[0], conf[1])
            else:
                logging.info("%s -> %s", txt[:120].replace("\n", " "), label)

        i += batch_size
        time.sleep(interval)


def run_kafka(kafka_bootstrap, topic, vectorizer, model):
    try:
        from kafka import KafkaConsumer
    except Exception:
        logging.error("kafka-python not installed. pip install kafka-python")
        sys.exit(1)

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    try:
        for msg in consumer:
            payload = msg.value
            text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
            X = vectorizer.transform([text])
            pred = int(model.predict(X)[0])
            label = "Positive" if pred == 1 else "Negative"
            proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
            if proba is not None:
                logging.info("%s -> %s (conf: %.2f/%.2f)", text[:120].replace("\n", " "), label, proba[0], proba[1])
            else:
                logging.info("%s -> %s", text[:120].replace("\n", " "), label)
    except KeyboardInterrupt:
        logging.info("Kafka consumer stopped")
    finally:
        try:
            consumer.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "kafka"], default="local")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--kafka-bootstrap", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    args = parser.parse_args()

    vec, mod = load_models()
    if vec is None or mod is None:
        logging.error("Missing sklearn artifacts in models/sklearn. Run train_sklearn.py first.")
        sys.exit(1)

    if args.mode == "local":
        run_local(args.csv, vec, mod, batch_size=args.batch_size, interval=args.interval, loop=args.loop)
    else:
        run_kafka(args.kafka_bootstrap, args.topic, vec, mod)


if __name__ == "__main__":
    main()

