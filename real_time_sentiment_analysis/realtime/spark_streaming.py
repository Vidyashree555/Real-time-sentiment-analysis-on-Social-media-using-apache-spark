#!/usr/bin/env python3
"""
Real-time streaming runner for sentiment classification.

Modes:
- local: simulate streaming from CSV using scikit-learn artifacts (preferred) or Spark PipelineModel.
- kafka: consume JSON messages from Kafka and classify using Spark PipelineModel (preferred) or scikit-learn artifacts.

This file keeps Spark and Kafka logic separate so it can run on Windows without Spark.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "cleaned_tweets.csv"
DEFAULT_SKLEARN_DIR = ROOT / "models" / "sklearn"
DEFAULT_SPARK_MODEL = ROOT / "models" / "spark" / "spark_sentiment_model"
KAFKA_BOOTSTRAP_DEFAULT = "localhost:9092"
KAFKA_TOPIC_DEFAULT = "tweets"


def get_abs(rel_path: str | Path) -> str:
    p = Path(rel_path)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[1] / p).resolve()
    return str(p)


def load_sklearn_artifacts(sklearn_dir: str | Path = DEFAULT_SKLEARN_DIR) -> Tuple[Optional[object], Optional[object]]:
    sklearn_dir = Path(sklearn_dir)
    vec_path = sklearn_dir / "tfidf_vectorizer.pkl"
    model_path = sklearn_dir / "sentiment_model.pkl"
    if not (vec_path.exists() and model_path.exists()):
        logging.debug("sklearn artifacts not found at %s", sklearn_dir)
        return None, None
    try:
        vec = joblib.load(str(vec_path))
        clf = joblib.load(str(model_path))
        logging.info("Loaded scikit-learn artifacts from %s", sklearn_dir)
        return vec, clf
    except Exception:
        logging.exception("Failed to load sklearn artifacts from %s", sklearn_dir)
        return None, None


def try_load_spark_model(path: str | Path = DEFAULT_SPARK_MODEL) -> Tuple[Optional[object], Optional[object]]:
    path = Path(path)
    if not path.exists():
        logging.debug("Spark model path does not exist: %s", path)
        return None, None
    try:
        from pyspark.sql import SparkSession  # type: ignore
        from pyspark.ml import PipelineModel  # type: ignore

        spark = SparkSession.builder.appName("RealTimeSentiment").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        model = PipelineModel.load(str(path))
        logging.info("Loaded Spark PipelineModel from %s", path)
        return spark, model
    except Exception as e:
        logging.warning("Could not load Spark model (will fallback to sklearn): %s", e)
        return None, None


def run_local_mode_sklearn(csv_path: str, vectorizer, model, batch_size: int = 50, interval: float = 2.0, loop: bool = False):
    try:
        import pandas as pd
    except Exception:
        logging.error("pandas is required for local mode. Install requirements and retry.")
        sys.exit(1)

    if not os.path.exists(csv_path):
        logging.error("CSV not found at %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    text_col = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)
    if text_col is None:
        logging.error("CSV must contain a 'text' or 'clean_text' column")
        sys.exit(1)

    texts = df[text_col].astype(str).fillna("").tolist()
    if not texts:
        logging.error("No text rows found in CSV")
        sys.exit(1)

    idx = 0
    round_no = 0
    while True:
        batch = texts[idx: idx + batch_size]
        if not batch:
            if loop:
                idx = 0
                round_no += 1
                logging.info("Restarting local simulation (round %d)", round_no)
                continue
            else:
                logging.info("Local simulation complete")
                break

        X = vectorizer.transform(batch)
        preds = model.predict(X)
        probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        for i, txt in enumerate(batch):
            p = int(preds[i])
            label = "Positive" if p == 1 else "Negative"
            if probs is not None:
                conf = probs[i]
                logging.info("%s -> %s (conf: %.2f/%.2f)", txt[:120].replace("\n", " "), label, float(conf[0]), float(conf[1]))
            else:
                logging.info("%s -> %s", txt[:120].replace("\n", " "), label)

        idx += batch_size
        time.sleep(interval)


def run_kafka_mode_sklearn(kafka_bootstrap: str, topic: str, vectorizer, model):
    try:
        from kafka import KafkaConsumer
    except Exception:
        logging.error("kafka-python not installed. Install it with: pip install kafka-python")
        sys.exit(1)

    logging.info("Starting Kafka sklearn consumer on topic '%s' (bootstrap=%s)", topic, kafka_bootstrap)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    try:
        for msg in consumer:
            try:
                payload = msg.value
                text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
                X = vectorizer.transform([text])
                pred = int(model.predict(X)[0])
                label = "Positive" if pred == 1 else "Negative"
                proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
                if proba is not None:
                    logging.info("%s -> %s (conf: %.2f/%.2f)", text[:120].replace("\n", " "), label, float(proba[0]), float(proba[1]))
                else:
                    logging.info("%s -> %s", text[:120].replace("\n", " "), label)
            except Exception:
                logging.exception("Failed to process Kafka message")
    except KeyboardInterrupt:
        logging.info("Kafka consumer stopped by user")
    finally:
        try:
            consumer.close()
        except Exception:
            pass


def run_kafka_mode_spark(spark, model, kafka_bootstrap: str = KAFKA_BOOTSTRAP_DEFAULT, topic: str = KAFKA_TOPIC_DEFAULT):
    from pyspark.sql.functions import col, from_json
    from pyspark.sql.types import StructType, StringType

    schema = StructType().add("text", StringType())

    df_kafka = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap) \
        .option("subscribe", topic) \
        .option("startingOffsets", "latest") \
        .load()

    df_parsed = df_kafka.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # rename text to clean_text and add dummy "target" if pipeline expects it
    df_input = df_parsed.withColumn("target", col("text").substr(0, 0).cast("double")).withColumnRenamed("text", "clean_text")
    preds = model.transform(df_input).select("clean_text", "prediction")

    query = preds.writeStream.outputMode("append").format("console").start()
    logging.info("Started Spark structured streaming job on topic '%s'", topic)
    query.awaitTermination()


def run_local_mode_spark(spark, model, csv_path: str, batch_size: int = 100, interval: float = 2.0):
    """A simple single-run transform using Spark (not continuous streaming)."""
    from pyspark.sql.functions import col
    if not os.path.exists(csv_path):
        logging.error("CSV data not found at %s", csv_path)
        sys.exit(1)

    sdf = spark.read.csv(csv_path, header=True)
    text_col = "clean_text" if "clean_text" in sdf.columns else ("text" if "text" in sdf.columns else None)
    if text_col is None:
        logging.error("CSV must contain 'text' or 'clean_text' column")
        sys.exit(1)

    sdf = sdf.withColumnRenamed(text_col, "clean_text")
    sdf = sdf.withColumn("target", col("clean_text").substr(0, 0).cast("double"))
    preds = model.transform(sdf).select("clean_text", "prediction").toPandas()
    for _, row in preds.iterrows():
        logging.info("%s -> %s", row["clean_text"][:120].replace("\n", " "), int(row["prediction"]))


def main():
    parser = argparse.ArgumentParser(description="Streaming runner for sentiment classification")
    parser.add_argument("--mode", choices=["local", "kafka"], default="local")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true", help="Loop local simulation")
    parser.add_argument("--kafka-bootstrap", default=KAFKA_BOOTSTRAP_DEFAULT)
    parser.add_argument("--topic", default=KAFKA_TOPIC_DEFAULT)
    parser.add_argument("--spark-model", default=str(DEFAULT_SPARK_MODEL))
    args = parser.parse_args()

    # Try Spark first (preferred for Kafka streaming); if not available, fall back to sklearn artifacts
    spark_session, spark_model = try_load_spark_model(args.spark_model)
    skl_vectorizer, skl_model = (None, None)
    if spark_model is None:
        skl_vectorizer, skl_model = load_sklearn_artifacts()

    if args.mode == "local":
        # local simulation: prefer sklearn (faster and Windows-friendly)
        if skl_model is None:
            if spark_model is not None:
                logging.info("Using Spark PipelineModel for local simulation (single-run transform).")
                run_local_mode_spark(spark_session, spark_model, args.csv, batch_size=args.batch_size, interval=args.interval)
            else:
                logging.error("No sklearn model available for local simulation. Please train sklearn model first.")
                sys.exit(1)
        else:
            run_local_mode_sklearn(args.csv, skl_vectorizer, skl_model, batch_size=args.batch_size, interval=args.interval, loop=args.loop)
    else:
        # kafka mode: prefer spark structured streaming; otherwise use sklearn consumer
        if spark_session is not None and spark_model is not None:
            run_kafka_mode_spark(spark_session, spark_model, kafka_bootstrap=args.kafka_bootstrap, topic=args.topic)
        else:
            if skl_model is None:
                logging.error("No model available for Kafka mode. Train sklearn or Spark model first.")
                sys.exit(1)
            run_kafka_mode_sklearn(args.kafka_bootstrap, args.topic, skl_vectorizer, skl_model)


if __name__ == "__main__":
    main()
