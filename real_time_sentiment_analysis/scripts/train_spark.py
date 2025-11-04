from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

DATA_PATH = "data/cleaned_tweets.csv"
MODEL_PATH = "models/spark/spark_sentiment_model"

def main():
    spark = SparkSession.builder.appName("SparkSentimentTraining").getOrCreate()

    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    df = df.dropna().withColumn("target", col("target").cast("double"))

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=50)

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)

    # Evaluate
    preds = model.transform(test)
    correct = preds.where(col("prediction") == col("target")).count()
    total = preds.count()
    acc = correct / total if total else 0.0
    print(f"Spark model accuracy: {acc:.4f}  (on {total} samples)")

    model.write().overwrite().save(MODEL_PATH)
    print(f"Saved Spark PipelineModel → {MODEL_PATH}")

    spark.stop()

if __name__ == "__main__":
    main()
