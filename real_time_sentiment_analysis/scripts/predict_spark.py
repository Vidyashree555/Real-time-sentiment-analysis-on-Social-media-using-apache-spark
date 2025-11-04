from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

MODEL_PATH = "models/spark/spark_sentiment_model"

def predict(text: str):
    spark = SparkSession.builder.appName("SparkPredict").getOrCreate()
    model = PipelineModel.load(MODEL_PATH)
    df = spark.createDataFrame([[0, text]], ["target", "clean_text"])
    out = model.transform(df).select("clean_text", "prediction").collect()[0]
    spark.stop()
    label = "Positive" if out["prediction"] == 1.0 else "Negative"
    return label

if __name__ == "__main__":
    sample = "I absolutely love this phone!"
    print(sample, "->", predict(sample))
