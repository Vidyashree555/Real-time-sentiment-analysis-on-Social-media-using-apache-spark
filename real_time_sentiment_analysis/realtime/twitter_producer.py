import json
from kafka import KafkaProducer
import tweepy
from pathlib import Path

# Load Twitter API credentials
with open(Path(__file__).parent / "twitter_credentials.json") as f:
    creds = json.load(f)

# Setup Tweepy client (bearer token recommended for v2 API)
client = tweepy.Client(bearer_token=creds["BEARER_TOKEN"])

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

QUERY = "technology OR AI OR sports -is:retweet lang:en"

print("Fetching live tweets... Press Ctrl+C to stop.")

try:
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=QUERY, tweet_fields=["text"], max_results=10).flatten(limit=1000):
        msg = {"text": tweet.text}
        producer.send("tweets", value=msg)
        print("Sent:", msg)
except KeyboardInterrupt:
    print("Stopped streaming.")
finally:
    producer.close()
