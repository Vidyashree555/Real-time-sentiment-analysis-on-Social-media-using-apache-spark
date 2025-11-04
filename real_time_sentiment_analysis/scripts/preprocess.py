import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path

# Download stopwords the first time
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))

RAW_PATH = Path("data/tweets.csv")
OUT_PATH = Path("data/cleaned_tweets.csv")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # URLs
    text = re.sub(r"[@#]\w+", " ", text)                 # @user, #hashtag
    text = re.sub(r"[^a-zA-Z\s]", " ", text)             # non-letters
    text = text.lower()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", header=None)
    # Try Sentiment140 layout:
    # 0=target (0 neg, 4 pos), 5=text
    if df.shape[1] >= 6:
        df = df[[0, 5]]
        df.columns = ["target", "text"]
        df["target"] = df["target"].apply(lambda x: 1 if int(x) == 4 else 0)
        return df

    # Otherwise assume generic CSV with header:
    # try reading again with header
    df = pd.read_csv(path)
    if {"target", "text"}.issubset(df.columns):
        return df[["target", "text"]]
    raise ValueError("CSV format not recognized. Provide Sentiment140 or a CSV with 'target' and 'text' columns.")

def main():
    print(f"Reading: {RAW_PATH}")
    df = load_raw(RAW_PATH)
    print(f"Rows: {len(df)}")

    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    df = df[["target", "clean_text"]].dropna()
    df = df[df["clean_text"].str.strip().ne("")]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data → {OUT_PATH.resolve()}  (rows: {len(df)})")

if __name__ == "__main__":
    main()
