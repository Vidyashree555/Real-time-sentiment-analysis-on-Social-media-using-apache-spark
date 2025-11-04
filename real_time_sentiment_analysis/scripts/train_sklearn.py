import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

DATA_PATH = Path("data/cleaned_tweets.csv")
MODEL_DIR = Path("models/sklearn")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Loading cleaned data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"].astype(str).values
    y = df["target"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model + vectorizer
    joblib.dump(model, MODEL_DIR / "sentiment_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    print(f"Saved model + vectorizer → {MODEL_DIR}")

    # Save metrics
    metrics = {
        "accuracy": acc,
        "classification_report": report
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2), yticks=np.arange(2),
        xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
        ylabel="True label", xlabel="Predicted label", title="Confusion Matrix"
    )
    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png")
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    main()
