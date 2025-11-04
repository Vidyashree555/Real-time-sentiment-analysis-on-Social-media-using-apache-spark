import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analysis Demo", page_icon="💬")

# Resolve model paths relative to the repository root (this file's parent/..)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "sklearn"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VEC_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS = MODEL_DIR / "metrics.json"
CM_PNG = MODEL_DIR / "confusion_matrix.png"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    metrics = {}
    if METRICS.exists():
        with open(METRICS, "r") as f:
            metrics = json.load(f)
    return model, vectorizer, metrics

model, vectorizer, metrics = load_artifacts()

st.title("💬 Real-Time Sentiment Analysis")
st.caption("Dashboard uses scikit-learn model. A Spark MLlib model is also trained for your report.")

# --- Tab layout
tab1, tab2, tab3 = st.tabs(["🔤 Single Text", "📦 Batch CSV", "📈 Metrics"])

# --- Tab 1: Single text prediction
with tab1:
    text = st.text_area("Enter text:", height=120, placeholder="Type a tweet or sentence...")
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None

            label = "Positive 😀" if pred == 1 else "Negative 😞"
            st.success(f"Sentiment: {label}")
            if proba is not None:
                st.write(f"Confidence → Negative: {proba[0]:.2f} | Positive: {proba[1]:.2f}")

# --- Tab 2: Batch CSV
with tab2:
    st.write("Upload a CSV with a column named **text** or **clean_text**.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        text_col = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)
        if text_col is None:
            st.error("CSV must have a 'text' or 'clean_text' column.")
        else:
            X = df[text_col].astype(str).fillna("")
            X_vec = vectorizer.transform(X)
            df["pred"] = model.predict(X_vec)
            df["sentiment"] = df["pred"].map({0:"Negative", 1:"Positive"})
            st.subheader("Preview")
            st.dataframe(df.head(30))

            # Pie chart
            counts = df["sentiment"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

            # Download
            st.download_button(
                "Download Results CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

# --- Tab 3: Metrics
with tab3:
    if metrics:
        st.subheader("Accuracy")
        st.write(f"{metrics.get('accuracy', 0):.4f}")
        if "classification_report" in metrics:
            st.subheader("Classification Report")
            cr = pd.DataFrame(metrics["classification_report"]).transpose()
            st.dataframe(cr)
    else:
        st.info("No metrics.json found. Train with scripts/train_sklearn.py first.")

    if CM_PNG.exists():
        st.subheader("Confusion Matrix")
        st.image(str(CM_PNG))
