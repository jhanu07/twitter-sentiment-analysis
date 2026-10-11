from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_tweets.csv"

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 Twitter Sentiment Analyzer")
st.write("Enter a short post to estimate whether its tone is positive, negative, or neutral.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_pipeline():
    data = load_data()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(data["text"], data["sentiment"])
    return model

data = load_data()
model = train_pipeline()

with st.sidebar:
    st.header("About this project")
    st.write("A beginner-friendly NLP mini project built with Python, TF-IDF, Logistic Regression, and Streamlit.")
    st.write(f"Sample rows: **{len(data)}**")
    st.caption("The included examples are educational sample posts, not a live feed from X/Twitter.")

tab1, tab2, tab3 = st.tabs(["Predict sentiment", "Explore sample data", "Try multiple posts"])

with tab1:
    text = st.text_area("Enter a tweet or short post", placeholder="Example: I really enjoyed this movie!", height=120)
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            classes = model.named_steps["classifier"].classes_
            confidence = float(max(probabilities)) * 100
            col1, col2 = st.columns(2)
            col1.metric("Predicted sentiment", prediction.title())
            col2.metric("Model confidence", f"{confidence:.1f}%")
            st.caption("Confidence is the model's estimated probability, not a guarantee that the prediction is correct.")
            prob_df = pd.DataFrame({"Sentiment": [c.title() for c in classes], "Probability": probabilities})
            st.bar_chart(prob_df.set_index("Sentiment"))

with tab2:
    st.subheader("Sample sentiment distribution")
    counts = data["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.bar_chart(counts)
    with col2:
        st.dataframe(data.sample(min(10, len(data)), random_state=42), use_container_width=True, hide_index=True)
    st.download_button(
        "Download sample dataset as CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="sample_tweets.csv",
        mime="text/csv"
    )

with tab3:
    st.write("Upload a CSV file with a column named `text` to analyze multiple posts.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            if "text" not in uploaded_df.columns:
                st.error("Your CSV must contain a column named `text`.")
            else:
                valid = uploaded_df["text"].fillna("").astype(str)
                uploaded_df["predicted_sentiment"] = [
                    model.predict([value])[0] if value.strip() else "empty" for value in valid
                ]
                st.dataframe(uploaded_df, use_container_width=True, hide_index=True)
                st.bar_chart(uploaded_df["predicted_sentiment"].value_counts())
                st.download_button(
                    "Download predictions",
                    data=uploaded_df.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Could not read this CSV file: {exc}")

st.divider()
st.caption("Learning project only. The model is trained on a small sample dataset and may make mistakes on real-world posts.")
