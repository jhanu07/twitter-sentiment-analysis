from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_tweets.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH).dropna(subset=["text", "sentiment"])
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["sentiment"], test_size=0.25, random_state=42, stratify=data["sentiment"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
matrix = confusion_matrix(y_test, predictions, labels=["negative", "neutral", "positive"])

joblib.dump(model, MODEL_DIR / "sentiment_pipeline.joblib")
results = {
    "test_accuracy": round(float(accuracy), 4),
    "test_rows": int(len(y_test)),
    "class_labels": ["negative", "neutral", "positive"],
    "confusion_matrix": matrix.tolist(),
    "classification_report": report,
    "note": "Results are from a small educational sample dataset; do not treat them as benchmark performance."
}
(OUTPUT_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

print(f"Test accuracy on sample data: {accuracy:.2%}")
print("Classification report:")
print(classification_report(y_test, predictions, zero_division=0))
print(f"Saved model to: {MODEL_DIR / 'sentiment_pipeline.joblib'}")
print(f"Saved metrics to: {OUTPUT_DIR / 'evaluation_results.json'}")
