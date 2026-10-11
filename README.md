# Twitter Sentiment Analyzer

A beginner-friendly NLP mini project that predicts whether a short post sounds **positive**, **negative**, or **neutral**. It uses Python, TF-IDF, Logistic Regression, and Streamlit.

## Features

- Predict sentiment for a single post
- Show the model's estimated probability for each sentiment
- Explore the included sample dataset
- Upload a CSV with a `text` column for batch predictions
- Download predictions as a CSV
- Train and evaluate the model with a separate script

## Tech stack

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Project structure

```text
Twitter_Sentiment_Analyzer/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── sample_tweets.csv
├── models/                 # Generated model files are not included
├── outputs/                # Evaluation results are generated here
├── .gitignore
├── LICENSE
└── README.md
```

## Run locally

### 1. Install Python

Use Python 3.10 or newer.

### 2. Open a terminal in this folder

Create and activate a virtual environment (recommended):

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the web app

```bash
streamlit run app.py
```

Your browser should open the app. If it does not, use the local address shown in the terminal.

### 5. Optional: train and evaluate the model

```bash
python train_model.py
```

This saves the trained model in `models/` and evaluation metrics in `outputs/`.

## Dataset note

`data/sample_tweets.csv` contains 120 short, manually written example posts for learning and demonstration. They are not collected from X/Twitter and do not represent public opinion. The small dataset is intended to make the project easy to run without API keys or large downloads.

## Limitations

- The model learns from a small sample dataset and may not generalize to real posts.
- Sarcasm, slang, emojis, mixed feelings, and context can be difficult to classify.
- The confidence value is an estimated model probability, not a guarantee of correctness.
- No live Twitter/X API connection is included.

## Ideas for your own improvements

- Add more properly licensed labeled data.
- Compare Logistic Regression with Naive Bayes.
- Add a confusion-matrix visualization.
- Improve text cleaning for hashtags, emojis, and URLs.
- Deploy the app after reviewing hosting requirements.


## License and attribution

This project is an original educational starter project created for learning. It is released under the MIT License; see `LICENSE`. If you add external datasets, code, images, or other assets, check their licences and include any required attribution.
