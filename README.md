# Amazon Reviews Sentiment Analysis

Classify Amazon product reviews as **Positive** or **Negative** using a TF-IDF + ML pipeline, exposed through a Flask API and a simple web UI.

## Features

- Text preprocessing: HTML cleanup, lowercasing, punctuation removal, stopword filtering, SymSpell spelling correction, and spaCy lemmatization
- Trained model served via Flask (`/predict`, `/batch_predict`, `/health`)
- Web frontend (`frontend.html`) for interactive single-review analysis

## Project structure

| File | Description |
|------|-------------|
| `main.ipynb` | Model training and experimentation |
| `app.py` | Flask API for sentiment prediction |
| `frontend.html` | Browser UI that calls the API |
| `model.pkl` / `tfidf_vectorizer.pkl` | Saved model and vectorizer (train first if missing) |
| `frequency_dictionary_en_82_765.txt` | SymSpell frequency dictionary |

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install flask flask-cors pandas numpy nltk spacy symspellpy scikit-learn
python -m spacy download en_core_web_sm
```

Train the model in `main.ipynb` and save `model.pkl` and `tfidf_vectorizer.pkl` in the project root if they are not already present.

## Run

```bash
python app.py
```

API: `http://localhost:5000`

Open `frontend.html` in a browser (or serve it locally) to use the UI.

### Example API call

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"review\": \"This product exceeded my expectations!\"}"
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check and model load status |
| POST | `/predict` | Predict sentiment for one review (`{"review": "..."}`) |
| POST | `/batch_predict` | Predict for multiple reviews (`{"reviews": ["...", "..."]}`) |

## Tech stack

Python, Flask, scikit-learn (TF-IDF), NLTK, spaCy, SymSpell, HTML/CSS/JS frontend
