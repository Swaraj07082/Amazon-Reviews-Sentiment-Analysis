from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
import spacy
from symspellpy import SymSpell, Verbosity

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Please install spaCy model: python -m spacy download en_core_web_sm")

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# Load stopwords
sw_list = stopwords.words('english')

# Preprocessing functions (from your notebook)
def remove_tags(raw_text):
    """Remove HTML tags"""
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text

def remove_punctuation(string):
    """Remove punctuation"""
    string = re.sub(r'[^\w\s]', '', string)
    return string

def correct_text(text):
    """Correct spelling using SymSpell"""
    corrected_words = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected_words)

def pos_tagging_lemmatization(text):
    """POS tagging and lemmatization using spaCy"""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def preprocess_text(text):
    """Complete preprocessing pipeline"""
    # Remove HTML tags
    text = remove_tags(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Remove stopwords
    text = [item for item in text.split() if item not in sw_list]
    text = " ".join(text)
    
    # Spelling correction
    text = correct_text(text)
    
    # POS tagging and lemmatization
    text = pos_tagging_lemmatization(text)
    
    return text

# Load trained model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Warning: Model files not found. Please train and save your model first.")
    model = None
    tfidf = None

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Amazon Review Sentiment Analysis API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Predict sentiment of a review",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": tfidf is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment of a review"""
    try:
        # Get review from request
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({
                "error": "No review provided",
                "message": "Please send a JSON with 'review' field"
            }), 400
        
        review_text = data['review']
        
        if not review_text or len(review_text.strip()) == 0:
            return jsonify({
                "error": "Empty review",
                "message": "Review text cannot be empty"
            }), 400
        
        # Check if model is loaded
        if model is None or tfidf is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please train and save the model first"
            }), 500
        
        # Preprocess the review
        processed_review = preprocess_text(review_text)
        
        # Transform using TF-IDF
        review_vector = tfidf.transform([processed_review])
        
        # Make prediction
        prediction = model.predict(review_vector)[0]
        prediction_proba = model.predict_proba(review_vector)[0]
        
        # Get confidence score
        confidence = float(max(prediction_proba)) * 100
        
        # Map prediction to sentiment
        sentiment = "Positive" if prediction == 2 else "Negative"
        
        # Return response
        return jsonify({
            "success": True,
            "original_review": review_text,
            "processed_review": processed_review,
            "sentiment": sentiment,
            "prediction_class": int(prediction),
            "confidence": round(confidence, 2),
            "probabilities": {
                "negative": round(float(prediction_proba[0]) * 100, 2),
                "positive": round(float(prediction_proba[1]) * 100, 2)
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple reviews"""
    try:
        data = request.get_json()
        
        if not data or 'reviews' not in data:
            return jsonify({
                "error": "No reviews provided",
                "message": "Please send a JSON with 'reviews' array"
            }), 400
        
        reviews = data['reviews']
        
        if not isinstance(reviews, list):
            return jsonify({
                "error": "Invalid format",
                "message": "Reviews must be an array"
            }), 400
        
        if model is None or tfidf is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please train and save the model first"
            }), 500
        
        results = []
        
        for review_text in reviews:
            if not review_text or len(review_text.strip()) == 0:
                results.append({
                    "error": "Empty review",
                    "original_review": review_text
                })
                continue
            
            # Preprocess and predict
            processed_review = preprocess_text(review_text)
            review_vector = tfidf.transform([processed_review])
            prediction = model.predict(review_vector)[0]
            prediction_proba = model.predict_proba(review_vector)[0]
            confidence = float(max(prediction_proba)) * 100
            sentiment = "Positive" if prediction == 2 else "Negative"
            
            results.append({
                "original_review": review_text,
                "sentiment": sentiment,
                "confidence": round(confidence, 2)
            })
        
        return jsonify({
            "success": True,
            "total_reviews": len(reviews),
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "error": "Batch prediction failed",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("API will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)