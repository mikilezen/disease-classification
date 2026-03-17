import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

# Ensure NLTK looks in our app-local directory first
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Download required NLTK resources only if they are missing
_REQUIRED_NLTK = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}

for pkg, resource_path in _REQUIRED_NLTK.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

app = FastAPI()

model = joblib.load("./src/logistic_regression_model.joblib")
vectorizer = joblib.load("./src/tfidf_vectorizer.joblib")
encoder = joblib.load("./src/label_encoder.joblib")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

class PredictionRequest(BaseModel):
    text: str


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    processed = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words
    ]
    return " ".join(processed)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictionRequest):
    processed = preprocess_text(req.text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)
    label = encoder.inverse_transform(pred)
    return {"prediction": label[0]}

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)