import string
from pathlib import Path

import joblib
import nltk
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": request.url.path},
    )

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
NLTK_DIR = BASE_DIR / "nltk_data"
NLTK_DIR.mkdir(parents=True, exist_ok=True)

if str(NLTK_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DIR))

_REQUIRED_NLTK = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}

for pkg, resource_path in _REQUIRED_NLTK.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(pkg, download_dir=str(NLTK_DIR))

model = joblib.load(SRC_DIR / "logistic_regression_model.joblib")
vectorizer = joblib.load(SRC_DIR / "tfidf_vectorizer.joblib")
encoder = joblib.load(SRC_DIR / "label_encoder.joblib")


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

class PredictionRequest(BaseModel):
    text: str

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    processed = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
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
