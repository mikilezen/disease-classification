import requests

def test_prediction():
    url = "https://disease-classification-1-q58o.onrender.com/predict"
    samples = [
        "I love this product, it's amazing!",
        "This is the worst experience ever",
        "It is okay, nothing special",
    ]

    for text in samples:
        response = requests.post(url, json={"text": text})
        print("\nInput:", text)
        print("Status:", response.status_code)
        print("Body:", response.text)

if __name__ == "__main__":
    test_prediction()
