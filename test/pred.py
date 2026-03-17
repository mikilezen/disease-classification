# import requests
# res = requests.post('http://localhost:8000/predict', json={'text': 'This is a sample text for prediction.'})
# print(res.json())
import requests

def test_prediction():
    url = "http://127.0.0.1:8000/predict"

    samples = [
        "I love this product, it's amazing!",
        "This is the worst experience ever",
        "It is okay, nothing special"
    ]

    for text in samples:
        try:
            response = requests.post(url, json={"text": text})

            if response.status_code == 200:
                result = response.json()
                print(f"\nInput: {text}")
                print(f"Prediction: {result['prediction']}")
            else:
                print(f"\nError {response.status_code}: {response.text}")

        except Exception as e:
            print(f"\nRequest failed: {e}")

if __name__ == "__main__":
    test_prediction()