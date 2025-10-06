import requests

with open('src/sample_window.csv', 'rb') as f:
    files = {'file': f}
    r = requests.post('http://localhost:5000/predict', files=files)
    print(r.text)