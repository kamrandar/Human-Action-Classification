from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from features import extract_stat_features

app = Flask(__name__)
obj = joblib.load('models/rf_baseline.joblib')
model = obj['model']

LABEL_MAP = {
    1: 'Working at Computer',
    2: 'Standing Up/Walking/Going Stairs',
    3: 'Standing',
    4: 'Walking',
    5: 'Going Up/Down Stairs',
    6: 'Walking and Talking',
    7: 'Talking while Standing'
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    f = request.files['file']
    df = pd.read_csv(f, header=None, names=['x','y','z'])
    df['magnitude'] = np.sqrt(df.x**2 + df.y**2 + df.z**2)
    X = df[['x','y','z','magnitude']].values
    feats = extract_stat_features(np.expand_dims(X, axis=0))
    pred = int(model.predict(feats)[0])
    return jsonify({'label': pred, 'label_name': LABEL_MAP.get(pred)}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
