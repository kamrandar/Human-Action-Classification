import argparse
import joblib
import numpy as np
import pandas as pd
from features import extract_stat_features

LABEL_MAP = {
    1: 'Working at Computer',
    2: 'Standing Up/Walking/Going Stairs',
    3: 'Standing',
    4: 'Walking',
    5: 'Going Up/Down Stairs',
    6: 'Walking and Talking',
    7: 'Talking while Standing'
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_csv', type=str, help='CSV file with columns x,y,z (no header). Provide ~104 rows for a 2s window.')
    p.add_argument('--model', type=str, default='models/rf_baseline.joblib')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    obj = joblib.load(args.model)
    model = obj['model']
    if args.input_csv:
        # Read CSV with header, keep only x, y, z columns
        df = pd.read_csv(args.input_csv)
        df = df[['x', 'y', 'z']]
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
        df['magnitude'] = np.sqrt(df.x**2 + df.y**2 + df.z**2)
        X = df[['x','y','z','magnitude']].values
        feats = extract_stat_features(np.expand_dims(X, axis=0))
        pred = int(model.predict(feats)[0])
        print('Predicted label:', pred, LABEL_MAP.get(pred, 'Unknown'))
    else:
        print('Provide --input_csv pointing to a window of samples (for example 104 rows for a 2s window).')
