import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from load_data import load_all_csv
from preprocess import add_magnitude, scale_per_participant
from segmentation import sliding_window_per_participant
from features import extract_stat_features

def evaluate(model_path, data_folder, out_reports_folder):
    print("Loading model...")
    try:
        obj = joblib.load(model_path)
        if isinstance(obj, dict) and 'model' in obj:
            clf = obj['model']
            print("Model loaded successfully.")
        else:
            raise ValueError("The model file does not contain a 'model' key or is not a dictionary.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading data...")
    df = load_all_csv(data_folder)
    df = add_magnitude(df)
    df = scale_per_participant(df)

    print("Segmenting into windows...")
    X_w, y, parts = sliding_window_per_participant(df, window_size=104, step=52)
    X_feat = extract_stat_features(X_w, fs=52.0)

    print("Evaluating...")
    y_pred = clf.predict(X_feat)
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    cm_path = Path(out_reports_folder) / 'confusion_matrix_eval.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print('Confusion matrix saved to', cm_path)

if __name__ == '__main__':
    data_folder = str(Path(__file__).resolve().parents[1] / 'data')
    model_path = str(Path(__file__).resolve().parents[1] / 'models' / 'Final_model.joblib')
    reports_folder = str(Path(__file__).resolve().parents[1] / 'reports')
    Path(reports_folder).mkdir(exist_ok=True)
    evaluate(model_path, data_folder, reports_folder)