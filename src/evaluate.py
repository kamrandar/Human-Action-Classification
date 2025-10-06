import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_all_csv
from preprocess import add_magnitude, scale_per_participant
from segmentation import sliding_window_per_participant
from features import extract_stat_features

def evaluate(model_path, data_folder, reports_folder):
    df = load_all_csv(data_folder)
    df = add_magnitude(df)
    df = scale_per_participant(df)
    X_w, y, parts = sliding_window_per_participant(df, window_size=104, step=52)
    X_feat = extract_stat_features(X_w)
    obj = joblib.load(model_path)
    clf = obj['model']
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion matrix')
    plt.savefig(Path(reports_folder)/'confusion_matrix_eval.png', bbox_inches='tight')
    plt.close()
    return True

if __name__ == '__main__':
    data_folder = str(Path(__file__).resolve().parents[1] / 'data')
    reports_folder = str(Path(__file__).resolve().parents[1] / 'reports')
    model_path = str(Path(__file__).resolve().parents[1] / 'models' / 'rf_baseline.joblib')
    evaluate(model_path, data_folder, reports_folder)
