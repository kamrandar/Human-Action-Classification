import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SMOTE  # Import SMOTE

from load_data import load_all_csv
from preprocess import add_magnitude, scale_per_participant
from segmentation import sliding_window_per_participant
from features import extract_stat_features

def run_training(data_folder, out_models_folder, out_reports_folder):
    print("Loading data...")
    df = load_all_csv(data_folder)
    print("Raw shape:", df.shape)
    df = add_magnitude(df)
    df = scale_per_participant(df)
    print("After magnitude and scaling:", df.shape)

    print("Dropping rows with label 0...")
    df = df[df['label'] != 0]  # Drop rows where the label is 0
    print("After dropping label 0, shape:", df.shape)

    print("Segmenting into windows...")
    X_w, y, parts = sliding_window_per_participant(df, window_size=104, step=52)
    print("Windows shape:", X_w.shape, y.shape)

    print("Extracting features...")
    X_feat = extract_stat_features(X_w, fs=52.0)
    print("Feature matrix shape:", X_feat.shape)

    # Apply SMOTE for oversampling
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_feat, y)
    print("After SMOTE, feature matrix shape:", X_resampled.shape, "Labels shape:", y_resampled.shape)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    print("Training with hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    clf = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='f1_weighted'
    )
    clf.fit(X_train, y_train)
    print("Best parameters:", clf.best_params_)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    cm_path = Path(out_reports_folder) / 'confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # Save model
    model_path = Path(out_models_folder) / 'Final_model.joblib'
    joblib.dump({'model': clf}, model_path)

    # Save a small feature importance plot
    try:
        fi = clf.feature_importances_
        plt.figure(figsize=(10, 4))
        plt.plot(fi, marker='o')
        plt.title('Feature importances (by index)')
        plt.xlabel('Feature index')
        plt.ylabel('Importance')
        plt.grid(True)
        fi_path = Path(out_reports_folder) / 'feature_importances.png'
        plt.savefig(fi_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print('Could not save feature importances:', e)

    # Also save a CSV summary of class counts
    pd.DataFrame({'label': y_resampled}).value_counts().to_frame('count').to_csv(Path(out_reports_folder) / 'label_counts.csv')

    return model_path, cm_path

if __name__ == '__main__':
    data_folder = str(Path(__file__).resolve().parents[1] / 'data')
    models_folder = str(Path(__file__).resolve().parents[1] / 'models')
    reports_folder = str(Path(__file__).resolve().parents[1] / 'reports')
    Path(models_folder).mkdir(exist_ok=True)
    Path(reports_folder).mkdir(exist_ok=True)
    model_path, cm_path = run_training(data_folder, models_folder, reports_folder)
    print('Model saved to', model_path)
    print('Confusion matrix saved to', cm_path)