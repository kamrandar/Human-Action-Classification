import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_magnitude(df: pd.DataFrame):
    df = df.copy()
    df["magnitude"] = np.sqrt(df.x**2 + df.y**2 + df.z**2)
    return df

def scale_per_participant(df: pd.DataFrame, cols=["x","y","z","magnitude"]):
    df = df.copy()
    scaler = StandardScaler()
    # scale per participant to avoid leakage across subjects
    parts = df['participant'].unique()
    out = []
    for p in parts:
        sub = df[df['participant']==p].copy()
        sub[cols] = scaler.fit_transform(sub[cols])
        out.append(sub)
    return pd.concat(out, ignore_index=True)
