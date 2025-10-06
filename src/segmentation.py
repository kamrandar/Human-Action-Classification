import numpy as np
from typing import Tuple, List

def sliding_window_per_participant(df, cols=['x','y','z','magnitude'], window_size=104, step=52):
    X = []
    y = []
    parts = []
    participants = df['participant'].unique()
    for p in participants:
        sub = df[df['participant']==p].reset_index(drop=True)
        vals = sub[cols].values
        labels = sub['label'].values.astype(int)
        n = len(sub)
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            seg = vals[start:end]
            lab = np.bincount(labels[start:end]).argmax()
            X.append(seg)
            y.append(lab)
            parts.append(p)
    return np.array(X), np.array(y), np.array(parts)
