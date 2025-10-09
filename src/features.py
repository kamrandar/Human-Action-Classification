import numpy as np

def zero_crossings(arr):
    return ((arr[:-1] * arr[1:]) < 0).sum()

def iqr(arr):
    return np.percentile(arr,75) - np.percentile(arr,25)

def dominant_freq(data, fs=52.0):
    # return frequency (Hz) of largest FFT magnitude excluding DC
    n = len(data)
    if n <= 1:
        return 0.0
    yf = np.abs(np.fft.rfft(data))
    xf = np.fft.rfftfreq(n, 1.0/fs)
    if len(yf) <= 1:
        return 0.0
    # ignore the zero-frequency term at index 0
    idx = 1 + np.argmax(yf[1:]) if len(yf)>1 else 0
    return float(xf[idx])

def extract_stat_features(X_windowed, fs=52.0):
    # X_windowed shape: (n_windows, window_len, n_channels)
    feats = []
    for w in X_windowed:
        f = []
        # for each channel compute features
        for ch in range(w.shape[1]):
            data = w[:,ch]
            f.extend([
                float(np.mean(data)), float(np.std(data)), float(np.min(data)), float(np.max(data)),
                float(np.median(data)), float(iqr(data)), float(zero_crossings(data)),
                float(np.sum(data**2)/len(data) if len(data)>0 else 0.0),
                float(dominant_freq(data, fs=fs)),
                float(np.mean(np.abs(np.diff(data)))),  # Mean absolute difference
                float(np.sum(np.abs(data) > np.std(data))),  # Number of points above 1 std
                float(np.percentile(data, 90) - np.percentile(data, 10)),  # 90-10 percentile range
                float(np.correlate(data, data, mode='valid')[0]) # Autocorrelation
            ])
        feats.append(f)
    return np.array(feats)
