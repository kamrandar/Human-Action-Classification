import glob
import pandas as pd
from pathlib import Path

def load_all_csv(folder: str):
    files = sorted(glob.glob(f"{folder}/*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None, names=["seq","x","y","z","label"])
        df["participant"] = Path(f).stem
        dfs.append(df)
    if len(dfs)==0:
        return pd.DataFrame(columns=["seq","x","y","z","label","participant"])
    data = pd.concat(dfs, ignore_index=True)
    return data

if __name__ == '__main__':
    df = load_all_csv(str(Path(__file__).resolve().parents[1] / 'data'))
    print("Loaded shape:", df.shape)
    print(df.head())
