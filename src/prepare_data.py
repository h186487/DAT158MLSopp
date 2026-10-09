from pathlib import Path

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

OUT = Path("data/processed")


def main():
    m = fetch_ucirepo(id=73)
    X, y = m.data.features, m.data.targets

    # Rensing
    X = X.replace("?", "unknown").fillna("unknown")
    const_cols = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const_cols)
    y_bin = (y.iloc[:, 0] == "p").astype(int)   # 1 = giftig

    # Split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_bin, test_size=0.2, stratify=y_bin, random_state=42)

    OUT.mkdir(parents=True, exist_ok=True)
    Xtr.assign(poisonous=ytr).to_csv(OUT / "train.csv", index=False)
    Xte.assign(poisonous=yte).to_csv(OUT / "test.csv", index=False)
    print("Droppet:", const_cols, "| train:", Xtr.shape, "| test:", Xte.shape)


if __name__ == "__main__":
    main()
