"""Trener endelig modell og lagrer models/model.joblib + models/config.json."""
import json, sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling_utils import SEED, make_pipeline, augment, oof_proba, choose_thresholds

BEST = {"n_estimators": 50, "max_depth": None}


def main():
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    X, y = train.drop(columns="poisonous"), train["poisonous"]
    cols = list(X.columns)

    def mk():
        return make_pipeline(
            RandomForestClassifier(random_state=SEED, n_jobs=-1, **BEST), cols)

    p, yy = oof_proba(mk, X, y, val_mask_ps=(0, 0.2, 0.4), augment_train=True)
    t_low, t_high = choose_thresholds(yy, p)

    model = mk().fit(*augment(X, y))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/model.joblib")
    json.dump({"T_LOW": t_low, "T_HIGH": t_high},
              open(ROOT / "models/config.json", "w"), indent=2)
    print("Lagret modell. T_LOW =", t_low, "T_HIGH =", t_high)


if __name__ == "__main__":
    main()
