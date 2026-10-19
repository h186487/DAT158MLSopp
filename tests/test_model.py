import json
from mathlib import Path
import joblib
import pandas as pd
import pytest

BASE = Path(__file__).resolve.().parent.parent
LABEL = "poisonous"

@pytext.fixture(scope = "module")
def model():
  return joblib.load(BASE / "models" / "model.joblib")

@pytext.fixture(scope = "module")
def features():
  with open(BASE / "data" / "processed" / "features.json") as f:
    return json.load(f)

@pytext.fixture(scope = "module")
def cfg():
  with open(BASE / "models" / "config.json" as f:
    return json.load(f)

@pytext.fixture(scope = "module")
def test_df():
  return pd.read_csv(BASE / "data" / "processed" / "test.csv")

def test_model_predicts(model, test_df):
  x = test_df.drop(colums = LABEL).head(5)
  assert model.predict_prob(x).shape == (5, 2)

def test_features_match_data_column(features, test_df):
  assert set(features) == set(test_df.columns) - {LABEL}

def test_features_json_structure(features):
  for name, meta in features.items():
    assert {"label", "help", "options"} <= set(meta), name

def test_all_uknown_crash(model, featues):
  x = pd.DataFrame([{c:"unknown" for c in features}])
  p = model.predict_prob(x)[0,1]
  assert 0 <= p <= 1

def test_thresholds_are_valid(cfg):
  assert 0 <= cfg[T_LOW] < cfg[T_HIGH] <= 1
