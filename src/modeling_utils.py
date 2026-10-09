import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

SEED = 42

def make_pipeline(model, columns):
    """Pipeline: one-hot-koding av de oppgitte kolonnene + modell."""
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), list(columns))]
    )
    return Pipeline([("pre", pre), ("model", model)])


def mask_unknown(X, p, seed=SEED):
    """Erstatter hver verdi med 'unknown' med sannsynlighet p ('Vet ikke')."""
    rng = np.random.default_rng(seed)
    return X.mask(rng.random(X.shape) < p, "unknown")


def corrupt(X, p, seed=SEED):
    """Erstatter hver verdi med en TILFELDIG ANNEN verdi fra samme kolonne
    med sannsynlighet p (brukeren svarer feil)."""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    for col in X.columns:
        values = X[col].unique()
        hit = rng.random(len(X)) < p
        Xn.loc[hit, col] = rng.choice(values, hit.sum())
    return Xn


def augment(X, y, ps=(0.2, 0.4), seed=SEED):
    """Originale data + maskerte kopier. KUN for treningsdata."""
    Xs, ys = [X], [y]
    for i, p in enumerate(ps):
        Xs.append(mask_unknown(X, p, seed + i + 1))
        ys.append(y)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)


def cv_evaluate(make_model, X, y, n_splits=5, n_repeats=1,
                augment_train=False, seed=SEED):
    """Stratifisert (repetert) CV. Returnerer én rad per fold.
    Augmentering skjer INNE i hver fold, kun på treningsdelen."""
    rows = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, va in skf.split(X, y):
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            Xva, yva = X.iloc[va], y.iloc[va]
            if augment_train:
                Xtr, ytr = augment(Xtr, ytr)
            pred = make_model().fit(Xtr, ytr).predict(Xva)
            tn, fp, fn, tp = confusion_matrix(yva, pred, labels=[0, 1]).ravel()
            rows.append({
                "recall": recall_score(yva, pred, zero_division=0),
                "precision": precision_score(yva, pred, zero_division=0),
                "accuracy": accuracy_score(yva, pred),
                "fn": fn, "fp": fp,
            })
    return pd.DataFrame(rows)


def pm(df, col, digits=4):
    """Formater 'snitt ± std' for tabeller."""
    return f"{df[col].mean():.{digits}f} ± {df[col].std():.{digits}f}"


def oof_proba(make_model, X, y, val_mask_ps=(0,), augment_train=False,
              n_splits=5, seed=SEED):
    """Out-of-fold sannsynligheter på treningsdata. Hver rad får en
    prediksjon fra en modell som ALDRI har sett raden.
    val_mask_ps: valideringsradene maskeres også (0 = uendret)."""
    ps, ys = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        if augment_train:
            Xtr, ytr = augment(Xtr, ytr)
        m = make_model().fit(Xtr, ytr)
        for j, q in enumerate(val_mask_ps):
            Xv = X.iloc[va] if q == 0 else mask_unknown(X.iloc[va], q, seed + 100 * k + j)
            ps.append(m.predict_proba(Xv)[:, 1])
            ys.append(y.iloc[va].to_numpy())
    return np.concatenate(ps), np.concatenate(ys)


def choose_thresholds(y, p, min_precision=0.99, cap_low=0.05):
    """T_LOW:  p <= T_LOW  -> 'ingen tegn på gift'
       T_HIGH: p >= T_HIGH -> 'sannsynlig giftig'. Mellom -> 'usikker'.
    Velges på validerings-sannsynligheter (aldri på test)."""
    y = np.asarray(y)
    grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5,
            0.7, 0.9, 0.95, 0.99]
    # T_LOW: høyeste terskel der INGEN giftige havner under den, med halv margin
    ok = [t for t in grid if (p[y == 1] >= t).all()]
    t_low = min((max(ok) if ok else grid[0]) / 2, cap_low)
    # T_HIGH: laveste terskel der precision for 'giftig' er høy nok
    t_high = 0.99
    for t in grid:
        sel = p >= t
        if sel.sum() > 0 and (y[sel] == 1).mean() >= min_precision:
            t_high = t
            break
    return float(t_low), float(t_high)
