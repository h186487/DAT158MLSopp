from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix


def report(name, y_true, y_pred):
    """Returnerer en dict med metrikker. Positiv klasse (1) = giftig."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Modell": name,
        "Recall (giftig)": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "Precision (giftig)": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "False negatives": fn,
        "False positives": fp,
    }
