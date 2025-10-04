import sqlite3, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump
from diff_model.diff_features import extract_diff_features

DB_NAME = "accessibility.db"
MODEL_PATH = "diff_model.joblib"

def load_labeled_diffs():
    """
    Une diff_approvals y diff_rejections con screen_diffs.
    Devuelve X, y para entrenamiento.
    """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT d.id, d.removed, d.added, d.modified,
               CASE
                 WHEN a.id IS NOT NULL THEN 1
                 WHEN r.id IS NOT NULL THEN 0
               END as label
        FROM screen_diffs d
        LEFT JOIN diff_approvals a ON a.diff_id = d.id
        LEFT JOIN diff_rejections r ON r.diff_id = d.id
        WHERE label IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    X, y = [], []
    for row in rows:
        diff = {
            "removed": json.loads(row[1] or "[]"),
            "added": json.loads(row[2] or "[]"),
            "modified": json.loads(row[3] or "[]")
        }
        feats = extract_diff_features(diff)
        X.append(feats)
        y.append(row[4])
    return np.vstack(X), np.array(y)

def train_and_save():
    X, y = load_labeled_diffs()
    if len(X) == 0:
        print("⚠️ No hay datos etiquetados para entrenar.")
        return
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    dump(model, MODEL_PATH)
    print(f"✅ diff-model entrenado y guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
