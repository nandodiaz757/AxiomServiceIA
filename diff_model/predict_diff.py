import json, sqlite3, numpy as np
from fastapi import APIRouter, HTTPException
from joblib import load
from diff_model.diff_features import extract_diff_features

router = APIRouter()
model = load("diff_model.joblib")

@router.get("/diff/score/{diff_id}")
def score_diff(diff_id: int):
    conn = sqlite3.connect("accessibility.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT removed, added, modified
        FROM screen_diffs
        WHERE id = ?
    """, (diff_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Diff not found")

    diff = {
        "removed": json.loads(row[0] or "[]"),
        "added": json.loads(row[1] or "[]"),
        "modified": json.loads(row[2] or "[]")
    }
    feats = extract_diff_features(diff).reshape(1, -1)
    proba = model.predict_proba(feats)[0, 1]  # probabilidad de “cambio real”

    return {"diff_id": diff_id, "real_change_score": float(proba)}
