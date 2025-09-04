from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sqlite3
from datetime import datetime
import json
import joblib
import numpy as np
import os
import asyncio
from sklearn.cluster import MiniBatchKMeans

import os
import json
import sqlite3
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import joblib
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from sklearn.cluster import MiniBatchKMeans

app = FastAPI()

# ----------------------------
# Configuración
# ----------------------------
DB_NAME = "accessibility.db"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_GENERAL_ON_COLLECT = os.getenv("TRAIN_GENERAL_ON_COLLECT", "1") == "1"

# Locks para entrenamientos concurrentes
_model_locks: Dict[str, asyncio.Lock] = {}
def _get_lock(key: str) -> asyncio.Lock:
    if key not in _model_locks:
        _model_locks[key] = asyncio.Lock()
    return _model_locks[key]

# ----------------------------
# BD
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accessibility_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tester_id TEXT,
            build_id TEXT,
            timestamp INTEGER,
            event_type TEXT,
            event_type_name TEXT,
            package_name TEXT,
            class_name TEXT,
            text TEXT,
            content_description TEXT,
            screens_id TEXT,
            screen_names TEXT,
            header_text TEXT,
            actual_device TEXT,
            version TEXT,
            collect_node_tree TEXT,
            additional_info TEXT,
            tree_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screen_diffs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tester_id TEXT,
            build_id TEXT,
            screen_name TEXT,
            removed TEXT,
            added TEXT,
            modified TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ----------------------------
# Modelos de entrada
# ----------------------------
class AccessibilityEvent(BaseModel):
    tester_id: Optional[str] = Field(None, alias="actualDevice")
    build_id: Optional[str] = Field(None, alias="version")

    timestamp: Optional[int] = Field(None, alias="timestamp")
    event_type: Optional[int] = Field(None, alias="eventType")
    event_type_name: Optional[str] = Field(None, alias="eventTypeName")
    package_name: Optional[str] = Field(None, alias="packageName")
    class_name: Optional[str] = Field(None, alias="className")
    text: Optional[str] = Field(None, alias="text")
    content_description: Optional[str] = Field(None, alias="contentDescription")
    screens_id: Optional[str] = Field(None, alias="screensId")
    screen_names: Optional[str] = Field(None, alias="screenNames")
    header_text: Optional[str] = Field(None, alias="headerText")
    actual_device: Optional[str] = Field(None, alias="actualDevice")
    version: Optional[str] = Field(None, alias="version")
    collect_node_tree: Optional[list] = Field(None, alias="collectNodeTree")
    additional_info: Optional[Dict[str, Any]] = Field(None, alias="additionalInfo")
    tree_data: Optional[Dict[str, Any]] = Field(None, alias="treeData")

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

# ----------------------------
# Utilidades de árbol
# ----------------------------
def ensure_list(tree):
    if isinstance(tree, str):
        try:
            return json.loads(tree)
        except Exception:
            return []
    return tree or []


def compare_trees(old_tree, new_tree):
    try:
        old_tree = ensure_list(old_tree)
        new_tree = ensure_list(new_tree)

        def make_key(node):
            if not isinstance(node, dict):
                return None
            # Usa signature si existe
            if "signature" in node and node["signature"]:
                return node["signature"]
            # Si hay xpath, úsalo con className
            if "xpath" in node and node["xpath"]:
                return f"{node['xpath']}:{node.get('className', '')}"
            # Fallback a className+viewId
            if "viewId" in node and node["viewId"]:
                return f"{node.get('className', '')}:{node['viewId']}"
            # Última opción: className sola
            return node.get("className")

        old_index = {make_key(n): n for n in old_tree if isinstance(n, dict) and make_key(n)}
        new_index = {make_key(n): n for n in new_tree if isinstance(n, dict) and make_key(n)}

        removed = [n for k, n in old_index.items() if k not in new_index]
        added   = [n for k, n in new_index.items() if k not in old_index]

        modified: List[Dict[str, Any]] = []
        for k, new_node in new_index.items():
            if k in old_index:
                old_node = old_index[k]
                changes = {}
                for field in ["className", "text", "desc", "viewId", "pkg", "bounds",
                              "clickable", "enabled", "focusable", "xpath"]:
                    if old_node.get(field) != new_node.get(field):
                        changes[field] = {"old": old_node.get(field), "new": new_node.get(field)}
                if changes:
                    modified.append({
                        "node": {
                            "signature": k,
                            "className": new_node.get("className"),
                            "text": new_node.get("text"),
                            "xpath": new_node.get("xpath"),
                        },
                        "changes": changes
                    })

        return removed, added, modified

    except Exception as e:
        print("❌ Error en compare_trees:", str(e))
        return [], [], []

# ----------------------------
# Utilidades de modelo
# ----------------------------
def model_paths(tester_id: Optional[str], build_id: Optional[str]):
    if tester_id and build_id:
        base = os.path.join(MODELS_DIR, tester_id, build_id)
    elif tester_id:
        base = os.path.join(MODELS_DIR, tester_id, "default")
    else:
        base = os.path.join(MODELS_DIR, "anonymous", "default")
    os.makedirs(base, exist_ok=True)
    individual_path = os.path.join(base, "model.pkl")

    general_base = os.path.join(MODELS_DIR, "general")
    os.makedirs(general_base, exist_ok=True)
    general_path = os.path.join(general_base, "model.pkl")

    return individual_path, general_path

def load_model(path: str) -> Optional[MiniBatchKMeans]:
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def init_model_for_data(X: np.ndarray, max_k: int = 3) -> Optional[MiniBatchKMeans]:
    if X is None or len(X) == 0:
        return None
    n_clusters = max(1, min(max_k, len(np.unique(X))))
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=64)

def features_from_rows(rows) -> np.ndarray:
    X = np.array([int(r[0]) for r in rows if str(r[0]).isdigit()]).reshape(-1, 1)
    return X

# ----------------------------
# Entrenamiento
# ----------------------------
async def _train_incremental_logic(tester_id: Optional[str],
                                   build_id: Optional[str],
                                   batch_size: int = 200):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_type
        FROM accessibility_data
        WHERE event_type IS NOT NULL
          AND IFNULL(tester_id,'') = IFNULL(?, '')
          AND IFNULL(build_id,'')  = IFNULL(?, '')
        ORDER BY created_at DESC
        LIMIT ?
    """, (tester_id, build_id, int(batch_size)))
    rows = cursor.fetchall()
    conn.close()

    X = features_from_rows(rows)
    if len(X) == 0:
        return

    individual_path, _ = model_paths(tester_id, build_id)
    lock = _get_lock(f"ind:{tester_id}:{build_id}")

    async with lock:
        model = load_model(individual_path)
        if model is None:
            model = init_model_for_data(X, max_k=3)
            if model is None:
                return
            model.partial_fit(X)
        else:
            model.partial_fit(X)

        joblib.dump(model, individual_path)

async def _train_general_logic(batch_size: int = 1000):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_type
        FROM accessibility_data
        WHERE event_type IS NOT NULL
        ORDER BY created_at DESC
        LIMIT ?
    """, (int(batch_size),))
    rows = cursor.fetchall()
    conn.close()

    X = features_from_rows(rows)
    if len(X) == 0:
        return

    _, general_path = model_paths(None, None)
    lock = _get_lock("gen")

    async with lock:
        model = load_model(general_path)
        if model is None:
            model = init_model_for_data(X, max_k=5)
            if model is None:
                return
            model.partial_fit(X)
        else:
            model.partial_fit(X)

        joblib.dump(model, general_path)

# ----------------------------
# Worker de análisis
# ----------------------------
async def analyze_and_train(event: AccessibilityEvent):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT collect_node_tree
        FROM accessibility_data
        WHERE screen_names = ? AND IFNULL(tester_id,'') = IFNULL(?, '')
              AND IFNULL(build_id,'')  = IFNULL(?, '')
        ORDER BY created_at DESC
        LIMIT 2
    """, (event.screen_names, event.tester_id, event.build_id))
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 2:
        return

    prev_tree = json.loads(rows[1][0]) if rows[1][0] else []
    latest_tree = json.loads(rows[0][0]) if rows[0][0] else []

    # Debug antes de comparar
    # print("========== DEBUG TREE COMPARISON ==========")
    # print("Prev:", json.dumps(prev_tree, indent=2, ensure_ascii=False))
    # print("New:", json.dumps(latest_tree, indent=2, ensure_ascii=False))
    # print("===========================================")

    removed, added, modified = compare_trees(prev_tree, latest_tree)

    if removed or added or modified:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO screen_diffs (tester_id, build_id, screen_name, removed, added, modified)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event.tester_id,
            event.build_id,
            event.screen_names,
            json.dumps(removed),
            json.dumps(added),
            json.dumps(modified)
        ))
        conn.commit()
        conn.close()

        await _train_incremental_logic(event.tester_id, event.build_id)
        if TRAIN_GENERAL_ON_COLLECT:
            await _train_general_logic()
# ----------------------------
# Endpoints
# ----------------------------
@app.post("/collect")
async def collect_event(event: AccessibilityEvent):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO accessibility_data (
            tester_id, build_id, timestamp, event_type, event_type_name, package_name, class_name, text,
            content_description, screens_id, screen_names, header_text, actual_device, version,
            collect_node_tree, additional_info, tree_data, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event.tester_id,
        event.build_id,
        event.timestamp,
        str(event.event_type) if event.event_type is not None else None,
        event.event_type_name,
        event.package_name,
        event.class_name,
        event.text,
        event.content_description,
        event.screens_id,
        event.screen_names,
        event.header_text,
        event.actual_device,
        event.version,
        json.dumps(event.collect_node_tree) if event.collect_node_tree else None,
        json.dumps(event.additional_info) if event.additional_info else None,
        json.dumps(event.tree_data) if event.tree_data else None,
        datetime.now()
    ))
    conn.commit()
    conn.close()

    asyncio.create_task(analyze_and_train(event))
    return {"status": "collected"}

@app.get("/status")
async def get_status(testerId: Optional[str] = Query(None),
                     buildId: Optional[str] = Query(None),
                     limit: int = Query(5)):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT screen_name, removed, added, modified, created_at
        FROM screen_diffs
        WHERE IFNULL(tester_id,'') = IFNULL(?, '')
          AND IFNULL(build_id,'')  = IFNULL(?, '')
        ORDER BY created_at DESC
        LIMIT ?
    """, (testerId, buildId, int(limit)))
    rows = cursor.fetchall()
    conn.close()

    diffs = [
        {
            "screen_name": r[0],
            "removed": json.loads(r[1]),
            "added": json.loads(r[2]),
            "modified": json.loads(r[3]),
            "created_at": r[4]
        }
        for r in rows
    ]

    return {
        "status": "changes" if diffs else "no_changes",
        "diffs": diffs
    }


@app.get("/predict/cluster")
async def predict_cluster(testerId: Optional[str] = Query(None),
                          buildId: Optional[str] = Query(None),
                          batch_size: int = Query(50)):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_type
        FROM accessibility_data
        WHERE event_type IS NOT NULL
          AND IFNULL(tester_id,'') = IFNULL(?, '')
          AND IFNULL(build_id,'')  = IFNULL(?, '')
        ORDER BY created_at DESC
        LIMIT ?
    """, (testerId, buildId, int(batch_size)))
    rows = cursor.fetchall()
    conn.close()

    X = features_from_rows(rows)
    if len(X) == 0:
        return {"warning": "No hay datos recientes con event_type para predecir."}

    ind_path, gen_path = model_paths(testerId, buildId)
    model = load_model(ind_path)
    using = "individual"
    if model is None:
        model = load_model(gen_path)
        using = "general"
    if model is None:
        return {"error": "No hay modelo entrenado."}

    labels = model.predict(X).tolist()
    centers = getattr(model, "cluster_centers_", None)
    return {
        "using_model": using,
        "testerId": testerId,
        "buildId": buildId,
        "samples": int(len(X)),
        "predicted_labels": labels,
        "cluster_centers": centers.tolist() if centers is not None else None
    }

# ----------------------------
# Endpoints recuperados
# ----------------------------

@app.get("/data")
async def get_data(limit: int = Query(10),
                   testerId: Optional[str] = Query(None),
                   buildId: Optional[str] = Query(None)):
    """Consultar datos crudos desde la BD"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, tester_id, build_id, event_type, event_type_name, package_name, class_name, text, created_at
        FROM accessibility_data
        WHERE IFNULL(tester_id,'') = IFNULL(?, '')
          AND IFNULL(build_id,'')  = IFNULL(?, '')
        ORDER BY created_at DESC
        LIMIT ?
    """, (testerId, buildId, int(limit)))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "tester_id": r[1],
            "build_id": r[2],
            "event_type": r[3],
            "event_type_name": r[4],
            "package_name": r[5],
            "class_name": r[6],
            "text": r[7],
            "created_at": r[8],
        }
        for r in rows
    ]


@app.post("/train/incremental")
async def train_incremental(tester_id: Optional[str] = Query(None),
                            build_id: Optional[str] = Query(None),
                            batch_size: int = Query(200)):
    """Entrenar modelo incremental individual"""
    await _train_incremental_logic(tester_id, build_id, batch_size)
    return {"status": "incremental trained", "tester_id": tester_id, "build_id": build_id}


@app.post("/train/general")
async def train_general(batch_size: int = Query(1000)):
    """Entrenar modelo general"""
    await _train_general_logic(batch_size)
    return {"status": "general trained"}


@app.get("/predict")
async def predict_changes(testerId: Optional[str] = Query(None),
                          buildId: Optional[str] = Query(None)):
    """Comparar las últimas 2 pantallas y devolver diffs"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT collect_node_tree
        FROM accessibility_data
        WHERE IFNULL(tester_id,'') = IFNULL(?, '')
          AND IFNULL(build_id,'')  = IFNULL(?, '')
        ORDER BY created_at DESC
        LIMIT 2
    """, (testerId, buildId))
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 2:
        return {"status": "no_changes"}

    prev_tree = json.loads(rows[1][0]) if rows[1][0] else []
    latest_tree = json.loads(rows[0][0]) if rows[0][0] else []

    removed, added, modified = compare_trees(prev_tree, latest_tree)

    return {
        "status": "changes" if (removed or added or modified) else "no_changes",
        "diffs": {
            "removed": removed,
            "added": added,
            "modified": modified,
        }
    }


@app.get("/model/info")
async def model_info(testerId: Optional[str] = Query(None),
                     buildId: Optional[str] = Query(None)):
    """Información de paths de modelos"""
    ind, gen = model_paths(testerId, buildId)
    return {
        "testerId": testerId,
        "buildId": buildId,
        "individual_path": ind,
        "individual_exists": os.path.exists(ind),
        "general_path": gen,
        "general_exists": os.path.exists(gen),
    }
