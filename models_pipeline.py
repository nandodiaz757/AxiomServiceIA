# models_pipeline.py
import os
import json
import sqlite3
import hashlib
import logging
from typing import Optional, Tuple

import numpy as np
import joblib

# sklearn helpers (KMeans, classifier for hybrid)
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# HMM
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None  # si no está disponible, evitamos crash y solo usaremos KMeans/classifier

logger = logging.getLogger(__name__)
DB_NAME = "accessibility.db"  # ajusta según tu config
MIN_HMM_SAMPLES = 50  # mínimo para entrenar HMM por pantalla
MODEL_BASE = "models"  # estructura confirmada:
# models/{app_name}/{tester_id}/{build_id}/{screen_id}/hmm.joblib
# models/{app_name}/general/{screen_id}/hmm.joblib

# ----------------------------
# Helpers de I/O de modelos
# ----------------------------
def model_dir_for(app_name: str, tester_id: str, build_id: str, screen_id: str) -> str:
    return os.path.join(MODEL_BASE, app_name or "default_app", tester_id or "general", str(build_id or "latest"), str(screen_id))

def model_dir_general(app_name: str, screen_id: str) -> str:
    return os.path.join(MODEL_BASE, app_name or "default_app", "general", str(screen_id))

def save_model(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_incremental_model(tester_id: str, build_id: str, app_name: str, screen_id: str):
    path = os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hybrid_incremental.joblib")
    return load_model(path)

def load_general_model(app_name: str, screen_id: str):
    path = os.path.join(model_dir_general(app_name, screen_id), "hybrid_general.joblib")
    return load_model(path)

# ----------------------------
# Feature extraction helpers
# ----------------------------
def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def preprocess_tree(tree):
    # placeholder: tu preprocesamiento real (remover wrappers, compactar, etc.)
    return ensure_list(tree)

def flatten_tree(tree):
    result = []
    def _walk(subtree, path=[]):
        if isinstance(subtree, dict):
            result.append((list(path), subtree))
            for i, ch in enumerate(subtree.get("children") or []):
                _walk(ch, path + [i])
        elif isinstance(subtree, list):
            for i, n in enumerate(subtree):
                _walk(n, path + [i])
    _walk(tree, [])
    return result


def normalize_node(node: dict) -> dict:
    # Mantén la lógica de normalización que ya tienes (SAFE_KEYS, tipos, etc.)
    SAFE_KEYS = [
        "viewId", "className", "headerText", "text", "contentDescription", "desc", "hint",
        "checked", "enabled", "focusable", "clickable", "selected", "scrollable",
        "password", "pressed", "activated", "visible", "progress", "max", "value", "rating", "level",
        "inputType", "orientation", "index", "layoutParams", "pkg", "textColor", "backgroundColor", "fontSize", "alpha"
    ]
    BOOL_FIELDS = {"checked", "enabled", "focusable", "clickable", "selected", "scrollable", "password", "pressed", "activated", "visible"}
    NUM_FIELDS = {"progress", "max", "value", "rating", "level"}

    normalized = {}
    for k in SAFE_KEYS:
        v = node.get(k)


        if k == "checked":
            if "checked" in node:  # solo procesar si existe en el nodo original
                val = node["checked"]
                if isinstance(val, bool):
                    normalized[k] = val
                elif isinstance(val, str):
                    normalized[k] = val.strip().lower() in ("true", "1", "yes", "checked")
                elif node.get("android:checked") is not None:
                    v2 = str(node.get("android:checked")).strip().lower()
                    normalized[k] = v2 in ("true", "1", "yes", "checked")
                else:
                    normalized[k] = False
            else:
                # si no existía, no tocarlo
                normalized[k] = node.get("checked", None)
            continue  

        if k in BOOL_FIELDS:
            normalized[k] = bool(v) if v not in (None, "", "null") else False
        elif k in NUM_FIELDS:
            try:
                normalized[k] = float(v)
            except Exception:
                normalized[k] = None
        elif v is None:
            normalized[k] = ""
        else:
            normalized[k] = str(v).strip()
    # quitar efímeros
    for ef in ("pressed", "focused", "activated", "selected", "visible"):
        normalized.pop(ef, None)
    return normalized

def subtree_hash(node):
    node_copy = dict(node)
    children = node_copy.pop("children", [])
    node_str = json.dumps(node_copy, sort_keys=True, ensure_ascii=False)
    h = hashlib.md5(node_str.encode()).hexdigest()
    for c in children or []:
        h = hashlib.md5((h + subtree_hash(c)).encode()).hexdigest()
    return h[:20]

def make_key(n):
    nn = normalize_node(n)
    cls = (nn.get("className") or "").strip()
    pkg = (nn.get("pkg") or "").strip()
    view_id = (nn.get("viewId") or "").strip()
    text_like = (nn.get("text") or nn.get("desc") or nn.get("contentDescription") or nn.get("hint") or "").strip()
    parts = cls.split(".")
    norm_class = parts[-1].lower() if parts else ""
    base_class = parts[-2].lower() if len(parts) > 1 else ""
    full_class_sig = f"{base_class}.{norm_class}" if base_class else norm_class

    if view_id:
        key = f"{pkg}|{full_class_sig}|{view_id}"
    elif text_like:
        safe_text = text_like.replace("\n", " ").strip()[:200]
        key = f"{pkg}|{full_class_sig}|textsig:{safe_text}"
    else:
        key = f"{pkg}|{full_class_sig}|subtree:{subtree_hash(n)}"

    # estados relevantes
    state_parts = []
    # if norm_class in ("checkbox", "radiobutton", "switch"):
    #     state_parts.append(f"checked:{nn.get('checked')}")
    if norm_class in ("ratingbar", "seekbar"):
        if nn.get("rating") is not None:
            state_parts.append(f"rating:{nn.get('rating')}")
        if nn.get("progress") is not None:
            state_parts.append(f"progress:{nn.get('progress')}")
    if state_parts:
        key += "|" + "|".join(state_parts)
    return key

def overlap_ratio(old_nodes, new_nodes):
    # texto simple overlap
    old_texts = {normalize_node(n).get("text","").strip().lower() for n in old_nodes if normalize_node(n).get("text")}
    new_texts = {normalize_node(n).get("text","").strip().lower() for n in new_nodes if normalize_node(n).get("text")}

    if not old_texts and not new_texts:
        return 1.0
    inter = len(old_texts & new_texts)
    union = len(old_texts | new_texts)
    return inter / max(union, 1)

def to_bool(val):
    """
    Convierte cualquier valor a booleano.
    Strings "true"/"false" se convierten a True/False.
    Otros valores se convierten usando bool().
    """
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)

# ----------------------------
# compare_trees (tu versión completa, con integración de modelo)
# ----------------------------
def compare_trees(old_tree, new_tree, app_name: str = None,
                  tester_id: Optional[str] = None, build_id: Optional[str] = None,
                  screen_id: Optional[str] = None, use_general: bool = False):
    """
    Compare two accessibility trees with high sensitivity to ANY textual/node change.
    Integrates hybrid model feedback (incremental or general) controlled by use_general flag.
    """

    # 🔹 Filtrar nodos efímeros ANTES de indexar
    EPHEMERAL_CLASSES = {
        "android.view.View",
        "android.widget.FrameLayout",
        "androidx.compose.ui.platform.ComposeView",
        "androidx.compose.ui.viewinterop.ViewFactoryHolder",
    }

    def filter_ephemeral_nodes(tree):
        """Elimina nodos efímeros que no afectan la UI visible."""
        return [
            n for _, n in flatten_tree(tree)
            if not (
                n.get("className") in EPHEMERAL_CLASSES
                and not (n.get("text") or n.get("contentDescription"))
            )
        ]

    old_tree = ensure_list(old_tree)
    new_tree = ensure_list(new_tree)

    old_tree = preprocess_tree(old_tree)
    new_tree = preprocess_tree(new_tree)

    old_tree = filter_ephemeral_nodes(old_tree)
    new_tree = filter_ephemeral_nodes(new_tree)

    if not old_tree:
        logger.info("No previous snapshot - base initial")
        return {
            "removed": [], "added": [], "modified": [],
            "text_diff": {"removed_texts": [], "added_texts": [], "overlap_ratio": 1.0},
            "structure_similarity": 1.0,
            "has_changes": True  # new snapshot counts as change
        }

    # ---------------- index trees (keys)
    old_idx = {}
    for path, node in flatten_tree(old_tree):
        old_idx[make_key(node)] = normalize_node(node)

    new_idx = {}
    for path, node in flatten_tree(new_tree):
        new_idx[make_key(node)] = normalize_node(node)

    # debug text overlap raw
    old_nodes = [n for _, n in flatten_tree(old_tree)]
    new_nodes = [n for _, n in flatten_tree(new_tree)]
    text_overlap_raw = overlap_ratio(old_nodes, new_nodes)

    # added / removed
    added = [{"node": {"key": k, "class": v.get("className")}, "changes": {}} for k, v in new_idx.items() if k not in old_idx]
    removed = [{"node": {"key": k, "class": v.get("className")}, "changes": {}} for k, v in old_idx.items() if k not in new_idx]


    # modified
    TEXT_FIELDS = ["text", "contentDescription", "headerText", "hint"]
    BOOL_FIELDS = ["checked", "enabled", "focusable", "clickable", "selected", "scrollable", "password", "pressed", "activated", "visible"]
    NUM_FIELDS = ["progress", "max", "value", "rating", "level"]
    OTHER_FIELDS = ["className", "viewId", "pkg"]
    VISUAL_FIELDS = ["textColor", "backgroundColor", "fontSize", "alpha"]
    IGNORED_STATE_FIELDS = {"enabled", "focusable", "clickable", "scrollable", "pressed", "activated", "visible"}

    modified = []
    ignored_changes = []

    for k, nn in new_idx.items():
        if k in old_idx:
            oldn = old_idx[k]
            changes = {}

            print(f"\n🔹 Comparando nodo: {k} ({nn.get('className')})")

            for f in TEXT_FIELDS + BOOL_FIELDS + NUM_FIELDS + OTHER_FIELDS + VISUAL_FIELDS:
                old_val, new_val = oldn.get(f), nn.get(f)

                            # Normalizar booleanos
                if f in BOOL_FIELDS:
                    old_val, new_val = to_bool(old_val), to_bool(new_val)

                if old_val != new_val:
                    # si es campo efímero y no relevante, lo ignoramos como cambio funcional
                    if f in IGNORED_STATE_FIELDS and old_val is False and new_val is True:
                        ignored_changes.append((nn.get("className"), f, (old_val, new_val)))
                        #print(f"⚠️ Ignorado (efímero) {f}: {old_val} → {new_val}")
                    else:
                        changes[f] = {"old": old_val, "new": new_val}
                        #print(f"✅ Cambio detectado {f}: {old_val} → {new_val}")
                    # --- chequeo extra para CheckBox y Switch ---
            if nn.get("className") in ("android.widget.CheckBox", "android.widget.Switch"):
                print(f"[DEBUG CHECKED ATTR] oldn={oldn.get('checked')} | nn={nn.get('checked')}")
                old_checked = to_bool(oldn.get("checked", False))
                new_checked = to_bool(nn.get("checked", False))
                if old_checked != new_checked:
                    changes["checked"] = {"old": old_checked, "new": new_checked}
                    print(f"✅ Cambio CHECKED detectado: {old_checked} → {new_checked}")        

            # 🔹 Asegurarse de registrar CheckBox explícitamente
            if nn.get("className") == "android.widget.CheckBox":
                old_checked = to_bool(oldn.get("checked", False))
                new_checked = to_bool(nn.get("checked", False))
                #print(f"CheckBox actual: old_checked={old_checked}, new_checked={new_checked}")
                if old_checked != new_checked:
                    changes["checked"] = {"old": old_checked, "new": new_checked}
                    #print(f"✅ Cambio CHECKED detectado: {old_checked} → {new_checked}")


            if changes:
                modified.append({"node": {"key": k, "class": nn.get("className")}, "changes": changes})
                #print(f"🔹 Nodo modificado registrado: {changes}")

    # detectar cambios de texto entre removed/added (moved text)
    def extract_text_from_key(key: str) -> str:
        if not key:
            return ""
        parts = key.split("|")
        for p in parts:
            if "text:" in p:
                return p.split("text:")[-1].strip()
        # fallback: busca trozos legibles
        if len(parts) > 1 and len(parts[-1]) > 2:
            return parts[-1].strip()
        return ""

    import difflib
    def similarity_ratio(a: str, b: str) -> float:
        a, b = (a or "").strip().lower(), (b or "").strip().lower()
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    text_diff = {}

    detected_text_mods = []
    for r in list(removed):
        r_text = extract_text_from_key(r["node"]["key"])
        for a in list(added):
            a_text = extract_text_from_key(a["node"]["key"])
            sim = similarity_ratio(r_text, a_text)
            if r_text and a_text and sim > 0.6:
                modified.append({
                    "node": {"key": a["node"]["key"], "class": a["node"]["class"]},
                    "changes": {
                        "text": {"old": r_text, "new": a_text},
                        "similarity": f"{sim:.2f}"
                    }
                })
                detected_text_mods.append((r_text, a_text))

    # 🔹 Elimina de removed/added los que ya fueron pareados
    if detected_text_mods:
        removed = [
            r for r in removed
            if all(similarity_ratio(extract_text_from_key(r["node"]["key"]), old) <= 0.6 for old, _ in detected_text_mods)
        ]
        added = [
            a for a in added
            if all(similarity_ratio(extract_text_from_key(a["node"]["key"]), new) <= 0.6 for _, new in detected_text_mods)
        ]

        # 🧩 NUEVO: registrar los cambios de texto detectados
        text_diff["modified_texts"] = detected_text_mods
        print(f"⚠️ Cambios de texto detectados: {detected_text_mods}")

        # 🧩 NUEVO: asegurar que se marque el diff como cambio real
        has_changes = True

    # 🔹 Detectar cambios de texto entre nodos iguales por key
    for old_path, old_node in flatten_tree(old_tree):
        old_text = normalize_node(old_node).get("text", "").strip()
        if not old_text:
            continue

        for new_path, new_node in flatten_tree(new_tree):
            if old_node.get("key") == new_node.get("key"):
                new_text = normalize_node(new_node).get("text", "").strip()
                if old_text and new_text and old_text != new_text:
                    sim = similarity_ratio(old_text, new_text)
                    if sim > 0.6:  # umbral de similitud
                        modified.append({
                            "node": {"key": old_node.get("key"), "class": old_node.get("className")},
                            "changes": {
                                "text": {"old": old_text, "new": new_text},
                                "similarity": f"{sim:.2f}"
                            }
                        })
                        detected_text_mods.append((old_text, new_text))


    # global text diffs
    try:
        old_texts = {normalize_node(n).get("text","").strip().lower() for _, n in flatten_tree(old_tree) if normalize_node(n).get("text")}
        new_texts = {normalize_node(n).get("text","").strip().lower() for _, n in flatten_tree(new_tree) if normalize_node(n).get("text")}

        diff_texts = new_texts - old_texts
        removed_texts = old_texts - new_texts
    except Exception:
        diff_texts = set()
        removed_texts = set()

    # structure similarity (cosine of structural vectors or reuse overlap)
    try:
        # usar overlap_raw como proxy; si quieres, invoca tu ui_structure_similarity real
        structure_sim = float(text_overlap_raw)
    except Exception:
        structure_sim = 0.0

    # Decide has_changes con máxima sensibilidad: ANY text change or node change
    # Sensibilidad textual ULTRA: cualquier diferencia de texto, incluso espacios/Case -> cambio
    # Por eso usamos comparación exacta de strings en normalized nodes y text diffs
    # has_changes = bool(
    #     removed or added or modified
    #     or len(diff_texts) > 0
    #     or len(removed_texts) > 0
    # )

    # -------------------------------
# Decidir si hubo cambios reales en la UI
# -------------------------------

# ✅ Primero, calcular si hay algo en las listas
    has_changes = any([
        bool(removed),
        bool(added),
        bool(modified),
        bool(diff_texts),
        bool(removed_texts),
    ])

    # 🔍 Log para depuración clara
    print(f"📊 Diff summary -> removed={len(removed)}, added={len(added)}, modified={len(modified)}, "
        f"diff_texts={len(diff_texts)}, removed_texts={len(removed_texts)}, has_changes={has_changes}")

    # ✅ Si detectamos cambios en checked dentro de nodos modificados, forzar True
    if not has_changes:
        for m in modified:
            if "checked" in m.get("changes", {}):
                print("✅ Forzando has_changes=True por cambio en checked")
                has_changes = True
                break

    # -------------------------------
    # Integra modelos híbridos (fallback a general si use_general True)
    # -------------------------------
    model = None
    model_source = None
    # se intentará cargar incremental del tester/build/screen
    try:
        if tester_id and build_id and screen_id:
            model = load_incremental_model(tester_id, build_id, app_name, screen_id)
            if model is not None:
                model_source = "incremental"
        # fallback a general si está habilitado y no hay incremental
        if model is None and use_general and screen_id:
            model = load_general_model(app_name, screen_id)
            if model is not None:
                model_source = "general"
    except Exception as e:
        logger.warning("No se pudo cargar modelo: %s", e)
        model = None
        model_source = None

    model_confidence = None
    model_prediction = None
    # Si existe un modelo, evaluar y, si coincide, ajustar has_changes
    if model is not None:
        try:
            # transformar árbol a vector compatible (usa funciones de features que ya tengas)
            # Aquí usamos un vector simple: [len(removed), len(added), len(modified), structure_sim, text_overlap_raw]
            feat = np.array([[len(removed), len(added), len(modified), structure_sim, text_overlap_raw]], dtype=float)
            # normalizar si modelo lo requiere
            # Si el modelo es un Pipeline con scaler y clasificador, predict funcionará
            pred = None
            conf = None
            try:
                # 🧠 Compatibilidad extendida con modelos empaquetados (dict con scaler + clf)
                if isinstance(model, dict) and "clf" in model and "scaler" in model:
                    # 🔹 verificar número de features
                    if feat.shape[1] != model["scaler"].n_features_in_:
                        logger.warning(f"Número de features incompatible (X tiene {feat.shape[1]}, scaler espera {model['scaler'].n_features_in_}). Ignorando modelo empaquetado.")
                        pred = None
                        conf = None
                    else:
                        try:
                            Xs = model["scaler"].transform(feat)
                            pred = model["clf"].predict(Xs)[0]
                            if hasattr(model["clf"], "predict_proba"):
                                conf = float(np.max(model["clf"].predict_proba(Xs)[0]))
                            else:
                                conf = 1.0
                        except Exception as e:
                            logger.warning(f"Error al predecir con modelo empaquetado: {e}")
                            pred = None
                            conf = None
                else:
                    # 🔹 Modelos normales (Pipeline, RandomForest, etc.)
                    pred = model.predict(feat)[0]
                    if hasattr(model, "predict_proba"):
                        conf = float(np.max(model.predict_proba(feat)[0]))
                    else:
                        conf = 1.0

            except Exception as e:
                # si el modelo es HMM (caso especial)
                if GaussianHMM is not None and isinstance(model, GaussianHMM):
                    score = float(model.score(feat))
                    pred = "hmm_score"
                    conf = score
                else:
                    logger.warning(f"Error general al predecir modelo híbrido: {e}")

                # si el modelo es HMM (caso especial)
                if GaussianHMM is not None and isinstance(model, GaussianHMM):
                    score = float(model.score(feat))
                    pred = "hmm_score"
                    conf = score
                else:
                    logger.warning(f"Error general al predecir modelo híbrido: {e}")
            model_prediction = pred
            model_confidence = conf

            # Reglas: si modelo indica 'identical' (0) con confianza alta, desmarca cambios incluso si diff encuentra algo
            # (asume que tu pipeline de entrenamiento codifica 0=identical, 1=changed)
            # Solo marcar False si el modelo predice realmente identical con alta confianza
            if pred is not None and isinstance(pred, (int, np.integer)) and pred == 0 and conf is not None and conf > 0.85:
                has_changes = False
            # HMM score se maneja por heurística separada
            elif pred == "hmm_score":
                if model_confidence > -50 and structure_sim > 0.9:
                    has_changes = False
            # Si pred es None (modelo ignorado por features), NO tocar has_changes

            # Retroalimentación ligera online (si tu modelo lo soporta)
            try:
                if hasattr(model, "partial_fit"):
                    # etiqueta: 0 si no cambios, 1 si cambios (cast a int)
                    label = 1 if has_changes else 0
                    model.partial_fit(feat, [label])
                    # guardar modelo de vuelta (incremental)
                    if model_source == "incremental":
                        path = os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hybrid_incremental.joblib")
                        save_model(model, path)
                    elif model_source == "general":
                        path = os.path.join(model_dir_general(app_name, screen_id), "hybrid_general.joblib")
                        save_model(model, path)
            except Exception:
                # partial_fit no está soportado: ok, no actualizar online
                pass

        except Exception as e:
            logger.warning("Error aplicando modelo híbrido: %s", e)

    # -------------------------------
    # Garantizar que cambios funcionales en checked no se pierdan
    # -------------------------------
    # Si detectamos ANY cambio 'checked' en la lista de modified, forzamos has_changes=True
    if any("checked" in m.get("changes", {}) for m in modified):
        logger.info("Forzando has_changes=True por cambio en 'checked' (nodo modificado)")
        has_changes = True        

    # Si el modelo se ignora, conservar diffs
    if model is None or pred is None:
        has_changes = has_changes or bool(modified or added or removed)

    # devolver resultado completo
    return {
        "removed": removed,
        "added": added,
        "modified": modified,
        "text_diff": {
            "removed_texts": list(removed_texts),
            "added_texts": list(diff_texts),
            "overlap_ratio": text_overlap_raw
        },
        "structure_similarity": structure_sim,
        "has_changes": has_changes,
        "model_used": model_source,
        "model_prediction": model_prediction,
        "model_confidence": model_confidence
    }

# ----------------------------
# Entrenamiento incremental y general (por pantalla)
# ----------------------------
async def _train_incremental_logic_hybrid(
    enriched_vector: np.ndarray,
    tester_id: str,
    build_id: str,
    app_name: str,
    screen_id: str,
    min_samples: int = 1,
    use_general_as_base: bool = True
):
    """
    Entrena modelos híbridos por pantalla para QA (incremental).
    Guarda: models/{app}/{tester}/{build}/{screen}/hybrid_incremental.joblib
    """
    if enriched_vector is None or np.count_nonzero(enriched_vector) == 0:
        logger.debug("enriched_vector vacío - skip")
        return

    # tomar últimos N vectores de la DB para esa pantalla (reciente)
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT enriched_vector
            FROM accessibility_data
            WHERE enriched_vector IS NOT NULL AND screens_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (screen_id, max(10, min_samples*10)))
        rows = c.fetchall()
    X_db = []
    for r in rows:
        try:
            vec = json.loads(r[0])
            X_db.append(vec)
        except Exception:
            continue
    if not X_db:
        X = np.array([enriched_vector])
    else:
        X_db = [np.array(v, dtype=float).flatten() for v in X_db]
        # normalizar longitud
        EXPECTED_LEN = max(len(enriched_vector), max(len(v) for v in X_db))
        def pad(v):
            v = np.array(v, dtype=float).flatten()
            if len(v) < EXPECTED_LEN:
                return np.pad(v, (0, EXPECTED_LEN - len(v)))
            return v[:EXPECTED_LEN]
        X_db = np.array([pad(v) for v in X_db if np.count_nonzero(v) > 0])
        enriched_vector = pad(enriched_vector)
        X = np.unique(np.vstack([enriched_vector.reshape(1, -1), X_db]), axis=0)

    # filtrar vectores vacíos
    X = np.array([v for v in X if np.count_nonzero(v) > 0])

    if len(X) < min_samples:
        logger.debug("No hay suficientes muestras para incremental: %s < %s", len(X), min_samples)
        return

    # KMeans para clusters + un clasificador simple (RandomForest) como "híbrido"
    kmeans = MiniBatchKMeans(n_clusters=min(5, max(1, len(X))), random_state=42, batch_size= min(100, len(X)))
    kmeans.fit(X)
    labels = kmeans.predict(X)

    # pipeline simple: scaler + classifier
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=50)
    # etiqueta: 0 = identical (usaremos heurística: clusters con baja variabilidad = identical)
    # como ejemplo, generamos etiquetas dummy: si vector igual al enriched_vector -> 0 else 1
    # En producción, reemplaza etiquetas por ground-truth de QA (aprobado/no aprobado)
    y = []
    for v in X:
        y.append(0 if np.allclose(v, enriched_vector, atol=1e-8) else 1)
    clf.fit(Xs, y)

    # ensamblar objeto a guardar (kmeans + scaler + clf)
    model_obj = {"kmeans": kmeans, "scaler": scaler, "clf": clf}
    path = os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hybrid_incremental.joblib")
    save_model(model_obj, path)
    logger.info("Saved incremental hybrid model: %s", path)

    # Entrenar HMM si hay suficientes muestras y biblioteca disponible
    if GaussianHMM is not None and len(X) >= MIN_HMM_SAMPLES:
        try:
            n_components = max(2, min(5, len(X) // 10))
            hmm = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=200, tol=1e-3)
            hmm.fit(X)
            save_model(hmm, os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hmm.joblib"))
            logger.info("Saved incremental HMM: %s", os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hmm.joblib"))
        except Exception as e:
            logger.warning("HMM incremental failed: %s", e)

async def _train_general_logic_hybrid(
    app_name: str,
    batch_size: int = 1000,
    min_samples: int = 1,
    update_general: bool = False
):
    """
    Entrena modelos generales por pantalla (solo si update_general=True).
    Guarda: models/{app}/general/{screen}/hybrid_general.joblib
    """
    if not update_general:
        logger.debug("update_general=False -> skip general training")
        return

    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT collect_node_tree, enriched_vector, screens_id
            FROM accessibility_data
            WHERE collect_node_tree IS NOT NULL AND enriched_vector IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (batch_size,))
        rows = c.fetchall()

    # agrupar por pantalla
    groups = {}
    for collect_node_tree, enriched_vec, screen_id in rows:
        if not screen_id:
            continue
        try:
            vec = json.loads(enriched_vec)
        except Exception:
            continue
        groups.setdefault(screen_id, []).append(np.array(vec, dtype=float).flatten())

    for screen_id, vecs in groups.items():
        # filtrar y normalizar longitud
        if not vecs:
            continue
        EXPECTED_LEN = max(len(v) for v in vecs)
        def pad(v):
            if len(v) < EXPECTED_LEN:
                return np.pad(v, (0, EXPECTED_LEN - len(v)))
            return v[:EXPECTED_LEN]
        X = np.array([pad(v) for v in vecs if np.count_nonzero(v) > 0])
        X = np.unique(X, axis=0)
        if len(X) < min_samples:
            continue

        # entrenar KMeans + classifier
        kmeans = MiniBatchKMeans(n_clusters=min(5, max(1, len(X))), random_state=42, batch_size=min(100, len(X)))
        kmeans.fit(X)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = RandomForestClassifier(n_estimators=100)
        # etiqueta dummy: 0/1 unknown -> en producción reemplazar por ground truth
        y = [0 if i == 0 else 1 for i in range(len(Xs))]
        clf.fit(Xs, y)

        model_obj = {"kmeans": kmeans, "scaler": scaler, "clf": clf}
        save_model(model_obj, os.path.join(model_dir_general(app_name, screen_id), "hybrid_general.joblib"))
        logger.info("Saved general hybrid model for %s / screen %s", app_name, screen_id)

        # HMM general
        if GaussianHMM is not None and len(X) >= MIN_HMM_SAMPLES:
            try:
                n_components = max(2, min(5, len(X) // 10))
                hmm = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=200, tol=1e-3)
                hmm.fit(X)
                save_model(hmm, os.path.join(model_dir_general(app_name, screen_id), "hmm.joblib"))
                logger.info("Saved general HMM for %s / screen %s", app_name, screen_id)
            except Exception as e:
                logger.warning("HMM general failed: %s", e)
