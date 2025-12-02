# models_pipeline.py
import os
import json
import sqlite3
import hashlib
import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib 
from joblib import dump, load
# sklearn helpers (KMeans, classifier for hybrid)
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List
import logging
from difflib import SequenceMatcher
from SiameseEncoder import SiameseEncoder
import difflib

 

# HMM
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None  # si no está disponible, evitamos crash y solo usaremos KMeans/classifier

logger = logging.getLogger(__name__)
DB_NAME = "accessibility.db"  # ajusta según tu config
MIN_HMM_SAMPLES = 15  # mínimo para entrenar HMM por pantalla
MODEL_BASE = "models"  # estructura confirmada:
# models/{app_name}/{tester_id}/{build_id}/{screen_id}/hmm.joblib
# models/{app_name}/general/{screen_id}/hmm.joblib
TRAIN_GENERAL_ON_COLLECT = True
encoder = SiameseEncoder()  



# ----------------------------
# Helpers de I/O de modelos
# ----------------------------
def model_dir_for(app_name: str, tester_id: str, build_id: str, screen_id: str) -> str:
    return os.path.join(MODEL_BASE, app_name or "default_app", tester_id or "general", str(build_id or "latest"), str(screen_id))

def model_dir_general(app_name: str, screen_id_short: str) -> str:
    return os.path.join(MODEL_BASE, app_name or "default_app", "general", str(screen_id_short))

def save_model(obj, path: str):
    # print("🔥 SAVE PATH:", path)
    # print("🔥 DIRECTORY EXISTS?:", os.path.exists(os.path.dirname(path)))
    # print("🔥 FILE EXISTS BEFORE?:", os.path.exists(path))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"✅ Modelo guardado en: {path}")

def load_model(path: str):
    if os.path.exists(path):
        logger.info(f"📦 Cargando modelo desde: {path}")
        return joblib.load(path)
    logger.warning(f"⚠️ No se encontró modelo en: {path}")
    return None

def load_incremental_model(tester_id: str, build_id: str, app_name: str, screen_id: str):
    path = os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hybrid_incremental.joblib")
    return load_model(path)

def load_general_model(app_name: str, screen_id: str):
    path = os.path.join(model_dir_general(app_name, screen_id), "hybrid_general.joblib")
    return load_model(path)

def normalize_class_name(cls):
    """Normaliza nombres de clase para evitar duplicaciones entre Compose y Views."""
    if not cls:
        return ""
    if "ComposeView" in cls:
        return "ComposeContainer"
    if "ViewFactoryHolder" in cls:
        return "ComposeInterop"
    return cls

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
    node["className"] = normalize_class_name(node.get("className"))
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
        
                # --- CASO ESPECIAL: ENABLED ---
        if k == "enabled":
            if "enabled" in node or node.get("android:enabled") is not None:
                val = node.get("enabled", node.get("android:enabled"))
                if isinstance(val, bool):
                    normalized[k] = val
                elif isinstance(val, str):
                    normalized[k] = val.strip().lower() in ("true", "1", "yes", "enabled")
                else:
                    normalized[k] = False
            else:
                normalized[k] = False
            continue
        
                # --- PROGRESSBAR, SEEKBAR: progress / max / value ---
        if k in {"progress", "max", "value"}:
            try:
                normalized[k] = float(v)
            except (TypeError, ValueError):
                normalized[k] = 0.0
            continue

        # --- RATINGBAR: rating ---
        if k == "rating":
            try:
                normalized[k] = float(v)
            except (TypeError, ValueError):
                normalized[k] = 0.0
            continue

        # --- VISTAS CON OPACIDAD (alpha) ---
        if k == "alpha":
            try:
                normalized[k] = float(v)
            except (TypeError, ValueError):
                normalized[k] = 1.0  # totalmente visible por defecto
            continue        

        # --- RATINGBAR: rating / level ---
        if k in {"rating", "level"}:
            try:
                normalized[k] = float(v)
            except (TypeError, ValueError):
                normalized[k] = 0.0
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

    # nn = normalize_node(n)
    nn = normalize_node(n).copy()

    # ⚠️ Excluir atributos efímeros del cálculo de la clave
    for ephemeral in ("checked", "enabled", "selected", "pressed", "activated", "visible"):
        nn.pop(ephemeral, None)

    
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
        key = f"{pkg}|{full_class_sig}|subtree:{subtree_hash(nn)}"

    # estados relevantes
    state_parts = []
    if norm_class in ("ratingbar", "seekbar"):
        if nn.get("rating") is not None:
            state_parts.append(f"rating:{nn.get('rating')}")
        if nn.get("progress") is not None:
            state_parts.append(f"progress:{nn.get('progress')}")
    if state_parts:
        key += "|" + "|".join(state_parts)
    return key

def overlap_ratio(old_nodes, new_nodes):
    def extract_texts(nodes):
        texts = set()
        for n in nodes:
            nn = normalize_node(n)
            txt = (nn.get("text") or "").strip().lower()
            if txt:
                texts.add(txt)
        return texts

    old_texts = extract_texts(old_nodes)
    new_texts = extract_texts(new_nodes)

    if not old_texts and not new_texts:
        return 1.0

    inter = len(old_texts & new_texts)
    union = len(old_texts | new_texts)
    return inter / max(union, 1)


# def overlap_ratio(old_nodes, new_nodes):
#     # texto simple overlap
#     old_texts = {normalize_node(n).get("text","").strip().lower() for n in old_nodes if normalize_node(n).get("text")}
#     new_texts = {normalize_node(n).get("text","").strip().lower() for n in new_nodes if normalize_node(n).get("text")}

#     if not old_texts and not new_texts:
#         return 1.0
#     inter = len(old_texts & new_texts)
#     union = len(old_texts | new_texts)
#     return inter / max(union, 1)

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
    Early pre-check uses global_signature and partial_signature.
    """

    old_tree = ensure_list(old_tree)
    new_tree = ensure_list(new_tree)

    from backend import stable_signature 
    # ---------------------------
    # 🔹 Detección rápida por firma global / parcial
    # ---------------------------
    old_sig = stable_signature(old_tree)
    new_sig = stable_signature(new_tree)

    # if old_sig["global_signature"] == new_sig["global_signature"]:
    #     logger.info("💡 No hay cambios globales (estructura UI idéntica).")
    #     if old_sig["partial_signature"] == new_sig["partial_signature"]:
    #         # Nada cambió ni en layout ni en contenido clave
    #         return {
    #             "removed": [], "added": [], "modified": [],
    #             "text_diff": {"removed_texts": [], "added_texts": [], "overlap_ratio": 1.0},
    #             "structure_similarity": 1.0,
    #             "has_changes": False,
    #             "signature_check": "identical",
    #             "signatures": {"old": old_sig, "new": new_sig}
    #         }
    #     else:
    #         logger.info("⚠️ Mismo layout global pero distinto contenido parcial. Continuando análisis fino...")
    # else:
    #     logger.info("🔄 Cambio de firma global detectado → posible cambio de pantalla completa.")


     # 🔹 Filtrar nodos efímeros ANTES de indexar
    EPHEMERAL_CLASSES = {
        "android.view.View",
        "android.widget.FrameLayout",
        "androidx.compose.ui.platform.ComposeView",
        "androidx.compose.ui.viewinterop.ViewFactoryHolder",
        "androidx.compose.ui.node.ComposeNode",
        "androidx.compose.ui.platform.AbstractComposeView",
    }

    COMPOSE_PREFIXES = (
        "androidx.compose.",
        "com.google.accompanist.",
    )


    def filter_ephemeral_nodes(tree):
        """Elimina nodos efímeros que no afectan la UI visible, pero mantiene los Compose visibles."""
        filtered = []
        for _, n in flatten_tree(tree):
            cls = n.get("className", "")
            has_visible_info = bool(n.get("text") or n.get("contentDescription") or n.get("stateDescription"))
            # Si es Compose y tiene semántica, no lo elimines
            if is_compose_node(n) and has_visible_info:
                filtered.append(n)
            elif cls in EPHEMERAL_CLASSES and not has_visible_info:
                continue
            else:
                filtered.append(n)
        return filtered

    # def filter_ephemeral_nodes(tree):
    #     """Elimina nodos efímeros que no afectan la UI visible."""
    #     return [
    #         n for _, n in flatten_tree(tree)
    #         if not (
    #             n.get("className") in EPHEMERAL_CLASSES
    #             and not (n.get("text") or n.get("contentDescription"))
    #         )
    #     ]

    old_tree = preprocess_tree(old_tree)
    new_tree = preprocess_tree(new_tree)

    # 🔹 Aplanar una sola vez para todo el análisis
    flat_old = flatten_tree(old_tree)
    flat_new = flatten_tree(new_tree)

    # filtered_old_nodes = filter_ephemeral_nodes(old_tree)
    # filtered_new_nodes = filter_ephemeral_nodes(new_tree)


    # old_tree = filter_ephemeral_nodes(old_tree)
    # new_tree = filter_ephemeral_nodes(new_tree)

    # if not filtered_old_nodes:
    #     logger.info("No previous snapshot - base initial")
    #     return {
    #         "removed": [], "added": [], "modified": [],
    #         "text_diff": {"removed_texts": [], "added_texts": [], "overlap_ratio": 1.0},
    #         "structure_similarity": 1.0,
    #         "has_changes": True  # new snapshot counts as change
    #     }

        # 🔹 Filtrar nodos efímeros sobre las listas aplanadas
    filtered_old_nodes = [
        n for _, n in flat_old
        if not (
            n.get("className") in EPHEMERAL_CLASSES and
            not (n.get("text") or n.get("contentDescription"))
        )
    ]
    filtered_new_nodes = [
        n for _, n in flat_new
        if not (
            n.get("className") in EPHEMERAL_CLASSES and
            not (n.get("text") or n.get("contentDescription"))
        )
    ]

    # 🔹 Si no hay snapshot previo válido
    if not filtered_old_nodes:
        logger.info("No previous snapshot - base initial")
        return {
            "removed": [], "added": [], "modified": [],
            "text_diff": {"removed_texts": [], "added_texts": [], "overlap_ratio": 1.0},
            "structure_similarity": 1.0,
            "has_changes": True  # new snapshot counts as change
        }


    # ---------------- index trees (keys)

    for i, n in enumerate(filtered_old_nodes[:10]):
        if not isinstance(n, dict):
            print(f"⚠️ old_tree[{i}] no es dict, es {type(n).__name__}: {str(n)[:120]}")

    for i, n in enumerate(filtered_new_nodes[:10]):
        if not isinstance(n, dict):
            print(f"⚠️ new_tree[{i}] no es dict, es {type(n).__name__}: {str(n)[:120]}")

    old_idx = {make_key(n): normalize_node(n) for n in filtered_old_nodes}
    new_idx = {make_key(n): normalize_node(n) for n in filtered_new_nodes}

    print(f"\n📊 Total nodos old={len(old_idx)}, new={len(new_idx)}")
    print(f"🔑 Ejemplo claves old: {list(old_idx.keys())[:5]}")
    print(f"🔑 Ejemplo claves new: {list(new_idx.keys())[:5]}")

    common_keys = set(old_idx.keys()) & set(new_idx.keys())
    removed_keys = set(old_idx.keys()) - set(new_idx.keys())
    added_keys = set(new_idx.keys()) - set(old_idx.keys())

    print(f"➡️ Common: {len(common_keys)}, Added: {len(added_keys)}, Removed: {len(removed_keys)}")

    if removed_keys:
        print("❌ Removed keys (sample):", list(removed_keys)[:3])
    if added_keys:
        print("🆕 Added keys (sample):", list(added_keys)[:3])

    # old_idx = {}
    # for path, node in flatten_tree(old_tree):
    #     old_idx[make_key(node)] = normalize_node(node)

    # new_idx = {}
    # for path, node in flatten_tree(new_tree):
    #     new_idx[make_key(node)] = normalize_node(node)

    # debug text overlap raw
    # old_nodes = [n for _, n in flatten_tree(old_tree)]
    # new_nodes = [n for _, n in flatten_tree(new_tree)]

    # text_overlap_raw = overlap_ratio(old_nodes, new_nodes)
    text_overlap_raw = overlap_ratio(filtered_old_nodes, filtered_new_nodes)


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
        changes = {}
        
        if k in old_idx:
            oldn = old_idx[k]
            changes = {}
            print(f"\n🔍 Comparando nodo {k}")
            print(f"    ↳ old_text={oldn.get('text')} | new_text={nn.get('text')}")
            print(f"\n🔹 Comparando nodo: {k} ({nn.get('className')})")

            for f in TEXT_FIELDS + BOOL_FIELDS + NUM_FIELDS + OTHER_FIELDS + VISUAL_FIELDS:
                old_val, new_val = oldn.get(f), nn.get(f)

                            # Normalizar booleanos
                if f in BOOL_FIELDS:
                    old_val, new_val = to_bool(old_val), to_bool(new_val)

                if old_val != new_val:
                    print(f"    ⚠️ Campo distinto: {f}  ({old_val} → {new_val})")

                    # si es campo efímero y no relevante, lo ignoramos como cambio funcional
                    if f in IGNORED_STATE_FIELDS and old_val is False and new_val is True:
                        ignored_changes.append((nn.get("className"), f, (old_val, new_val)))
                        #print(f"⚠️ Ignorado (efímero) {f}: {old_val} → {new_val}")
                    else:
                        changes[f] = {"old": old_val, "new": new_val}
                        #print(f"✅ Cambio detectado {f}: {old_val} → {new_val}")
                    # --- chequeo extra para CheckBox y Switch ---

                
                if old_val != new_val:
                    print(f"    ⚠️ Campo distinto 2: {f}  ({old_val} → {new_val})")

                    cls = nn.get("className", "")
                    is_compose = cls.startswith("androidx.compose") or "compose" in cls.lower()

                    # 🔸 Caso 1: Vista clásica → ignora cambios efímeros (enabled, focusable, etc.)
                    if not is_compose and f in IGNORED_STATE_FIELDS and old_val is False and new_val is True:
                        ignored_changes.append((cls, f, (old_val, new_val)))
                        #print(f"⚠️ Ignorado (efímero View) {f}: {old_val} → {new_val}")

                    # 🔸 Caso 2: Compose → conserva incluso cambios en campos efímeros
                    elif is_compose and f in IGNORED_STATE_FIELDS:
                        changes[f] = {"old": old_val, "new": new_val}
                        #print(f"✅ Cambio Compose significativo {f}: {old_val} → {new_val}")

                    # 🔸 Caso 3: Resto de campos normales (texto, checked, etc.)
                    else:
                        changes[f] = {"old": old_val, "new": new_val}
                        #print(f"✅ Cambio detectado {f}: {old_val} → {new_val}")

                # def is_compose_class(cls: str) -> bool:
                #     cls = (cls or "").lower()
                #     return (
                #         "androidx.compose" in cls or
                #         "composeview" in cls or
                #         "viewfactoryholder" in cls
                #     )

                # if old_val != new_val:
                #     cls = nn.get("className", "")

                #     if is_compose_class(cls):
                #         # Jetpack Compose → no ignorar cambios efímeros
                #         changes[f] = {"old": old_val, "new": new_val}
                #         #print(f"✅ Compose change {f}: {old_val} → {new_val}")

                #     elif f in IGNORED_STATE_FIELDS and old_val is False and new_val is True:
                #         # Views clásicas → ignorar efímeros
                #         ignored_changes.append((cls, f, (old_val, new_val)))
                #         #print(f"⚠️ Ignorado (efímero View) {f}: {old_val} → {new_val}")

                #     else:
                #         # Caso normal
                #         changes[f] = {"old": old_val, "new": new_val}
                #         #print(f"✅ Cambio detectado {f}: {old_val} → {new_val}")
                # --- Fin del bloque de comparación de campos -

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

                    # --- chequeo extra para botones habilitados/deshabilitados ---
            if nn.get("className") in (
                "android.widget.Button",
                "com.google.android.material.button.MaterialButton",
                "android.widget.ImageButton"
            ):
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado: {old_enabled} → {new_enabled}")

            # --- chequeo extra para botones habilitados/deshabilitados ---
            if nn.get("className") in (
                "android.widget.Button",
                "com.google.android.material.button.MaterialButton",
                "android.widget.ImageButton"
            ):
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (Button): {old_enabled} → {new_enabled}")


            # --- RadioButton ---
            if nn.get("className") == "android.widget.RadioButton":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (RadioButton): {old_enabled} → {new_enabled}")


            # --- Switch ---
            if nn.get("className") == "android.widget.Switch":
                # Detectar cambios en habilitado/deshabilitado
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (Switch): {old_enabled} → {new_enabled}")

                # Detectar cambios en el estado (encendido/apagado)
                old_checked = to_bool(oldn.get("checked", False))
                new_checked = to_bool(nn.get("checked", False))
                if old_checked != new_checked:
                    changes["checked"] = {"old": old_checked, "new": new_checked}
                    print(f"✅ Cambio CHECKED detectado (Switch): {old_checked} → {new_checked}")


            # --- ToggleButton ---
            if nn.get("className") == "android.widget.ToggleButton":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (ToggleButton): {old_enabled} → {new_enabled}")

                old_checked = to_bool(oldn.get("checked", False))
                new_checked = to_bool(nn.get("checked", False))
                if old_checked != new_checked:
                    changes["checked"] = {"old": old_checked, "new": new_checked}
                    print(f"✅ Cambio CHECKED detectado (ToggleButton): {old_checked} → {new_checked}")


            # --- SeekBar ---
            if nn.get("className") == "android.widget.SeekBar":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (SeekBar): {old_enabled} → {new_enabled}")

                old_progress = float(oldn.get("progress", 0))
                new_progress = float(nn.get("progress", 0))
                if old_progress != new_progress:
                    changes["progress"] = {"old": old_progress, "new": new_progress}
                    print(f"✅ Cambio PROGRESS detectado (SeekBar): {old_progress} → {new_progress}")


            # --- ProgressBar ---
            if nn.get("className") == "android.widget.ProgressBar":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (ProgressBar): {old_enabled} → {new_enabled}")

                old_progress = float(oldn.get("progress", 0))
                new_progress = float(nn.get("progress", 0))
                if old_progress != new_progress:
                    changes["progress"] = {"old": old_progress, "new": new_progress}
                    print(f"✅ Cambio PROGRESS detectado (ProgressBar): {old_progress} → {new_progress}")


            # --- Spinner ---
            if nn.get("className") == "android.widget.Spinner":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (Spinner): {old_enabled} → {new_enabled}")


            # --- ImageView ---
            if nn.get("className") == "android.widget.ImageView":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (ImageView): {old_enabled} → {new_enabled}")


            # --- VideoView ---
            if nn.get("className") == "android.widget.VideoView":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (VideoView): {old_enabled} → {new_enabled}")


            # --- WebView ---
            if nn.get("className") == "android.webkit.WebView":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (WebView): {old_enabled} → {new_enabled}")

            # --- RatingBar ---
            if nn.get("className") == "android.widget.RatingBar":
                old_enabled = to_bool(oldn.get("enabled", True))
                new_enabled = to_bool(nn.get("enabled", True))
                if old_enabled != new_enabled:
                    changes["enabled"] = {"old": old_enabled, "new": new_enabled}
                    print(f"✅ Cambio ENABLED detectado (RatingBar): {old_enabled} → {new_enabled}")

                old_rating = float(oldn.get("rating", 0))
                new_rating = float(nn.get("rating", 0))
                if old_rating != new_rating:
                    changes["rating"] = {"old": old_rating, "new": new_rating}
                    print(f"✅ Cambio RATING detectado (RatingBar): {old_rating} → {new_rating}")        

        if changes:
            modified.append({"node": {"key": k, "class": nn.get("className")}, "changes": changes})
            #print(f"🔹 Nodo modificado registrado: {changes}")

    # detectar cambios de texto entre removed/added (moved text)


    # ---------------------------------------------------------
    # 🔹 Detección de cambios de texto entre nodos (versión semántica)
    # ---------------------------------------------------------
    
    # def extract_text_from_key(key: str) -> str:
    #     if not key:
    #         return ""
    #     parts = key.split("|")
    #     for p in parts:
    #         if "text:" in p:
    #             return p.split("text:")[-1].strip()
    #     if len(parts) > 1 and len(parts[-1]) > 2:
    #         return parts[-1].strip()
    #     return ""

    def extract_text_from_key(key: str) -> str:
        """Extrae el texto representativo de un nodo (tanto Views como Compose)."""
        if not key:
            return ""
        
        parts = key.split("|")
        candidates = []

        for p in parts:
            p = p.strip()
            # Android clásico
            if "text:" in p:
                candidates.append(p.split("text:")[-1].strip())
            elif "contentDescription:" in p:
                candidates.append(p.split("contentDescription:")[-1].strip())
            elif "hint:" in p:
                candidates.append(p.split("hint:")[-1].strip())
            # Jetpack Compose
            elif "stateDescription:" in p:
                candidates.append(p.split("stateDescription:")[-1].strip())
            elif "label:" in p:
                candidates.append(p.split("label:")[-1].strip())
            elif "role:" in p:
                candidates.append(p.split("role:")[-1].strip())
            elif "testTag:" in p:
                candidates.append(p.split("testTag:")[-1].strip())

        # Devolver el más significativo
        for c in candidates:
            if c and c.lower() not in {"", "null", "none"}:
                return c

        return ""


    def literal_similarity(a: str, b: str) -> float:
        """Similitud literal rápida"""
        a, b = (a or "").strip().lower(), (b or "").strip().lower()
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def semantic_similarity(a: str, b: str) -> float:
        """Similitud semántica usando el encoder siamés"""
        if not a or not b:
            return 0.0
        try:
            emb_a = encoder.encode_tree([{"text": a, "className": "TextView"}])
            emb_b = encoder.encode_tree([{"text": b, "className": "TextView"}])
            return float(torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0))
        except Exception:
            return literal_similarity(a, b)  # fallback

    text_diff = {}
    detected_text_mods = []

    for r in list(removed):
        r_text = extract_text_from_key(r["node"]["key"])
        for a in list(added):
            a_text = extract_text_from_key(a["node"]["key"])

            # 1️⃣ primero evalúa literal
            sim_lit = literal_similarity(r_text, a_text)

            # 2️⃣ si la similitud literal es incierta (0.2–0.8), usa el encoder
            if 0.2 < sim_lit < 0.8:
                sim_sem = semantic_similarity(r_text, a_text)
                sim = max(sim_lit, sim_sem)
            else:
                sim = sim_lit

            # 3️⃣ decide si es cambio real de texto
            if r_text and a_text and sim > 0.6:
                modified.append({
                    "node": {"key": a["node"]["key"], "class": a["node"]["class"]},
                    "changes": {
                        "text": {"old": r_text, "new": a_text},
                        "similarity": f"{sim:.2f}"
                    }
                })
                detected_text_mods.append((r_text, a_text))

    # 🔹 Limpieza de nodos ya pareados
    if detected_text_mods:
        removed = [
            r for r in removed
            if all(literal_similarity(extract_text_from_key(r["node"]["key"]), old) <= 0.6 for old, _ in detected_text_mods)
        ]
        added = [
            a for a in added
            if all(literal_similarity(extract_text_from_key(a["node"]["key"]), new) <= 0.6 for _, new in detected_text_mods)
        ]

        text_diff["modified_texts"] = detected_text_mods
        print(f"⚡ Cambios de texto detectados (híbrido): {detected_text_mods}")
        has_changes = True
    
    def similarity_ratio(a: str, b: str) -> float:
        """Calcula la similitud entre dos textos (0-1) usando difflib."""
        a, b = (a or "").strip().lower(), (b or "").strip().lower()
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    for old_path, old_node in flatten_tree(old_tree):
        old_text = normalize_node(old_node).get("text", "").strip()
        if not old_text:
            continue

        for new_path, new_node in flatten_tree(new_tree):
            if old_node.get("key") == new_node.get("key"):
                new_text = normalize_node(new_node).get("text", "").strip()
                if old_text and new_text and old_text != new_text:
                    sim = similarity_ratio(old_text, new_text)
                    if sim > 0.3:  # umbral de similitud
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

                # # si el modelo es HMM (caso especial)
                # if GaussianHMM is not None and isinstance(model, GaussianHMM):
                #     score = float(model.score(feat))
                #     pred = "hmm_score"
                #     conf = score
                # else:
                #     logger.warning(f"Error general al predecir modelo híbrido: {e}")
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

    if any("enabled" in m.get("changes", {}) for m in modified):
        logger.info("Forzando has_changes=True por cambio en 'enabled' (nodo modificado)")
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
    Usa embeddings del SiameseEncoder si hay árboles disponibles.
    Guarda: models/{app}/{tester}/{build}/{screen}/hybrid_incremental.joblib
    """
    if enriched_vector is None or np.count_nonzero(enriched_vector) == 0:
        logger.debug("enriched_vector vacío - skip")
        return

    # === Recuperar los árboles UI previos ===
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT collect_node_tree
            FROM accessibility_data
            WHERE collect_node_tree IS NOT NULL AND screens_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (screen_id, max(10, min_samples * 10)))
        rows = c.fetchall()

    trees = []
    for r in rows:
        try:
            tree = json.loads(r[0])
            if isinstance(tree, list):
                trees.append(tree)
        except Exception:
            continue

    # === Generar embeddings con el modelo siamés (si hay árboles) ===
    from SiameseEncoder import SiameseEncoder # asegúrate de tenerlo global o importado
    siamese_model = SiameseEncoder()

    X = None
    if trees:
        try:
            emb_batch = siamese_model.encode_batch(trees)
            X = emb_batch.cpu().numpy()
            logger.info(f"✅ Generados {len(X)} embeddings desde árboles UI.")
        except Exception as e:
            logger.warning(f"Fallo al generar embeddings con SiameseEncoder: {e}")
            X = None

    # Si no hay árboles válidos, usar enriched_vector como respaldo
    if X is None or len(X) == 0:
        X = np.array([enriched_vector])

    # Normalizar longitudes
    EXPECTED_LEN = X.shape[1]
    enriched_vector = enriched_vector.flatten()
    if len(enriched_vector) < EXPECTED_LEN:
        enriched_vector = np.pad(enriched_vector, (0, EXPECTED_LEN - len(enriched_vector)))
    else:
        enriched_vector = enriched_vector[:EXPECTED_LEN]

    # Agregar vector actual al dataset
    X = np.unique(np.vstack([enriched_vector.reshape(1, -1), X]), axis=0)

    if len(X) < min_samples:
        logger.debug("No hay suficientes muestras para incremental: %s < %s", len(X), min_samples)
        return

    # === Entrenamiento híbrido (KMeans + RF) ===
    kmeans = MiniBatchKMeans(
        n_clusters=min(5, max(1, len(X))),
        random_state=42,
        batch_size=min(100, len(X))
    )
    kmeans.fit(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)

    # Etiquetas dummy (idéntico o distinto)
    y = [0 if np.allclose(v, enriched_vector, atol=1e-8) else 1 for v in X]
    clf.fit(Xs, y)

    # === Guardado del modelo híbrido ===
    model_obj = {"kmeans": kmeans, "scaler": scaler, "clf": clf}
    path = os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hybrid_incremental.joblib")
    save_model(model_obj, path)
    logger.info("💾 Saved incremental hybrid model: %s", path)

    # === HMM opcional ===
    if GaussianHMM is not None and len(X) >= MIN_HMM_SAMPLES:
        try:
            n_components = max(2, min(5, len(X) // 10))
            hmm = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=200, tol=1e-3)
            hmm.fit(X)
            save_model(hmm, os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hmm.joblib"))
            logger.info("Saved incremental HMM: %s", os.path.join(model_dir_for(app_name, tester_id, build_id, screen_id), "hmm.joblib"))
        except Exception as e:
            logger.warning(f"HMM incremental failed: {e}")

async def _train_general_logic_hybrid(
    app_name: str,
    batch_size: int = 1000,
    min_samples: int = 1,
    update_general: bool = False
):
    """
    Entrena modelos generales por pantalla (solo si update_general=True).
    Usa embeddings del SiameseEncoder si hay árboles disponibles.
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
            WHERE collect_node_tree IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (batch_size,))
        rows = c.fetchall()

    if not rows:
        logger.warning("⚠️ No hay datos para entrenamiento general.")
        return

    # agrupar árboles y vectores por pantalla
    groups = {}
    for collect_node_tree, enriched_vec, screen_id_short in rows:
        if not screen_id_short:
            continue

        entry = groups.setdefault(screen_id_short, {"trees": [], "vecs": []})

        # árbol de accesibilidad
        try:
            tree = json.loads(collect_node_tree)
            if isinstance(tree, list):
                entry["trees"].append(tree)
        except Exception:
            pass

        # vector enriquecido
        try:
            if enriched_vec:
                entry["vecs"].append(np.array(json.loads(enriched_vec), dtype=float).flatten())
        except Exception:
            continue

    from SiameseEncoder import SiameseEncoder  # asegúrate de tenerlo global
    siamese_model = SiameseEncoder()

    # entrenar un modelo general por pantalla
    for screen_id_short, data in groups.items():
        trees = data["trees"]
        vecs = data["vecs"]

        X = None
        # 1️⃣ intentar generar embeddings desde árboles
        if trees:
            try:
                emb_batch = siamese_model.encode_batch(trees)
                X = emb_batch.cpu().numpy()
                logger.info(f"✅ Generados {len(X)} embeddings Siamese para screen {screen_id_short}")
            except Exception as e:
                logger.warning(f"Fallo al generar embeddings Siamese para {screen_id_short}: {e}")
                X = None

        # 2️⃣ fallback: usar enriched_vectors si no hay árboles válidos
        if X is None or len(X) == 0:
            if vecs:
                X = np.array(vecs)
                logger.debug(f"Usando enriched_vectors para screen {screen_id_short} ({len(X)} muestras)")
            else:
                logger.debug(f"No hay datos válidos para screen {screen_id_short}")
                continue

        # 3️⃣ limpieza
        X = np.array([v for v in X if np.count_nonzero(v) > 0])
        if len(X) < min_samples:
            logger.debug("No hay suficientes muestras para general: %s < %s", len(X), min_samples)
            continue

        # 4️⃣ clustering + clasificador
        kmeans = MiniBatchKMeans(
            n_clusters=min(5, max(1, len(X))),
            random_state=42,
            batch_size=min(100, len(X))
        )
        kmeans.fit(X)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # etiquetas dummy (0/1 aleatorias o secuenciales)
        y = [0 if i % 2 == 0 else 1 for i in range(len(Xs))]
        clf.fit(Xs, y)

        model_obj = {"kmeans": kmeans, "scaler": scaler, "clf": clf}
        safe_id = screen_id_short.replace("|", "_").replace("=", "-")
        model_path = os.path.join(model_dir_general(app_name, safe_id), "hybrid_general.joblib")
        # model_path = os.path.join(model_dir_general(app_name, screen_id), "hybrid_general.joblib")
        save_model(model_obj, model_path)
        logger.info("💾 Saved general hybrid model: %s", model_path)

        # 5️⃣ HMM opcional
        if GaussianHMM is not None and len(X) >= MIN_HMM_SAMPLES:
            try:
                n_components = max(2, min(5, len(X) // 10))
                hmm = GaussianHMM(
                    n_components=n_components,
                    covariance_type="diag",
                    n_iter=200,
                    tol=1e-3
                )
                hmm.fit(X)
                save_model(hmm, os.path.join(model_dir_general(app_name, screen_id_short), "hmm.joblib"))
                logger.info("Saved general HMM for %s / screen %s", app_name, screen_id_short)
            except Exception as e:
                logger.warning("HMM general failed: %s", e)
