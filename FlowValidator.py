import os
import sqlite3
import json
from collections import defaultdict
from datetime import datetime
from joblib import dump, load
import logging
import json
logger = logging.getLogger(__name__) # asegúrate que tu proyecto tenga un logger central
from db import get_conn_cm

DB_NAME = "accessibility.db"
FLOW_MODEL_DIR = "models/flows"


# ============================================
# 🔧 UTILIDADES
# ============================================

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(obj, path)

def load_model(path):
    return load(path)

# ============================================
# 🧩 CONSTRUCCIÓN Y ENTRENAMIENTO DE FLUJOS
# ============================================

def build_flow_trees_from_db(app_name: str):
    logger.info(f"🧠 Construyendo árbol de flujos para {app_name}...")
    
    # conn = get_conn()
    with get_conn_cm() as conn:
        with conn.cursor() as c:
            try:
                c = conn.cursor()
                c.execute("""
                    SELECT session_key, header_text
                    FROM accessibility_data
                    WHERE app_name = ? AND header_text IS NOT NULL
                    ORDER BY created_at ASC
                """, (app_name,))
                rows = c.fetchall()
                
            except Exception as e:
                logger.error(f"❌ Error obteniendo datos de accesibilidad: {e}")
                
    # finally:
    #     release_conn(conn)

    if not rows:
        logger.warning(f"⚠️ No hay datos de accesibilidad para {app_name}")
        return {}

    flows_by_session = defaultdict(list)
    for session_key, header_text in rows:
        flows_by_session[session_key].append(header_text.strip())

    flow_trees = {}
    for seq in flows_by_session.values():
        if not seq:
            continue
        root = seq[0]
        subtree = flow_trees.setdefault(root, {})
        current = subtree
        for nxt in seq[1:]:
            current = current.setdefault(nxt, {})

    os.makedirs(FLOW_MODEL_DIR, exist_ok=True)
    path = os.path.join(FLOW_MODEL_DIR, f"{app_name}_flows.joblib")
    save_model(flow_trees, path)
    logger.info(f"💾 Guardado árbol de flujos en {path}")
    return flow_trees

# ============================================
# 🔍 VALIDACIÓN DE FLUJOS OBSERVADOS
# ============================================

def get_sequence_from_db(session_key: str) -> list:
    try:
        with get_conn_cm() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT header_text
                    FROM accessibility_data
                    WHERE session_key = %s
                    ORDER BY created_at ASC
                """, (session_key,))
                rows = c.fetchall()
                return [r[0] for r in rows if r[0]]
    except Exception as e:
        logger.error(f"Error obteniendo secuencia para sesión {session_key}: {e}")  

    return [r[0] for r in rows if r[0]]

def validate_flow_sequence(app_name: str, seq: list[str]):
    if not seq:
        return {"valid": False, "reason": "Secuencia vacía"}

    model_path = os.path.join(FLOW_MODEL_DIR, f"{app_name}_flows.joblib")
    if not os.path.exists(model_path):
        return {"valid": False, "reason": "No hay modelo aprendido"}

    try:
        flow_trees = load_model(model_path)
    except Exception as e:
        return {"valid": False, "reason": f"Error al cargar modelo: {e}"}

    root = seq[0]
    if root not in flow_trees:
        return {"valid": False, "reason": f"Raíz desconocida: {root}"}

    current = flow_trees[root]
    for nxt in seq[1:]:
        if nxt not in current:
            return {"valid": False, "reason": f"Transición inválida: {nxt}"}
        current = current[nxt]

    return {"valid": True, "reason": "Flujo válido"}

# ============================================
# 🔁 ENTRENAMIENTO INCREMENTAL
# ============================================

def update_flow_trees_incremental(app_name: str, new_session_key: str):
    seq = get_sequence_from_db(new_session_key)
    if not seq:
        logger.debug(f"Sin secuencia válida para sesión {new_session_key}")
        return

    model_path = os.path.join(FLOW_MODEL_DIR, f"{app_name}_flows.joblib")
    if os.path.exists(model_path):
        flow_trees = load_model(model_path)
    else:
        flow_trees = {}

    root = seq[0]
    subtree = flow_trees.setdefault(root, {})
    current = subtree
    for nxt in seq[1:]:
        current = current.setdefault(nxt, {})

    save_model(flow_trees, model_path)
    logger.info(f"🔁 Árbol actualizado con sesión {new_session_key}")
