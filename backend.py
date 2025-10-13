from fastapi import FastAPI, Query, BackgroundTasks, Request, APIRouter, HTTPException, Depends, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
import sqlite3, json, joblib, numpy as np, os, hashlib, logging, asyncio, re, unicodedata
from hmmlearn import hmm
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
from diff_model.predict_diff import router as diff_router
from datetime import datetime
from email_service import send_email
from reset_service import generate_code, validate_code
import random, time
from collections import Counter
import math
from sklearn.cluster import KMeans 
from sklearn.cluster import MiniBatchKMeans
import httpx

# =========================================================
# CONFIGURACIÓN
# =========================================================
app = FastAPI()
router = APIRouter() 

if 'kmeans_model' not in globals():
    kmeans_model = KMeans(n_clusters=5)
if 'hmm_model' not in globals():
    hmm_model = hmm.GaussianHMM(n_components=5)

DB_NAME = "accessibility.db"
MODELS_DIR = "models"
logger = logging.getLogger("myapp")
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


ENRICHED_VECTOR_THRESHOLD = 0.5  # Ajusta este valor según tu lógica

# Estructura para historial de booleanos (para alertas de cambios)

BOOL_HISTORY = {}  
codes_db = {}
KMEANS_MODELS = {}
HMM_MODELS = {}
SEQ_LENGTH = {}  # Longitud de la secuencia para el modelo HMM 

# Limpieza inicial (por si hay residuos en hot reload)
KMEANS_MODELS.clear()
HMM_MODELS.clear()


# ===================== MODELOS BASE (fallbacks) =====================

# No necesitas inicializarlos aquí si los cargas por tester_id más adelante.
# Pero puedes tener un “modelo base” para fallback:
BASE_KMEANS = MiniBatchKMeans(n_clusters=2, random_state=42)
BASE_HMM = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=50)

# ===================== DIRECTORIOS =====================
os.makedirs(MODELS_DIR, exist_ok=True)
MODELS_DIR = os.path.join(os.getcwd(), "models", "trained")

# Locks para evitar entrenamientos simultáneos del mismo tester/build
_model_locks: Dict[str, asyncio.Lock] = {}


if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

os.makedirs(MODELS_DIR, exist_ok=True)
TRAIN_GENERAL_ON_COLLECT = True

# Locks para entrenamientos concurrentes
_model_locks: Dict[str, asyncio.Lock] = {}


class SendCodeRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str = Field(..., alias="newPassword")

class ResetRequest(BaseModel):
    email: EmailStr


# === Reentrenamiento automático del modelo diff ===
@app.on_event("startup")
@repeat_every(seconds=3600)  # cada hora, ajusta el intervalo a lo que necesites
def retrain_model() -> None:
    """
    Reentrena el modelo diff con las últimas aprobaciones/rechazos
    sin necesidad de parar el servidor.
    """
    from diff_model.train_diff_model import train_and_save
    train_and_save()


def _get_lock(key: str) -> asyncio.Lock:
    if key not in _model_locks:
        _model_locks[key] = asyncio.Lock()
    return _model_locks[key]


def sequence_entropy(seq: list[str]) -> float:
    """Calcula la entropía de Shannon de una secuencia de elementos."""
    if not seq:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    ent = -sum((count/total) * math.log2(count/total) for count in counts.values())
    return ent
# =========================================================
# BASE DE DATOS
# =========================================================

def get_db():
    """
    Devuelve una conexión SQLite que se cierra automáticamente
    al finalizar la petición.
    """
    db = sqlite3.connect(DB_NAME)
    db.row_factory = sqlite3.Row  # Para poder acceder a columnas por nombre
    try:
        yield db
    finally:
        db.close()
        
        
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
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
            signature TEXT,
            version TEXT,
            collect_node_tree TEXT,
            additional_info TEXT,
            tree_data TEXT,
            enriched_vector TEXT,
            cluster_id INTEGER,
            anomaly_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS screen_diffs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tester_id TEXT,
            build_id TEXT,
            screen_name TEXT,
            header_text TEXT,
            removed TEXT,
            added TEXT,
            modified TEXT,
            cluster_info TEXT,
            anomaly_score REAL DEFAULT 0,
            cluster_id INTEGER DEFAULT -1,  
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS boolean_history (
            screen_id TEXT,
            node_key TEXT,
            property TEXT,
            last_value BOOLEAN,
            PRIMARY KEY(screen_id, node_key, property)
        );
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS diff_trace (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tester_id TEXT,
            build_id TEXT,
            screen_name TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS diff_approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            diff_id INTEGER,
            approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS diff_rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            diff_id INTEGER,
            rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
        # --- OPCIONAL: para almacenar códigos de verificación ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS password_reset_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_data_tester_build_screen
        ON accessibility_data(tester_id, build_id, screen_names, created_at DESC)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_diffs_tester_build_screen
        ON screen_diffs(tester_id, build_id, screen_name, created_at DESC)
    """)
    conn.commit()
    conn.close()
    
init_db()

# =========================================================
# MODELOS DE ENTRADA
# =========================================================
class ActionEvent(BaseModel):
    type: str
    timestamp: float
    # otros campos que tenga cada acción

class AccessibilityEvent(BaseModel):
    # Identificación de tester y build (usa alias en camelCase para que coincida con el JSON entrante)
    tester_id: Optional[str] = Field(None, alias="actualDevice")
    build_id: Optional[str] = Field(None, alias="version")
    
    # Datos básicos del evento de accesibilidad
    timestamp: Optional[int] = Field(None, alias="timestamp")
    event_type: Optional[int] = Field(None, alias="eventType")
    event_type_name: Optional[str] = Field(None, alias="eventTypeName")
    package_name: Optional[str] = Field(None, alias="packageName")
    class_name: Optional[str] = Field(None, alias="className")
    text: Optional[str] = Field(None, alias="text")
    content_description: Optional[str] = Field(None, alias="contentDescription")
    
    # Información de flujo/pantalla
    screens_id: Optional[str] = Field(None, alias="screensId")
    screen_names: Optional[str] = Field(None, alias="screenNames")
    header_text: Optional[str] = Field(None, alias="headerText")
    
    # Datos de dispositivo y versión de la app
    actual_device: Optional[str] = Field(None, alias="actualDevices")    
    version: Optional[str] = Field(None, alias="versions")
    actions: Optional[List[ActionEvent]] = []
    
    # Árbol de nodos capturado (puede ser dict o lista de nodos)
    # collect_node_tree: Optional[Union[Dict, List]] = Field(None, alias="collectNodeTree")
    collect_node_tree: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        None, alias="collectNodeTree"
    )
    
    # Datos adicionales para enriquecer el modelo (libres)
    additional_info: Optional[Dict[str, Any]] = Field(None, alias="additionalInfo")
    tree_data: Optional[Dict[str, Any]] = Field(None, alias="treeData")

    class Config:
    # Permite poblar el modelo con los nombres de campo internos o los alias del JSON
        allow_population_by_field_name = True
    
    # Acepta campos extra que la app pueda enviar en el futuro sin romper validación
        extra = "allow"

# =========================================================
# UTILIDADES PARA ÁRBOLES Y HASH ESTABLE
# =========================================================
SAFE_KEYS = ["className", "text", "desc", "viewId", "pkg"]



def ui_structure_features(tree: dict) -> list[float]:
    """
    Cuenta componentes y propiedades de accesibilidad y agrega nuevas features.
    Devuelve:
    [buttons, text_fields, menus, recycler_views, web_views,
     enabled_count, clickable_count, focusable_count,
     visible_count, editable_count, image_views]
    """
    counts = {
        "buttons": 0,
        "text_fields": 0,
        "menus": 0,
        "recycler_views": 0,
        "web_views": 0,
        "image_views": 0
    }
    props = {
        "enabled": 0,
        "clickable": 0,
        "focusable": 0,
        "visible": 0,
        "editable": 0
    }

    def traverse(node):
        if not isinstance(node, dict):
            return
        cls = node.get("className", "")
        if "Button" in cls:        counts["buttons"]        += 1
        if "EditText" in cls:      counts["text_fields"]    += 1
        if "Menu" in cls:          counts["menus"]          += 1
        if "RecyclerView" in cls:  counts["recycler_views"] += 1
        if "WebView" in cls:       counts["web_views"]      += 1
        if "ImageView" in cls:     counts["image_views"]    += 1

        if node.get("enabled"):    props["enabled"]   += 1
        if node.get("clickable"):  props["clickable"] += 1
        if node.get("focusable"):  props["focusable"] += 1
        if node.get("visible", True):  props["visible"] += 1
        if "EditText" in cls and node.get("editable", True): props["editable"] += 1

        for ch in node.get("children", []):
            traverse(ch)

    traverse(tree)
    return list(counts.values()) + list(props.values())

def input_features(events):
    total_chars = sum(len(e.text or "") for e in events if e.type=="input")
    upper_ratio = total_chars and sum(ch.isupper() for e in events if e.type=="input" for ch in e.text)/total_chars
    action_seq  = [e.type for e in events]  # ["tap","scroll","input",...]
    seq_entropy = sequence_entropy(action_seq)
    # Nueva feature: número de taps y scrolls
    num_taps = sum(1 for e in events if e.type=="tap")
    num_scrolls = sum(1 for e in events if e.type=="scroll")
    return [total_chars, upper_ratio, seq_entropy, num_taps, num_scrolls]

def ensure_list(tree):
    if isinstance(tree, str):
        try:
            return json.loads(tree)
        except Exception:
            return []
    return tree or []

def normalize_node(node: Dict) -> Dict:
    return {k: (node.get(k) or "") for k in SAFE_KEYS}

def normalize_tree(nodes: List[Dict]) -> List[Dict]:
    return sorted([normalize_node(n) for n in nodes if isinstance(n, dict)],
                  key=lambda n: (n["className"], n["text"]))

def stable_signature(nodes: List[Dict]) -> str:
    return hashlib.sha256(json.dumps(normalize_tree(nodes), sort_keys=True).encode()).hexdigest()

def generate_code():
    """Genera un código de 6 dígitos"""
    return str(random.randint(100000, 999999))


def save_reset_code(email: str, code: str):
    try:
        expires_at = int(time.time()) + 900
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO password_reset_codes (email, code, expires_at)
            VALUES (?, ?, ?)
        """, (email, code, expires_at))
        conn.commit()
        conn.close()
        print(f"✅ Código guardado para {email}")
    except Exception as e:
        print(f"❌ Error guardando código en BD: {e}")


def validate_code(email: str, code: str, expiration_seconds: int = 300) -> bool:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT expires_at FROM password_reset_codes
        WHERE email = ? AND code = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (email, code))
    row = c.fetchone()
    conn.close()

    if not row:
        return False

    if int(time.time()) > row[0]:
        return False

    return True  


def update_bool_history(screen_id, db_conn):
    """Guarda BOOL_HISTORY en la BD."""
    cursor = db_conn.cursor()
    for node_key, props in BOOL_HISTORY.items():
        for prop, val in props.items():
            cursor.execute("""
                INSERT INTO boolean_history(screen_id, node_key, property, last_value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(screen_id, node_key, property)
                DO UPDATE SET last_value = excluded.last_value
            """, (screen_id, node_key, prop, int(val)))
    db_conn.commit()


def compare_trees(old_tree, new_tree, app_name: str = None):
    """
    Compara dos árboles de nodos accesibles y detecta nodos agregados, eliminados o modificados.
    Si existe un modelo general para la app, se usa como baseline adicional.
    """
    old_tree = ensure_list(old_tree)
    new_tree = ensure_list(new_tree)

    SAFE_KEYS = [
        "viewId", "className", "headerText", "text", "contentDescription",
        "checked", "enabled", "focusable", "clickable", "otherField"
    ]

    TEXT_FIELDS = ["text", "contentDescription", "headerText"]
    BOOL_FIELDS = ["checked", "enabled", "focusable", "clickable"]
    OTHER_FIELDS = ["className", "viewId", "otherField"]

    def normalize_node(node: dict) -> dict:
        """Convierte None a cadena vacía y asegura que todos los valores sean tipo str o bool válidos."""
        normalized = {}
        for k in SAFE_KEYS:
            v = node.get(k)
            if isinstance(v, bool):
                normalized[k] = v
            elif v is None:
                normalized[k] = ""
            else:
                normalized[k] = str(v).strip()
        return normalized

    def make_key(n, idx):
        """Crea una clave única para identificar nodos similares."""
        if not isinstance(n, dict):
            return None
        n = normalize_node(n)
        parts = [
            n.get("viewId"),
            n.get("className"),
            n.get("headerText"),
            n.get("text"),
            n.get("contentDescription"),
        ]
        return "|".join([str(p) for p in parts if p])

    # --------------------------------------------------------
    # 1️⃣ Índices de nodos previos y actuales
    # --------------------------------------------------------
    old_idx = {make_key(n, i): n for i, n in enumerate(old_tree) if make_key(n, i)}
    new_idx = {make_key(n, i): n for i, n in enumerate(new_tree) if make_key(n, i)}

    removed = [n for k, n in old_idx.items() if k not in new_idx]
    added = [n for k, n in new_idx.items() if k not in old_idx]
    modified = []

    # --------------------------------------------------------
    # 2️⃣ Comparar campo por campo entre árboles
    # --------------------------------------------------------
    for k, nn in new_idx.items():
        if k in old_idx:
            changes = {}
            old_node = normalize_node(old_idx[k])
            new_node = normalize_node(nn)

            for f in TEXT_FIELDS + BOOL_FIELDS + OTHER_FIELDS:
                old_val = old_node.get(f)
                new_val = new_node.get(f)

                if f in BOOL_FIELDS:
                    # Historial de booleanos
                    if k not in BOOL_HISTORY:
                        BOOL_HISTORY[k] = {}
                    if f not in BOOL_HISTORY[k]:
                        BOOL_HISTORY[k][f] = new_val
                        continue
                    if BOOL_HISTORY[k][f] != new_val:
                        changes[f] = {"old": BOOL_HISTORY[k][f], "new": new_val}
                        BOOL_HISTORY[k][f] = new_val
                else:
                    if old_val != new_val:
                        changes[f] = {"old": old_val, "new": new_val}

            if changes:
                modified.append({"node": {"key": k}, "changes": changes})

    # --------------------------------------------------------
    # 3️⃣ Verificar diferencias respecto al modelo general (baseline)
    # --------------------------------------------------------
    if app_name:
        baseline_path = os.path.join(MODELS_DIR, app_name, "general", "baseline_tree.json")
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline_tree = json.load(f)
                baseline_idx = {make_key(n, i): normalize_node(n)
                                for i, n in enumerate(baseline_tree) if make_key(n, i)}

                baseline_modified = []
                for m in modified:
                    key = m["node"]["key"]
                    if key in baseline_idx:
                        base_node = baseline_idx[key]
                        changes_vs_baseline = {}
                        for f, ch in m["changes"].items():
                            base_val = base_node.get(f)
                            if base_val != ch["new"]:
                                changes_vs_baseline[f] = {
                                    "old": base_val,
                                    "new": ch["new"],
                                    "relative_to": "baseline"
                                }
                        if changes_vs_baseline:
                            baseline_modified.append({
                                "node": m["node"],
                                "changes": changes_vs_baseline
                            })

                if baseline_modified:
                    logger.info(f"[compare_trees] {len(baseline_modified)} cambios respecto a baseline general de {app_name}")
                    # Fusionamos los cambios baseline en modified
                    modified.extend(baseline_modified)

            except Exception as e:
                logger.warning(f"[compare_trees] No se pudo procesar baseline de {app_name}: {e}")

    return removed, added, modified


# =========================================================
# VECTORIZACIÓN Y FEATURES
# =========================================================



def features_from_rows(rows) -> np.ndarray:
    vecs = [vector_from_tree(r[0]) for r in rows if r and r[0]]
    return np.vstack(vecs) if vecs else np.empty((0,3))
    

def vector_from_tree(tree_str: str) -> np.ndarray:
    """
    Genera un vector:
    [total_nodes, max_depth, text_nodes,
     buttons, text_fields, menus, recycler_views, web_views,
     enabled_count, clickable_count, focusable_count]
    """
    try:
        tree = json.loads(tree_str)
    except Exception:
        # Vector con 11 ceros si el JSON no es válido
        return np.zeros(11, dtype=float)

    def walk(node, depth=1):
        if not isinstance(node, dict):
            return (0, depth, 0)
        children = node.get("children", [])
        total, max_d, txt = 1, depth, 1 if node.get("text") else 0
        for ch in children:
            t, d, c = walk(ch, depth + 1)
            total += t
            max_d = max(max_d, d)
            txt += c
        return total, max_d, txt

    if isinstance(tree, list):
        totals = [walk(n) for n in tree]
        total_nodes = sum(t[0] for t in totals)
        max_depth   = max((t[1] for t in totals), default=0)
        text_nodes  = sum(t[2] for t in totals)
        # Para ui_structure_features, empaquetamos en un dict "raíz"
        struct_vec  = ui_structure_features({"children": tree})
    else:
        total_nodes, max_depth, text_nodes = walk(tree)
        struct_vec  = ui_structure_features(tree)

    base_vec = [total_nodes, max_depth, text_nodes]
    return np.array(base_vec + struct_vec, dtype=float)

def features_from_rows(rows) -> np.ndarray:
    vectors = [vector_from_tree(r[0]) for r in rows if r and r[0]]
    return np.vstack(vectors) if vectors else np.empty((0, 11))    

# =========================================================
# ENTRENAMIENTO HÍBRIDO (KMeans + HMM)
# =========================================================

# ===================== FUNCIÓN PRINCIPAL =====================
async def _train_model_hybrid(
    X,
    tester_id: str = "general",
    build_id: str = "default",
    app_name: str = "default_app", 
    lock: asyncio.Lock = None,
    max_clusters=3,        # ✅ ahora 3 clusters por defecto
    min_samples=3,         # ✅ subimos mínimo a 3 para mejor estabilidad
    desc="",
    n_hmm_states=3         # ✅ ahora usa 3 estados: estable, leve, estructural
):
    """
    Entrena modelos HMM + KMeans combinados y los guarda por tester_id/build_id.
    Si existen modelos previos, continúa el entrenamiento incrementalmente.
    """
    logger.info(f"[train_hybrid] Iniciando entrenamiento → tester_id={tester_id}, build_id={build_id}, desc={desc}")


    # ✅ Asegura que siempre haya un lock
    lock = lock or asyncio.Lock()

    async with lock:
        if len(X) < min_samples:
            logger.warning(f"[train_hybrid] tamaño de X={len(X)} < min_samples={min_samples}, desc={desc}")
            return

        # 📁 Ruta base por tester y build
        # base = os.path.join(MODELS_DIR, tester_id or "general", str(build_id) or "default")

        app_dir = os.path.join(MODELS_DIR, app_name)
        tester_dir = os.path.join(app_dir, tester_id or "general", str(build_id or "default"))
        os.makedirs(tester_dir, exist_ok=True)


        # build_folder = "default" if build_id in [None, "", "None"] else str(build_id)
        # base = os.path.join(MODELS_DIR, tester_id or "general", build_folder)
        # os.makedirs(base, exist_ok=True)

        # ===================== Cargar modelos previos =====================
        #prev_model_path = os.path.join(MODELS_DIR, tester_id or "general", str(int(build_id) - 1), "model.pkl")
        prev_kmeans, prev_hmm = None, None
        prev_model_path = None
        prev_build_id = None  

        # ✅ Manejo seguro del build_id
        # try:
        #     if build_id is not None and str(build_id).isdigit():
        #         prev_build_id = str(int(build_id) - 1)
        #         prev_model_path = os.path.join(MODELS_DIR, tester_id or "general", prev_build_id, "model.pkl")
        # except Exception:
        #     prev_model_path = None

        try:
            if build_id and str(build_id).isdigit():
                prev_build_id = str(int(build_id) - 1)
                prev_model_path = os.path.join(app_dir, tester_id, prev_build_id, "model.pkl")
        except Exception:
            prev_model_path = None    

        # ✅ Si no existe, usa el modelo general como base
        # if not prev_model_path or not os.path.exists(prev_model_path):
        #     general_model_path = os.path.join(app_dir, "general", "model.pkl")
        #     if os.path.exists(general_model_path):
        #         prev_model_path = general_model_path
        #         logger.info(f"[train_hybrid] Usando modelo general como base → {general_model_path}")

                # Cargar modelo previo si existe
        if prev_model_path and os.path.exists(prev_model_path):
            try:
                prev = joblib.load(prev_model_path)
                prev_kmeans = prev.get("kmeans")
                prev_hmm = prev.get("hmm")
                logger.info(f"[train_hybrid] Modelo previo encontrado → {prev_model_path}")
            except Exception as e:
                logger.warning(f"[train_hybrid] No se pudo cargar modelo previo: {e}")        

        if prev_build_id:
            prev_model_path = os.path.join(MODELS_DIR, tester_id or "general", prev_build_id, "model.pkl")
        else:
            prev_model_path = None


        if prev_model_path and os.path.exists(prev_model_path):
            try:
                prev = joblib.load(prev_model_path)
                prev_kmeans = prev.get("kmeans")
                prev_hmm = prev.get("hmm")
                logger.info(f"[train_hybrid] Modelo previo encontrado → {prev_model_path}")
            except Exception as e:
                logger.warning(f"[train_hybrid] No se pudo cargar modelo previo: {e}")

        # ===================== Entrenar KMeans =====================
        try:
            if prev_kmeans:
                kmeans = MiniBatchKMeans(
                    n_clusters=min(max_clusters, len(X)),
                    random_state=42,
                    init=prev_kmeans.cluster_centers_,
                    n_init=1
                ).fit(X)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=min(max_clusters, len(X)),
                    random_state=42
                ).fit(X)
        except Exception as e:
            logger.error(f"[train_hybrid] Error en KMeans: {e}")
            kmeans = BASE_KMEANS.fit(X)

        # ===================== Entrenar HMM =====================
        try:
            hmm_model = hmm.GaussianHMM(
                n_components=min(n_hmm_states, len(X)),
                covariance_type="diag",
                n_iter=300,
                tol=1e-3,
                random_state=42,
                verbose=False
            )

            if prev_hmm:
                hmm_model.startprob_ = prev_hmm.startprob_
                hmm_model.transmat_ = prev_hmm.transmat_
                hmm_model.means_ = prev_hmm.means_
                hmm_model.covars_ = prev_hmm.covars_

            hmm_model.fit(X, [len(X)])
        except Exception as e:
            logger.error(f"[train_hybrid] Error en HMM: {e}")
            hmm_model = BASE_HMM.fit(X)

        # ===================== Guardar modelos =====================
        try:
            joblib.dump({"kmeans": kmeans, "hmm": hmm_model}, os.path.join(tester_dir, "model.pkl"))
            joblib.dump(kmeans, os.path.join(tester_dir, "kmeans.joblib"))
            joblib.dump(hmm_model, os.path.join(tester_dir, "hmm.joblib"))

            # ✅ Actualizar el modelo general de la app si el tester no es "general"
            if tester_id != "general":
                general_dir = os.path.join(app_dir, "general")
                os.makedirs(general_dir, exist_ok=True)
                joblib.dump({"kmeans": kmeans, "hmm": hmm_model}, os.path.join(general_dir, "model.pkl"))
                logger.info(f"[train_hybrid] 🔄 Actualizado modelo general de {app_name}")

            logger.info(f"[train_hybrid] ✅ Modelos guardados correctamente en {tester_dir}")
        except Exception as e:
            logger.error(f"[train_hybrid] Error guardando modelos: {e}")
            
# =========================================================
# ENTRENAMIENTO HÍBRIDO (KMeans + HMM)  – versión mejorada
# =========================================================


def normalize_header(text: str) -> str:
    if not text:
        return ""
    # Quitar acentos y normalizar unicode
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # Minúsculas, strip y colapsar espacios
    return re.sub(r"\s+", " ", text.strip().lower())


def ensure_model_dimensions(kmeans, X, tester_id, build_id, app_name="default_app", desc=""):
    try:
        expected_features = kmeans.cluster_centers_.shape[1]
        current_features = X.shape[1]

        if current_features != expected_features:
            logger.warning(
                f"[{desc}] Dimensión inconsistente: modelo={expected_features}, nuevo={current_features}. Reentrenando..."
            )

            # ⚙️ Forzar reentrenamiento asincrónico
            lock_key = f"{app_name}:{tester_id}:{build_id}"
            lock = _get_lock(lock_key)

            asyncio.create_task(
                _train_model_hybrid(
                    X,
                    tester_id=tester_id,
                    build_id=build_id,
                    app_name=app_name,
                    lock=lock,
                    desc=f"retrain {desc}"
                )
            )

            return False

        return True

    except Exception as e:
        logger.warning(f"[{desc}] No se pudo validar dimensiones del modelo ({app_name}/{tester_id}/{build_id}): {e}")
        return False

def structure_signature_features(tree):
    """
    Extrae características estructurales de una jerarquía de nodos de UI.
    Compatible con vistas Android nativas y frameworks híbridos (Flutter, RN, etc.)
    """
    # Inicialización de contadores
    features = {
        # --- Android Clásico ---
        "Button": 0,
        "MaterialButton": 0,
        "ImageButton": 0,
        "EditText": 0,
        "TextView": 0,
        "ImageView": 0,
        "CheckBox": 0,
        "RadioButton": 0,
        "Switch": 0,
        "Spinner": 0,
        "SeekBar": 0,
        "ProgressBar": 0,
        "RecyclerView": 0,
        "ListView": 0,
        "ScrollView": 0,
        "LinearLayout": 0,
        "RelativeLayout": 0,
        "ConstraintLayout": 0,
        "FrameLayout": 0,
        "CardView": 0,

        # --- Jetpack Compose ---
        "ComposeView": 0,
        "Text": 0,           # usado por Compose
        "ButtonComposable": 0,

        # --- Híbridos (React Native, Flutter, Ionic, WebView) ---
        "RCTView": 0,        # React Native View
        "RCTText": 0,        # React Native Text
        "RCTImageView": 0,
        "FlutterView": 0,
        "WebView": 0,
        "IonContent": 0,
        "IonButton": 0,
        "IonInput": 0,
    }

    max_depth = 0
    total_nodes = len(tree)

    for node in tree:
        class_name = node.get("className", "") or ""
        depth = node.get("depth", 0)
        max_depth = max(max_depth, depth)

        for key in features.keys():
            if key.lower() in class_name.lower():
                features[key] += 1

    # Calcular métricas agregadas útiles
    interactive_elements = (
        features["Button"] + features["MaterialButton"] + features["EditText"] +
        features["CheckBox"] + features["RadioButton"] + features["Switch"] +
        features["Spinner"] + features["SeekBar"] + features["IonButton"]
    )

    media_elements = features["ImageView"] + features["RCTImageView"] + features["WebView"]

    layout_complexity = (
        features["LinearLayout"] + features["RelativeLayout"] +
        features["ConstraintLayout"] + features["FrameLayout"]
    )

    # Empaquetar en vector (mantén orden fijo)
    return [
        total_nodes,             # total de nodos en pantalla
        max_depth,               # profundidad máxima
        interactive_elements,    # elementos interactivos
        media_elements,          # componentes visuales
        layout_complexity,       # cantidad de layouts estructurales
        features["RecyclerView"], 
        features["ScrollView"],
        features["ComposeView"],
        features["FlutterView"],
        features["IonContent"],
    ]


    

async def analyze_and_train(event: AccessibilityEvent):
    # -------------------- Normalizar campos --------------------
    norm = _normalize_event_fields(event)
    t_id, b_id = norm.get("tester_id_norm"), norm.get("build_id_norm")
    s_name = normalize_header(event.header_text)

    # 🧩 NUEVO: obtener app_name (por package_name o dominio)
# 🧩 NUEVO: obtener app_name (por package_name o dominio)
    app_name = event.package_name or "default_app"

    tester_id = event.tester_id or "general"
    build_id = event.build_id

    # -------------------- Árbol y firma ------------------------
    latest_tree = ensure_list(event.collect_node_tree or event.tree_data or [])
    sig = stable_signature(latest_tree)


    # -------------------- Features enriquecidas ----------------
    struct_vec = np.array(ui_structure_features(latest_tree), dtype=float).flatten()
    sig_vec = np.array(structure_signature_features(latest_tree), dtype=float).flatten()

    timestamps = [e.timestamp for e in ensure_list(event.actions or [])]
    time_deltas = np.diff(timestamps) if len(timestamps) > 1 else [0]
    avg_dwell = float(np.mean(time_deltas)) if len(time_deltas) > 0 else 0
    num_gestos = sum(1 for e in event.actions or [] if e.type in ["tap", "scroll"])
    input_vec = np.array(input_features(event.actions or []), dtype=float).flatten()

    # ✅ Concatenación segura
    enriched_vector = np.concatenate([
        struct_vec,
        sig_vec,
        np.array([avg_dwell, num_gestos], dtype=float),
        input_vec
    ])

    # -------------------- Obtener builds previas ----------------
    with sqlite3.connect(DB_NAME) as conn:
        prev_rows = conn.execute("""
            SELECT collect_node_tree, signature, enriched_vector, build_id
            FROM accessibility_data
            WHERE TRIM(tester_id)=TRIM(?)
            ORDER BY created_at DESC
            LIMIT 5
        """, (t_id,)).fetchall()

    # -------------------- Comparación de árboles ----------------
    removed_all, added_all, modified_all = [], [], []



    for prev in prev_rows:
        collect_json, prev_sig, prev_vec_json, prev_build = prev
        prev_tree = ensure_list(json.loads(collect_json))

        # 🔹 Log: qué se compara con qué
        print(f"\n--- Comparación con build previa {prev_build} ---")
        print(f"Prev Build ID: {prev_build}")
        print(f"Prev Signature: {prev_sig}")
        print(f"Prev Tree nodes: {len(prev_tree)}")
        print(f"Latest Build ID: {b_id}")
        print(f"Latest Signature: {sig}")
        print(f"Latest Tree nodes: {len(latest_tree)}")


        removed, added, modified = compare_trees(prev_tree, latest_tree)
        removed_all.extend(removed)
        added_all.extend(added)
        modified_all.extend(modified)
 

    # Convertir cambios a JSON ordenado para comparar/inserción
    removed_j = json.dumps(removed_all, sort_keys=True)
    added_j = json.dumps(added_all, sort_keys=True)
    modified_j = json.dumps(modified_all, sort_keys=True)

    # -------------------- Insertar en screen_diffs (restaurado) ----------------
    try:
        with sqlite3.connect(DB_NAME) as conn:
            if not conn.execute("""
                SELECT 1 FROM screen_diffs
                WHERE IFNULL(header_text,'')=IFNULL(?, '')
                  AND removed=? AND added=? AND modified=?
                LIMIT 1
            """, (s_name, removed_j, added_j, modified_j)).fetchone():
                conn.execute("""
                    INSERT INTO screen_diffs (tester_id, build_id, header_text, removed, added, modified)
                    VALUES (?,?,?,?,?,?)
                """, (t_id, b_id, s_name, removed_j, added_j, modified_j))
                conn.commit()
    except Exception as e:
        print(f"⚠️ Error insertando en screen_diffs: {e}")

    # -------------------- Calcular anomaly_score HMM ----------------
    cluster_id, anomaly_score = None, None
    kmeans_model = KMEANS_MODELS.get(t_id)
    hmm_model = HMM_MODELS.get(t_id)



    # 🧩 NUEVO: cargar desde disco si no está en memoria
    #model_dir = os.path.join("models", t_id, str(b_id))
    model_dir = os.path.join("models", t_id or "general", str(b_id or "latest"))

    if not os.path.exists(model_dir):
        model_dir = os.path.join("models", "general", "default")

    if not kmeans_model and os.path.exists(os.path.join(model_dir, "kmeans.joblib")):
        try:
            kmeans_model = joblib.load(os.path.join(model_dir, "kmeans.joblib"))
            KMEANS_MODELS[t_id] = kmeans_model
            print(f"✅ Cargado KMeans para tester {t_id}")
        except Exception as e:
            print(f"⚠️ Error cargando kmeans.joblib: {e}")

    if not hmm_model and os.path.exists(os.path.join(model_dir, "hmm.joblib")):
        try:
            hmm_model = joblib.load(os.path.join(model_dir, "hmm.joblib"))
            HMM_MODELS[t_id] = hmm_model
            print(f"✅ Cargado HMM para tester {t_id}")
        except Exception as e:
            print(f"⚠️ Error cargando hmm.joblib: {e}")        

    if not hmm_model and os.path.exists(os.path.join(model_dir, "hmm.joblib")):
        try:
            hmm_model = joblib.load(os.path.join(model_dir, "hmm.joblib"))
            HMM_MODELS[t_id] = hmm_model
            print(f"✅ Cargado HMM para tester {t_id}")
        except Exception as e:
            print(f"⚠️ Error cargando hmm.joblib: {e}")
    
    if kmeans_model and hmm_model:
        try:
            # Verificar dimensiones
            if not ensure_model_dimensions(
                kmeans_model,
                enriched_vector.reshape(1, -1),
                t_id,
                b_id,
                desc="anomaly_score",
            ):
                print("⚠️ Modelo desactualizado — se omite esta predicción (reentrenamiento en curso)")
                return  # No guardar None en BD

            # ---------- Predicción de cluster ----------
            try:
                cluster_id = int(kmeans_model.predict(enriched_vector.reshape(1, -1))[0])
            except Exception as e:
                print(f"⚠️ Error prediciendo cluster_id: {e}")
                cluster_id = -1

            # ---------- Secuencia de acciones ----------
            seq = [e.type for e in ensure_list(event.actions or []) if e.type in ["tap", "scroll", "input"]]
            if not seq:
                seq = ["idle"]

            critical_nodes = sum(1 for n in latest_tree if n.get("className") in ["Button", "EditText"])
            seq.append(f"critical_{critical_nodes}")

            unique_states = {s: i for i, s in enumerate(set(seq))}
            encoded_seq = np.array([unique_states[s] for s in seq]).reshape(-1, 1)

            # ---------- Calcular logp ----------
            try:
                logp = hmm_model.score(encoded_seq)
                anomaly_score = max(0.0, float(-logp))
            except Exception as e:
                print(f"⚠️ Error en HMM.score(): {e}")
                anomaly_score = 0.0  # valor seguro

            print(f"[DEBUG] cluster_id={cluster_id}, anomaly_score={anomaly_score}")

        except Exception as e:
            print("⚠️ Error calculando anomaly_score:", e)
            anomaly_score = 0.0
            cluster_id = -1
    else:
        print(f"⚠️ No se encontraron modelos cargados para {t_id} (kmeans={kmeans_model}, hmm={hmm_model})")

    if (
        not removed_all and not added_all and not modified_all and
        anomaly_score is not None and anomaly_score < 0.5
    ):
        _insert_diff_trace(t_id, b_id, s_name, "Pantalla conocida: sin cambios ni anomalías")
        return  # 🔹 No reentrenar ni insertar más pilas eliminar si no funciona

    # -------------------- Insertar diff_trace ----------------
    def is_relevant_change(removed, added, modified, anomaly_score, threshold=0.5):
        num_changes = len(removed) + len(added) + len(modified)
        return num_changes > 0 or (anomaly_score is not None and anomaly_score > threshold)

    if is_relevant_change(removed_all, added_all, modified_all, anomaly_score):
        _insert_diff_trace(
            t_id, b_id, s_name,
            f"Removed={len(removed_all)}, Added={len(added_all)}, "
            f"Modified={len(modified_all)}, anomaly_score={anomaly_score if anomaly_score is not None else 'N/A'}"
        )
    else:
        _insert_diff_trace(t_id, b_id, s_name, "No hay cambios")

    # -------------------- Guardar vector enriquecido y cluster ----------------
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            UPDATE accessibility_data
            SET enriched_vector=?, cluster_id=?, anomaly_score=?
            WHERE TRIM(LOWER(header_text)) LIKE '%' || TRIM(LOWER(?)) || '%'
              AND TRIM(tester_id)=TRIM(?)
        """, (
            json.dumps(enriched_vector.tolist()),
            int(cluster_id) if cluster_id is not None else None,
            float(anomaly_score) if anomaly_score is not None else None,
            s_name,
            t_id
        ))
        conn.commit()

    # -------------------- Entrenamiento híbrido ----------------
    # asyncio.create_task(_train_incremental_logic_hybrid(t_id, b_id, enriched_vector=enriched_vector))
    # #await _train_incremental_logic_hybrid(t_id, b_id, enriched_vector=enriched_vector)
    # if TRAIN_GENERAL_ON_COLLECT:
    #     await _train_general_logic_hybrid(enriched_vector=enriched_vector)

    # ✅ Pasamos también app_name
    asyncio.create_task(
        _train_model_hybrid(
            X=np.array([enriched_vector]),
            tester_id=tester_id,
            build_id=build_id,
            app_name=app_name,  # 👈 NUEVO
            desc=f"{app_name} incremental"
        )
    )

    # ✅ Entrenar también el modelo general de la app
    if TRAIN_GENERAL_ON_COLLECT:
        await _train_model_hybrid(
            X=np.array([enriched_vector]),
            tester_id="general",
            build_id="latest",
            app_name=app_name,  # 👈 NUEVO
            desc=f"{app_name} general"
        )    

async def _train_incremental_logic_hybrid(
    tester_id: str,
    build_id: str,
    batch_size=200,
    min_samples=2,
    enriched_vector=None
):
    model_dir = os.path.join("models", tester_id, str(build_id or "latest"))
    os.makedirs(model_dir, exist_ok=True)

    # Cargar vectores históricos
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("""
            SELECT enriched_vector FROM accessibility_data
            WHERE tester_id=? AND build_id=?
            AND enriched_vector IS NOT NULL
        """, (tester_id, build_id)).fetchall()

    vectors_db = [json.loads(r[0]) for r in rows if r[0]]
    vectors_db = np.unique(vectors_db, axis=0) if len(vectors_db) > 0 else np.empty((0, 0))

    print(f"[DEBUG] Cantidad de vectores DB (incremental): {len(vectors_db)}")
    print(f"[DEBUG] Vectores únicos DB (incremental): {len(vectors_db)}")

    if len(vectors_db) < 5:
        print(f"⚠️ Muy pocos datos ({len(vectors_db)}) — se omite entrenamiento incremental.")
        return

    kmeans_model = MiniBatchKMeans(
        n_clusters=min(5, len(vectors_db)),
        random_state=42,
        n_init="auto",
        batch_size=batch_size
    )
    kmeans_model.fit(vectors_db)

    joblib.dump(kmeans_model, os.path.join(model_dir, "kmeans.joblib"))

    n_components = max(2, min(5, len(vectors_db) // 10))
    try:
        hmm_model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=200,
            tol=1e-3,
            verbose=False
        )
        hmm_model.fit(vectors_db)
        joblib.dump(hmm_model, os.path.join(model_dir, "hmm.joblib"))
    except Exception as e:
        print(f"⚠️ Error entrenando HMM: {e}")
        hmm_model = None

    KMEANS_MODELS[tester_id] = kmeans_model
    HMM_MODELS[tester_id] = hmm_model

    print(f"✅ Modelos guardados correctamente en {model_dir}")




async def _train_general_logic_hybrid(
    batch_size=1000,
    min_samples=2,
    enriched_vector: np.ndarray | None = None
):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT collect_node_tree
            FROM accessibility_data
            WHERE collect_node_tree IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (batch_size,))
        rows = c.fetchall()

        if len(rows) < min_samples:
            need = min_samples - len(rows)
            c.execute("""
                SELECT collect_node_tree
                FROM accessibility_data
                WHERE collect_node_tree IS NOT NULL
                ORDER BY created_at ASC
                LIMIT ?
            """, (need,))
            rows = c.fetchall() + rows

    X_db = features_from_rows(rows)

    if X_db.size > 0:
        print(f"[DEBUG] Cantidad de vectores DB: {X_db.shape[0]}")
        print(f"[DEBUG] Vectores únicos DB: {len(np.unique(X_db, axis=0))}")

        # ===================== NORMALIZACIÓN DE VECTORES =====================
        def normalize_vector_length(vec, expected_len):
            vec = np.array(vec, dtype=float).flatten()
            if len(vec) < expected_len:
                vec = np.pad(vec, (0, expected_len - len(vec)))  # rellena con ceros
            elif len(vec) > expected_len:
                vec = vec[:expected_len]  # recorta si es más largo
            return vec

        # Si no llega enriched_vector, crear uno neutro
        if enriched_vector is None:
            print("⚠️ 'enriched_vector' no se pasó correctamente. Se usará vector neutro.")
            enriched_vector = np.zeros(X_db.shape[1])

        # Calcula longitud esperada según el mayor vector
        EXPECTED_VECTOR_LEN = max(
            [len(enriched_vector)] +
            [len(v) for v in X_db if isinstance(v, (list, np.ndarray))]
        )

        # Normaliza todos los vectores
        enriched_vector = normalize_vector_length(enriched_vector, EXPECTED_VECTOR_LEN)
        X_db = np.array([
            normalize_vector_length(v, EXPECTED_VECTOR_LEN)
            for v in X_db
        ])

        logger.debug(f"[TRAIN] Longitud esperada de vector: {EXPECTED_VECTOR_LEN}")
        logger.debug(f"[TRAIN] enriched_vector shape: {enriched_vector.shape}, X_db shape: {X_db.shape}")

        # ===================== COMBINAR VECTORES =====================
        if enriched_vector is not None and not np.all(enriched_vector == 0):
            X = np.vstack([enriched_vector.reshape(1, -1), X_db])
        else:
            X = X_db

        # Elimina duplicados
        X = np.unique(X, axis=0)

        if len(X) < min_samples:
            print(f"[DEBUG] No hay suficientes muestras para entrenar: {len(X)} < {min_samples}")
            return

        # ===================== ENTRENAMIENTO HÍBRIDO =====================
        await _train_model_hybrid(
            X=X,
            tester_id="general",
            build_id="latest",
            app_name="default_app",                # o el app real si lo tienes
            lock=_get_lock("general:global:latest"),
            max_clusters=5,
            desc="general"
        )

        # ✅ Limpieza de caché
        KMEANS_MODELS.pop("general", None)
        HMM_MODELS.pop("general", None)
    else:
        print("[DEBUG] No hay datos suficientes en la base para entrenar.")


# =========================================================
# A partir de aquí se mantienen tus endpoints y lógica
# collect, checkForChanges, status, etc.
# Solo hay que reemplazar llamadas a
# _train_incremental_logic y _train_general_logic
# por sus versiones híbridas
# =========================================================

# =========================================================
# UTILIDADES PARA NORMALIZAR CAMPOS
        
def _insert_diff_trace(tester_id, build_id, screen, message):
    # Normalizar el screen_name
    screen_normalized = normalize_header(screen)

    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        # Verificar si ya existe exactamente el mismo registro
        exists = c.execute("""
            SELECT 1 FROM diff_trace
            WHERE tester_id=? AND build_id=? AND screen_name=? AND message=?
            LIMIT 1
        """, (tester_id, build_id, screen_normalized, message)).fetchone()

        # Insertar solo si no existe
        if not exists:
            c.execute("""
                INSERT INTO diff_trace (tester_id, build_id, screen_name, message)
                VALUES (?, ?, ?, ?)
            """, (tester_id, build_id, screen_normalized, message))
            conn.commit()        

def update_diff_trace(tester_id: str, build_id: str, screen: str, changes: List[str]) -> None:
    """
    Actualiza la tabla diff_trace:
      - Si hay cambios, borra mensajes 'No hay cambios' y agrega cada cambio.
      - Si no hay cambios, asegura que quede un único registro 'No hay cambios'.
    """
    screen_normalized = normalize_header(screen)

    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        if changes:
            # eliminar registros "No hay cambios" para ese tester/pantalla/build
            c.execute("""
                DELETE FROM diff_trace 
                WHERE tester_id=? AND build_id=? AND screen_name=? 
                AND message='No hay cambios'
            """, (tester_id, build_id, screen_normalized))
            for ch in changes:
                _insert_diff_trace(tester_id, build_id, screen_normalized, ch)
        else:
            # borrar otros mensajes y dejar solo "No hay cambios"
            c.execute("""
                DELETE FROM diff_trace 
                WHERE tester_id=? AND build_id=? AND screen_name=? 
                AND message <> 'No hay cambios'
            """, (tester_id, build_id, screen_normalized))
            c.execute("""
                INSERT INTO diff_trace (tester_id, build_id, screen_name, message)
                SELECT ?, ?, ?, 'No hay cambios'
                WHERE NOT EXISTS (
                    SELECT 1 FROM diff_trace 
                    WHERE tester_id=? AND build_id=? AND screen_name=? 
                    AND message='No hay cambios'
                )
            """, (tester_id, build_id, screen_normalized, tester_id, build_id, screen_normalized))
        conn.commit()


def last_hash_for_screen(tester_id: Optional[str],
                         screen_name: Optional[str]) -> Optional[str]:
    """
    Devuelve el último screens_id guardado para un tester/pantalla.
    Útil para saber si la pantalla ya fue procesada.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT screens_id
            FROM accessibility_data
            WHERE IFNULL(tester_id,'') = IFNULL(?, '')
              AND IFNULL(screen_names,'') = IFNULL(?, '')
            ORDER BY created_at DESC
            LIMIT 1
        """, (tester_id or "", screen_name or ""))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()
        
        
# =========================================================
def _normalize_event_fields(event: AccessibilityEvent) -> dict:
    """
    Devuelve un diccionario normalizado con nombres uniformes y
    alias comunes (snake_case) para su uso en modelos de ML.
    """
    raw = event.dict(by_alias=True, exclude_unset=True)
    tester_id = raw.get("testerId") or event.tester_id
    build_id  = raw.get("buildId")  or event.build_id

    return {
        **raw,
        "tester_id_norm": tester_id.strip() if tester_id else None,
        "build_id_norm":  build_id.strip()  if build_id  else None,
    }          

def normalize_node(node: Dict) -> Dict:
    """Filtra solo las claves estables y convierte None en cadena vacía."""
    return {k: (node.get(k) or "") for k in SAFE_KEYS}

def normalize_tree(nodes: List[Dict]) -> List[Dict]:
    """Normaliza y ordena la lista de nodos para que el orden no afecte el hash."""
    normalized = [normalize_node(n) for n in nodes]
    return sorted(normalized, key=lambda n: (n["className"], n["text"]))

def stable_signature(nodes: List[Dict]) -> str:
    """Genera un hash estable del árbol normalizado."""
    norm = normalize_tree(nodes)
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()   
    
# =========================================================
# ENDPOINTS API
# =========================================================
@app.post("/collect")
async def collect_event(event: AccessibilityEvent, background_tasks: BackgroundTasks):
    logger.debug("Raw request: %s", event.model_dump())
    try:
        # Normalizamos variantes de campos que pueden venir con distintos nombres
        raw_nodes = event.collect_node_tree or event.tree_data or []
        normalized_nodes = normalize_tree(raw_nodes)
        signature = stable_signature(raw_nodes)
        norm = _normalize_event_fields(event)
        tester_norm = norm.get("tester_id_norm")
        build_norm = norm.get("build_id_norm")
        screen_name = event.screen_names or ""
        screens_id_val = event.screens_id or norm.get("screensId") or None

        logger.info(f"[collect] normalized tester={tester_norm} build={build_norm} screen={screen_name} screens_id={screens_id_val}")

        # Evitar duplicados inmediatos: comparar último hash
        last = last_hash_for_screen(tester_norm, screen_name)
        logger.debug(f"[collect] last_hash={last} current_hash={screens_id_val}")

        # Si el hash actual es None (no enviado), igual insertamos el snapshot bruto,
        # pero si viene con screens_id y coincide con el último, podemos evitar insertar duplicado.
        do_insert = True
        if screens_id_val and last and str(last) == str(screens_id_val):
            do_insert = False
            logger.debug(f"[collect] last={last} ({type(last)}), screens_id_val={screens_id_val} ({type(screens_id_val)})")
            logger.info("[collect] Snapshot idéntico al último almacenado — no se inserta duplicado.")
        
        clean_header = (event.header_text or "").replace("\r", "").replace("\n", " ").strip()

        struct_vec = ui_structure_features(normalized_nodes)
        timestamps = [e.timestamp for e in ensure_list(event.actions or [])]
        time_deltas = np.diff(timestamps) if len(timestamps) > 1 else [0]
        avg_dwell = float(np.mean(time_deltas)) if len(time_deltas) > 0 else 0
        num_gestos = sum(1 for e in event.actions or [] if e.type in ["tap", "scroll"])
        input_vec = input_features(event.actions or [])

        # print("[DEBUG] normalized_nodes:", normalized_nodes)
        # print("[DEBUG] struct_vec:", struct_vec)
        # print("[DEBUG] timestamps:", timestamps)
        # print("[DEBUG] avg_dwell:", avg_dwell)
        # print("[DEBUG] num_gestos:", num_gestos)
        # print("[DEBUG] input_vec:", input_vec)

        #enriched_vector = np.array(struct_vec + [avg_dwell, num_gestos] + input_vec, dtype=float)

        sig_vec = np.array(structure_signature_features(normalized_nodes), dtype=float)
        input_vec = np.array(input_features(event.actions or []), dtype=float)
        combined = np.concatenate([
            np.array(struct_vec, dtype=float).flatten(),
            sig_vec.flatten(),
            np.array([avg_dwell, num_gestos], dtype=float),
            input_vec.flatten()
        ])
        enriched_vector = combined.astype(float)

        


        cluster_id: int | None = None
        anomaly_score: float | None = None

        if do_insert:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO accessibility_data (
                    tester_id, build_id, timestamp, event_type, event_type_name,
                    package_name, class_name, text, content_description, screens_id,
                    screen_names, header_text, collect_node_tree, signature,
                    additional_info, tree_data, enriched_vector, cluster_id, anomaly_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tester_norm, build_norm, event.timestamp, event.event_type,
                event.event_type_name, event.package_name, event.class_name,
                event.text, event.content_description, screens_id_val,
                event.screen_names, clean_header,
                json.dumps(normalized_nodes),
                signature,
                json.dumps(event.additional_info) if event.additional_info else None,
                json.dumps(event.tree_data) if event.tree_data else None,
                json.dumps(enriched_vector.tolist()) if enriched_vector is not None else None,
                cluster_id,
                anomaly_score
            ))
            conn.commit()
            conn.close()
            logger.info("[collect] Insert completado.")
        else:
            logger.info("[collect] Se omitió insert porque snapshot coincide con último.")

        # Añadir tarea en background para análisis/entrenamiento (siempre la lanzamos,
        # aunque no se haya insertado para mantener chequeos)
        background_tasks.add_task(analyze_and_train, event)
        return {"status": "success", "inserted": do_insert}
    except Exception as e:
        logger.error(f"Error en /collect: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/status")
async def get_status(
    testerId: Optional[str] = Query(None),
    buildId: Optional[str] = Query(None),
    screenName: Optional[str] = Query(None),
    limit: int = Query(5, ge=1, le=100)
):
    def safe_json(txt: str) -> list[Any]:
        try:
            return json.loads(txt) if txt else []
        except Exception:
            return []

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = """
        SELECT s.id,
               s.header_text,
               s.removed,
               s.added,
               s.modified,
               s.created_at
        FROM screen_diffs AS s
        LEFT JOIN diff_approvals AS a
               ON a.diff_id = s.id     
        WHERE a.id IS NULL
    """
    clauses, params = [], []

    if testerId:
        clauses.append("s.tester_id = ?")
        params.append(testerId)
    if buildId:
        clauses.append("s.build_id = ?")
        params.append(buildId)
    if screenName:
        clauses.append("s.header_text = ?")
        params.append(screenName)

    if clauses:
        query += " AND " + " AND ".join(clauses)

    query += " ORDER BY s.created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Construcción de diffs
    diffs = []
    for r in rows:
        removed = safe_json(r[2])   # ✅ columna correcta
        added = safe_json(r[3])     # ✅ columna correcta
        modified = safe_json(r[4])  # ✅ columna correcta
        if removed or added or modified:
            diffs.append({
                "id": r[0],
                "header_text": r[1],
                "removed": removed,
                "added": added,
                "modified": modified,
                "created_at": r[5],
            })

    if not diffs:
        return {}  # o Response(status_code=204)

    return {"status": "changes", "diffs": diffs}


@app.get("/train/general")
async def trigger_general_train(
    batch_size: int = Query(1000, ge=1),  # tamaño máximo de muestras para entrenar
    min_samples: int = Query(2, ge=1)     # mínimo de muestras para poder entrenar
):
    await _train_general_logic_hybrid(batch_size=batch_size, min_samples=min_samples)
    return {"status": "success", "message": "Entrenamiento general híbrido disparado"}


@app.get("/train/incremental")
async def trigger_incremental_train(
    tester_id: str = Query(...),
    build_id: str = Query(...),
    batch_size: int = Query(200, ge=1),
    min_samples: int = Query(2, ge=1)
):
    # ⚙️ Entrenamiento usando datos previos almacenados (sin enriched_vector directo)
    await _train_incremental_logic_hybrid(
        tester_id=tester_id,
        build_id=build_id,
        batch_size=batch_size,
        min_samples=min_samples,
        enriched_vector=None  # 👈 Añadir esto
    )
    return {
        "status": "success",
        "message": f"Entrenamiento incremental híbrido para {tester_id}/{build_id} disparado"
    }


# @app.get("/train/incremental")
# async def trigger_incremental_train(
#     tester_id: str = Query(...),
#     build_id: str = Query(...),
#     batch_size: int = Query(200, ge=1),
#     min_samples: int = Query(2, ge=1)
# ):
#     await _train_incremental_logic_hybrid(
#         tester_id=tester_id,
#         build_id=build_id,
#         batch_size=batch_size,
#         min_samples=min_samples
#     )
#     return {
#         "status": "success",
#         "message": f"Entrenamiento incremental híbrido para {tester_id}/{build_id} disparado"
#     }

@app.get("/screen/diffs")
def get_screen_diffs(
    tester_id: Optional[str] = Query(None),
    build_id: Optional[str] = Query(None),
    screen_name: Optional[str] = Query(None),
    only_pending: bool = Query(True)  # Nuevo parámetro opcional
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = """
        SELECT s.id, s.tester_id, s.build_id, s.screen_name, 
               s.removed, s.added, s.modified, s.created_at, s.cluster_info
        FROM screen_diffs AS s
        LEFT JOIN diff_approvals AS a
               ON a.diff_id = s.id
        WHERE 1=1
    """
    params = []

    if only_pending:
        query += " AND a.id IS NULL"  # Solo diffs sin aprobación

    if tester_id is not None:
        query += " AND (s.tester_id = ? OR (s.tester_id IS NULL AND ? = ''))"
        params.extend([tester_id, tester_id])

    if build_id is not None:
        query += " AND (s.build_id = ? OR (s.build_id IS NULL AND ? = ''))"
        params.extend([build_id, build_id])

    if screen_name is not None:
        query += " AND s.screen_name = ?"
        params.append(screen_name)

    # Solo registros que tengan cambios en removed, added o modified
    query += " AND (COALESCE(s.removed, '[]') != '[]' OR COALESCE(s.added, '[]') != '[]' OR COALESCE(s.modified, '[]') != '[]')"
    query += " ORDER BY s.created_at DESC"

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    diffs = []
    for row in rows:
        diffs.append({
            "id": row[0],
            "tester_id": row[1],
            "build_id": row[2],
            "screen_name": row[3],
            "removed": json.loads(row[4]) if row[4] else [],
            "added": json.loads(row[5]) if row[5] else [],
            "modified": json.loads(row[6]) if row[6] else [],
            "created_at": row[7],
            "cluster_info": json.loads(row[8]) if row[8] else {}
        })

    has_changes = bool(diffs)
    return {"screen_diffs": diffs, "has_changes": has_changes}



@app.get("/screen/exists")
async def screen_exists(buildId: str = Query(...)):
    """
    Devuelve {"exists": true/false} si hay al menos una fila
    en accessibility_data con build_id <= buildId.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1 FROM accessibility_data
        WHERE CAST(build_id AS INTEGER) <= CAST(? AS INTEGER)
        LIMIT 1
    """, (buildId,))

    row = cursor.fetchone()
    conn.close()

    return {"exists": bool(row)}
    
    
@app.post("/approve_diff")
async def approve_diff(request: Request):
    """
    Espera JSON: {"diff_id": <id>} o {"diff_id": "11"}.
    Registra la aprobación en DB y devuelve resultado.
    """
    # --- 1. Leer JSON de forma segura ---
    try:
        payload = await request.json()
    except ValueError:
        # El body estaba vacío o no era JSON válido
        raise HTTPException(status_code=400, detail="Cuerpo JSON inválido o vacío")

    # --- 2. Validar diff_id ---
    diff_id = payload.get("diff_id") or payload.get("id")
    if diff_id is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "diff_id missing"},
        )

    try:
        diff_id_int = int(diff_id)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "diff_id must be integer"},
        )

    # --- 3. Guardar en la base de datos ---
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diff_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diff_id INTEGER,
                approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO diff_approvals(diff_id) VALUES (?)", (diff_id_int,))
        conn.commit()
    except Exception as db_err:
        logger.exception("Error de base de datos en /approve_diff")
        raise HTTPException(status_code=500, detail=f"DB error: {db_err}")
    finally:
        conn.close()

    logger.info("Diff %s aprobado vía API", diff_id_int)
    return {"status": "success", "diff_id": diff_id_int}


@app.post("/reject_diff")
async def reject_diff(request: Request):
    """
    Espera JSON: {"diff_id": <id>} o {"diff_id": "11"}.
    Registra el rechazo en DB y devuelve resultado.
    """
    # --- 1. Leer JSON de forma segura ---
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Cuerpo JSON inválido o vacío")

    # --- 2. Validar diff_id ---
    diff_id = payload.get("diff_id") or payload.get("id")
    if diff_id is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "diff_id missing"},
        )

    try:
        diff_id_int = int(diff_id)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "diff_id must be integer"},
        )

    # --- 3. Guardar en la base de datos ---
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diff_rejections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diff_id INTEGER,
                rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO diff_rejections(diff_id) VALUES (?)", (diff_id_int,))
        conn.commit()
    except Exception as db_err:
        logger.exception("Error de base de datos en /reject_diff")
        raise HTTPException(status_code=500, detail=f"DB error: {db_err}")
    finally:
        conn.close()

    logger.info("Diff %s rechazado vía API", diff_id_int)
    return {"status": "success", "diff_id": diff_id_int}

@app.post("/cleanup_diffs")
async def cleanup_diffs(older_than_days: int = 90):
    """
    Borra diffs aprobados o rechazados anteriores a `older_than_days`.
    """
    from datetime import datetime, timedelta
    import sqlite3

    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Borrar diffs aprobados
    c.execute("""
        DELETE FROM screen_diffs
        WHERE id IN (
            SELECT s.id
            FROM screen_diffs s
            JOIN diff_approvals a ON a.diff_id = s.id
            WHERE s.created_at < ?
        )
    """, (cutoff_str,))

    # Borrar diffs rechazados
    c.execute("""
        DELETE FROM screen_diffs
        WHERE id IN (
            SELECT s.id
            FROM screen_diffs s
            JOIN diff_rejections r ON r.diff_id = s.id
            WHERE s.created_at < ?
        )
    """, (cutoff_str,))

    conn.commit()
    conn.close()
    return {"status": "success", "message": f"Difs anteriores a {cutoff_str} eliminados."}
    
@app.post("/cleanup_approvals_rejections")
async def cleanup_approvals_rejections(older_than_days: int = 90):
    from datetime import datetime, timedelta
    import sqlite3

    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("DELETE FROM diff_approvals WHERE created_at < ?", (cutoff_str,))
    c.execute("DELETE FROM diff_rejections WHERE created_at < ?", (cutoff_str,))

    conn.commit()
    conn.close()
    return {"status": "success", "message": f"Approvals y rejections anteriores a {cutoff_str} eliminados."}
    


@app.on_event("startup")
@repeat_every(seconds=86400)  # 1 día
def scheduled_cleanup() -> None:
    from datetime import datetime, timedelta
    import sqlite3

    older_than_days = 90
    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        DELETE FROM screen_diffs
        WHERE id IN (
            SELECT s.id
            FROM screen_diffs s
            JOIN diff_approvals a ON a.diff_id = s.id
            WHERE s.created_at < ?
        )
    """, (cutoff_str,))
    c.execute("""
        DELETE FROM screen_diffs
        WHERE id IN (
            SELECT s.id
            FROM screen_diffs s
            JOIN diff_rejections r ON r.diff_id = s.id
            WHERE s.created_at < ?
        )
    """, (cutoff_str,))
    c.execute("DELETE FROM diff_approvals WHERE created_at < ?", (cutoff_str,))
    c.execute("DELETE FROM diff_rejections WHERE created_at < ?", (cutoff_str,))
    conn.commit()
    conn.close()
    
    
   
@router.get("/reports/screen-changes")
def screen_changes(build_id: str, db=Depends(get_db)):
    rows = db.execute("""
        SELECT screen_name, removed, added, modified, created_at
        FROM screen_diffs
        WHERE build_id = ?
        ORDER BY created_at DESC
    """, (build_id,)).fetchall()

    return [
        {
            "screen_name": r["screen_name"],
            "removed": json.loads(r["removed"] or "[]"),
            "added": json.loads(r["added"] or "[]"),
            "modified": json.loads(r["modified"] or "[]"),
            "timestamp": r["created_at"]
        }
        for r in rows
    ]
    
@router.get("/reports/ui-stability")
def ui_stability(
    start_date: datetime,
    end_date: datetime,
    db=Depends(get_db)
):
    rows = db.execute("""
        SELECT screen_name,
               COUNT(*) as changes,
               COUNT(DISTINCT build_id) as builds_affected
        FROM screen_diffs
        WHERE created_at BETWEEN ? AND ?
        GROUP BY screen_name
        ORDER BY changes DESC
    """, (start_date, end_date)).fetchall()

    return [
        {
            "screen_name": r["screen_name"],
            "total_changes": r["changes"],
            "builds_affected": r["builds_affected"]
        }
        for r in rows
    ]   
    
@router.get("/reports/capture-coverage")
def capture_coverage(
    base_build: str,
    compare_build: str,
    db=Depends(get_db)
):
    base = db.execute("""
        SELECT DISTINCT screen_names
        FROM accessibility_data
        WHERE build_id = ?
    """, (base_build,)).fetchall()

    comp = db.execute("""
        SELECT DISTINCT screen_names
        FROM accessibility_data
        WHERE build_id = ?
    """, (compare_build,)).fetchall()

    base_screens = {r[0] for r in base}
    comp_screens = {r[0] for r in comp}

    return {
        "base_build": base_build,
        "compare_build": compare_build,
        "only_in_base": list(base_screens - comp_screens),
        "only_in_compare": list(comp_screens - base_screens),
        "common": list(base_screens & comp_screens)
    }    
    
app.include_router(router)
        
@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()
    tester_id = websocket.query_params.get("tester_id")
    build_id = websocket.query_params.get("build_id")
    while True:
        # Aquí enviar diffs desde tu DB o memoria
        await websocket.send_json({"tester_id": tester_id, "build_id": build_id, "diffs": []})
        await asyncio.sleep(5)        


@router.post("/send-reset-code")
def send_reset_code(req: ResetRequest):
    print(f"📩 Petición recibida desde Android: {req.email}")
    try:
        code = generate_code()
        codes_db[req.email] = {"code": code, "expires": time.time() + 300}  # 5 minutos
        
        
        save_reset_code(req.email, code)

        send_email(
            to_email=req.email,
            subject="Password Reset Code",
            body_text="Here is your verification code.",
            code=code
        )
        return {"message": "Email sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al enviar correo: {e}")


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest):
    # Validar código desde SQLite (sin eliminarlo)
    if not validate_code(req.email, req.code):
        raise HTTPException(status_code=400, detail="Código inválido o expirado")

    try:
        # Actualizar la contraseña real en tu DB principal
        print(f"🔑 Password for {req.email} updated to: {req.new_password}")
        # aquí iría tu lógica real de actualización en tu tabla de usuarios

        # Solo si la actualización es exitosa, eliminar el código
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            DELETE FROM password_reset_codes
            WHERE email = ? AND code = ?
        """, (req.email, req.code))
        conn.commit()
        conn.close()

        return {"success": True, "message": "Contraseña actualizada correctamente"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando contraseña: {e}")

@router.get("/qa_summary/{build_id}")
def qa_summary(build_id: str, db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute("""
        SELECT screen_name, message
        FROM diff_trace
        WHERE build_id=? 
        ORDER BY screen_name
    """, (build_id,)).fetchall()

    summary = {}
    for r in rows:
        screen = r["screen_name"]
        summary.setdefault(screen, []).append(r["message"])

    return JSONResponse(content=summary)


@app.get("/qa_summary/{build_id}")
def qa_summary(build_id: str, tester_id: Optional[str] = None):
    """
    Devuelve resumen de cambios por pantalla para QA.
    - Removed / Added / Modified
    - anomaly_score
    - cluster_id
    - histórico de builds previas (últimas 5)
    """
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()

        # Obtener los diffs por pantalla
        c.execute("""
            SELECT header_text, removed, added, modified, cluster_info, created_at
            FROM screen_diffs
            WHERE build_id = ?
        """, (build_id,))
        diffs = c.fetchall()

        summary = []
        for row in diffs:
            header_text, removed, added, modified, cluster_info, created_at = row
            summary.append({
                "screen": header_text,
                "removed": json.loads(removed) if removed else [],
                "added": json.loads(added) if added else [],
                "modified": json.loads(modified) if modified else [],
                "cluster_info": cluster_info,
                "timestamp": created_at
            })

        # Obtener anomaly_score y cluster_id de accessibility_data
        c.execute("""
            SELECT header_text, enriched_vector, cluster_id, anomaly_score
            FROM accessibility_data
            WHERE build_id = ?
            {}
        """.format("AND tester_id=?" if tester_id else ""), (build_id,) if not tester_id else (build_id, tester_id))
        vecs = c.fetchall()

        for row in vecs:
            header_text, enriched_vector, cluster_id, anomaly_score = row
            for s in summary:
                if normalize_header(s["screen"]) == normalize_header(header_text):
                    s.update({
                        "enriched_vector": json.loads(enriched_vector) if enriched_vector else None,
                        "cluster_id": cluster_id,
                        "anomaly_score": anomaly_score
                    })

    # Ordenar por anomaly_score descendente
    summary.sort(key=lambda x: x.get("anomaly_score") or 0, reverse=True)

    return JSONResponse(content={"build_id": build_id, "summary": summary})


@app.get("/qa_dashboard/{build_id}", response_class=HTMLResponse)
def qa_dashboard(build_id: str, tester_id: Optional[str] = None):
    """
    Dashboard ligero para QA:
    - Cambios por pantalla
    - Removed/Added/Modified
    - Anomaly score y cluster
    - Colores según criticidad
    """
    import json
    import requests

    # Llamar al endpoint de resumen interno
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get(f"/qa_summary/{build_id}" + (f"?tester_id={tester_id}" if tester_id else ""))
    summary = response.json()["summary"]

    # Generar HTML simple con colores
    html_content = """
    <html>
    <head>
        <title>QA Dashboard - Build {build_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            .removed {{ background-color: #f8d7da; }} 
            .added {{ background-color: #d4edda; }}
            .modified {{ background-color: #fff3cd; }}
            .anomaly-high {{ font-weight: bold; color: red; }}
        </style>
    </head>
    <body>
        <h2>QA Dashboard - Build {build_id}</h2>
        <table>
            <tr>
                <th>Screen</th>
                <th>Removed</th>
                <th>Added</th>
                <th>Modified</th>
                <th>Anomaly Score</th>
                <th>Cluster ID</th>
            </tr>
    """.format(build_id=build_id)

    for s in summary:
        removed_count = len(s.get("removed", []))
        added_count = len(s.get("added", []))
        modified_count = len(s.get("modified", []))
        anomaly_score = s.get("anomaly_score") or 0
        cluster_id = s.get("cluster_id") or "-"
        
        removed_cls = "removed" if removed_count > 0 else ""
        added_cls = "added" if added_count > 0 else ""
        modified_cls = "modified" if modified_count > 0 else ""
        anomaly_cls = "anomaly-high" if anomaly_score > 1 else ""  # ajustar umbral

        html_content += f"""
        <tr>
            <td>{s['screen']}</td>
            <td class="{removed_cls}">{removed_count}</td>
            <td class="{added_cls}">{added_count}</td>
            <td class="{modified_cls}">{modified_count}</td>
            <td class="{anomaly_cls}">{anomaly_score:.2f}</td>
            <td>{cluster_id}</td>
        </tr>
        """

    html_content += """
        </table>
        <p>Colores: <span class='removed'>Removed</span>, <span class='added'>Added</span>, <span class='modified'>Modified</span></p>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/qa_dashboard_advanced/{tester_id}", response_class=HTMLResponse)
def qa_dashboard_advanced(tester_id: str, builds: Optional[int] = 5):
    """
    Dashboard visual avanzado para QA:
    - Cambios por pantalla
    - Evolución de builds recientes
    - Anomaly score y cluster visualizados
    """
    import json
    import sqlite3

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Tomar las últimas N builds para el tester
    c.execute("""
        SELECT screen_diffs.build_id, screen_diffs.header_text, screen_diffs.removed, 
        screen_diffs.added, screen_diffs.modified, screen_diffs.anomaly_score, 
        screen_diffs.cluster_id
        FROM screen_diffs
        LEFT JOIN accessibility_data USING (tester_id, header_text)
        WHERE tester_id = ?
        ORDER BY screen_diffs.created_at DESC
        LIMIT ?
    """, (tester_id, builds*50))  # asume hasta 50 pantallas por build
    rows = c.fetchall()
    conn.close()

    # Preparar datos
    builds_dict = {}
    for r in rows:
        build_id = r[0]
        screen = r[1]
        removed = len(json.loads(r[2])) if r[2] else 0
        added = len(json.loads(r[3])) if r[3] else 0
        modified = len(json.loads(r[4])) if r[4] else 0
        anomaly_score = r[5] or 0
        cluster_id = r[6] or "-"
        if build_id not in builds_dict:
            builds_dict[build_id] = []
        builds_dict[build_id].append({
            "screen": screen,
            "removed": removed,
            "added": added,
            "modified": modified,
            "anomaly_score": anomaly_score,
            "cluster_id": cluster_id
        })

    builds_sorted = sorted(builds_dict.keys())  # orden cronológico

    # Generar HTML con Chart.js
    html_content = f"""
    <html>
    <head>
        <title>QA Dashboard Avanzado - Tester {tester_id}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            .removed {{ background-color: #f8d7da; }} 
            .added {{ background-color: #d4edda; }}
            .modified {{ background-color: #fff3cd; }}
            .anomaly-high {{ font-weight: bold; color: red; }}
        </style>
    </head>
    <body>
        <h2>QA Dashboard Avanzado - Tester {tester_id}</h2>
    """

    # Tabla por build
    for build_id in builds_sorted:
        html_content += f"<h3>Build: {build_id}</h3>"
        html_content += """
        <table>
            <tr>
                <th>Screen</th>
                <th>Removed</th>
                <th>Added</th>
                <th>Modified</th>
                <th>Anomaly Score</th>
                <th>Cluster ID</th>
            </tr>
        """
        for s in builds_dict[build_id]:
            removed_cls = "removed" if s["removed"] > 0 else ""
            added_cls = "added" if s["added"] > 0 else ""
            modified_cls = "modified" if s["modified"] > 0 else ""
            anomaly_cls = "anomaly-high" if s["anomaly_score"] > 1 else ""

            html_content += f"""
            <tr>
                <td>{s['screen']}</td>
                <td class="{removed_cls}">{s['removed']}</td>
                <td class="{added_cls}">{s['added']}</td>
                <td class="{modified_cls}">{s['modified']}</td>
                <td class="{anomaly_cls}">{s['anomaly_score']:.2f}</td>
                <td>{s['cluster_id']}</td>
            </tr>
            """
        html_content += "</table>"

    # Gráfico agregado: tendencia de Removed / Added / Modified por build
    removed_series = [sum(s["removed"] for s in builds_dict[b]) for b in builds_sorted]
    added_series = [sum(s["added"] for s in builds_dict[b]) for b in builds_sorted]
    modified_series = [sum(s["modified"] for s in builds_dict[b]) for b in builds_sorted]

    html_content += f"""
    <canvas id="trendChart" width="800" height="400"></canvas>
    <script>
        const ctx = document.getElementById('trendChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(builds_sorted)},
                datasets: [
                    {{ label: 'Removed', data: {removed_series}, borderColor: 'red', fill: false }},
                    {{ label: 'Added', data: {added_series}, borderColor: 'green', fill: false }},
                    {{ label: 'Modified', data: {modified_series}, borderColor: 'orange', fill: false }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'top' }},
                    title: {{ display: true, text: 'Cambios por Build' }}
                }}
            }}
        }});
    </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


app.include_router(diff_router)
app.include_router(router, prefix="/api")
