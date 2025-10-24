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
from PIL import Image
from collections import Counter
import math
from sklearn.cluster import KMeans 
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import httpx

from stable_signature import normalize_node

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

IGNORED_NODE_SUFFIXES = [
    "|enabled:True",
    "|focusable:True",
    "|clickable:True",
    "|checked:False",
    "|selected:False",
]

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
            is_stable INTEGER DEFAULT 0,
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
            diff_hash TEXT UNIQUE NOT NULL,
            text_diff TEXT,  
            cluster_id INTEGER DEFAULT -1,  
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
       CREATE TABLE IF NOT EXISTS ignored_changes_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tester_id TEXT,
            build_id INTEGER,
            header_text TEXT,
            signature TEXT,
            class_name TEXT,
            field TEXT,
            old_value TEXT,
            new_value TEXT,
            reason TEXT,
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
            approved INTEGER DEFAULT 0, 
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
SAFE_KEYS = [
    "viewId", "className", "headerText", "text", "contentDescription", "desc", "hint",
    "checked", "enabled", "focusable", "clickable", "selected", "scrollable",
    "password", "pressed", "activated", "visible",
    "progress", "max", "value", "rating", "level",
    "inputType", "orientation", "index", "layoutParams", "pkg",
    "textColor", "backgroundColor", "fontSize", "alpha"
]

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

# def normalize_node(node: Dict) -> Dict:
#     return {k: (node.get(k) or "") for k in SAFE_KEYS}

def normalize_tree(nodes: List[Dict]) -> List[Dict]:
    return sorted([normalize_node(n) for n in nodes if isinstance(n, dict)],
                  key=lambda n: (n["className"], n["text"]))

def stable_signature(nodes: List[Dict]) -> str:
    return hashlib.sha256(
        json.dumps(normalize_tree(nodes), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

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


 # =========================================================
# UTILIDADES COMPARE TREES
# =========================================================   
def estimate_text_color(bitmap):
    """
    Recibe un PIL Image (bitmap) del nodo y devuelve un color hexadecimal promedio.
    """
    if bitmap is None:
        return None
    
    # Reducir tamaño para acelerar cálculo (opcional)
    small = bitmap.resize((10, 10))
    
    # Convertir a RGB
    small = small.convert("RGB")
    
    # Obtener todos los pixeles
    pixels = list(small.getdata())
    
    # Calcular promedio de R, G, B
    r_avg = sum(p[0] for p in pixels) // len(pixels)
    g_avg = sum(p[1] for p in pixels) // len(pixels)
    b_avg = sum(p[2] for p in pixels) // len(pixels)
    
    # Convertir a hex
    return f"#{r_avg:02X}{g_avg:02X}{b_avg:02X}"

def preprocess_tree(tree):
    """
    Limpia, filtra y normaliza un árbol antes de la comparación.
    Ignora diferencias irrelevantes como cambios de contenedor o de jerarquía.
    """
    def _flatten(node_list):
        result = []
        for node in ensure_list(node_list):
            if not isinstance(node, dict):
                continue
            cls = node.get("className", "")
            text = (node.get("text") or node.get("desc") or "").strip()
            # Ignorar wrappers sin texto ni interactividad
            if cls in [
                "android.widget.FrameLayout",
                "android.view.View",
                "androidx.compose.ui.platform.ComposeView",
                "androidx.compose.ui.viewinterop.ViewFactoryHolder",
            ] and not text:
                # Desapila sus hijos
                children = node.get("children") or []
                result.extend(_flatten(children))
                continue

            # Normaliza valores vacíos
            norm_node = {
                "className": cls,
                "text": text.lower().strip(),
                "clickable": bool(node.get("clickable")),
                "focusable": bool(node.get("focusable")),
                "enabled": bool(node.get("enabled", True)),
                "pkg": node.get("pkg", ""),
            }
            result.append(norm_node)
        return result

    flat = _flatten(tree)

    # Ordenar los nodos por clase + texto, ignorando jerarquía
    flat.sort(key=lambda n: (n["className"], n["text"]))
    print("Flattened nodes:", flat)
    return flat




def compare_trees(old_tree, new_tree, app_name: str = None):
    """
    Compara dos árboles de nodos accesibles y detecta nodos agregados, eliminados o modificados.
    Ajustado para detectar cambios de texto y filtrar ruido de estados efímeros.
    """
    
    old_tree = ensure_list(old_tree)
    new_tree = ensure_list(new_tree)

    # 🔹 Normalización estructural para ignorar wrappers o contenedores irrelevantes
    old_tree = preprocess_tree(old_tree)
    new_tree = preprocess_tree(new_tree)

    if not old_tree:
        print("⚠️ No hay snapshot previo para comparar, se guarda como base inicial.")
        return {
            "removed": [],
            "added": [],
            "modified": [],
            "text_diff": {"removed_texts": [], "added_texts": []}
        }

    SAFE_KEYS = [
        "viewId", "className", "headerText", "text", "contentDescription", "desc", "hint",
        "checked", "enabled", "focusable", "clickable", "selected", "scrollable",
        "password", "pressed", "activated", "visible",  
        "progress", "max", "value", "rating", "level",
        "inputType", "orientation", "index", "layoutParams", "pkg",
        "textColor", "backgroundColor", "fontSize", "alpha"
    ]

    IGNORED_STATE_FIELDS = {
        "enabled", "focusable", "clickable", "scrollable", "pressed", "activated", "visible"
    }

    TEXT_FIELDS = ["text", "contentDescription", "headerText", "hint"]
    BOOL_FIELDS = [
        "checked", "enabled", "focusable", "clickable", "selected", "scrollable",
        "password", "pressed", "activated", "visible"
    ]
    NUM_FIELDS = ["progress", "max", "value", "rating", "level"]
    OTHER_FIELDS = ["className", "viewId", "pkg"]
    VISUAL_FIELDS = ["textColor", "backgroundColor", "fontSize", "alpha"]

    # ---------------- flatten ----------------
    def flatten_tree(tree):
        """Flatten tree generando paths relativos para cada nodo."""
        result = []

        def _walk(subtree, path=[]):
            if isinstance(subtree, dict):
                idx = path[-1] if path else 0
                result.append((list(path), subtree))  # Guardamos path completo
                for i, ch in enumerate(subtree.get("children") or []):
                    _walk(ch, path + [i])
            elif isinstance(subtree, list):
                for i, n in enumerate(subtree):
                    _walk(n, path + [i])

        _walk(tree, [])
        return result
    

    def subtree_hash(node):
        """
        Calcula un hash estable para un nodo y todo su subárbol.
        Útil para nodos sin viewId ni texto para detectar movimientos.
        """
        node_copy = dict(node)
        children = node_copy.pop("children", [])
        node_str = json.dumps(node_copy, sort_keys=True, ensure_ascii=False)
        hash_value = hashlib.md5(node_str.encode()).hexdigest()
        
        # Combina con hashes de los hijos
        for child in children or []:
            hash_value = hashlib.md5((hash_value + subtree_hash(child)).encode()).hexdigest()
        return hash_value[:20]   
    
    # Lista de campos efímeros que queremos ignorar
    EPHEMERAL_FIELDS = {"pressed", "focused", "activated", "selected", "visible"}

    # ---------------- normalize ----------------

    def normalize_node(node: dict) -> dict:
        normalized = {}
        for k in SAFE_KEYS:
            v = node.get(k)
            if k in BOOL_FIELDS:
                normalized[k] = bool(v) if v not in (None, "", "null") else False
            elif k in NUM_FIELDS:
                try:
                    normalized[k] = float(v)
                except (TypeError, ValueError):
                    normalized[k] = None
            elif v is None:
                normalized[k] = ""
            else:
                normalized[k] = str(v).strip()

        # 🔹 Filtrar campos efímeros automáticamente
        for ef in EPHEMERAL_FIELDS:
            if ef in normalized:
                # Podrías poner debug opcional aquí
                # print(f"🔧 IGNORANDO EFÍMERO → {ef}={normalized[ef]} | class={node.get('className')}")
                normalized.pop(ef)

        # 🔧 Debug opcional de nodo final normalizado
        if 'pressed' in node:
            print(f"🔧 NODE DEBUG → class={node.get('className')}, view_id={node.get('viewId')}, pressed={node.get('pressed')}")

        return normalized


    print("🧩 RAW NODE → class={cls}, view_id={view_id}, text={nn.get('text')!r}")

    # def make_base_key(n):
    #     cls = (n.get("className") or "").strip()
    #     pkg = (n.get("pkg") or "").strip()
    #     view_id = (n.get("viewId") or "").strip()

    #     if view_id:
    #         base_key = f"{pkg}|{cls}|{view_id}"
    #     else:
    #         base_key = f"{pkg}|{cls}|subtree:{subtree_hash(n)}"
    #     return base_key
    
    def make_base_key(n):
        cls = (n.get("className") or "").strip()
        pkg = (n.get("pkg") or "").strip()
        view_id = (n.get("viewId") or "").strip()
        text_like = (n.get("text") or "").strip()

        parts = cls.split(".")
        norm_class = parts[-1].lower() if parts else ""
        base_class = parts[-2].lower() if len(parts) > 1 else ""
        full_class_sig = f"{base_class}.{norm_class}" if base_class else norm_class

        if view_id:
            base_key = f"{pkg}|{full_class_sig}|{view_id}"
        elif text_like:
            base_key = f"{pkg}|{full_class_sig}|text:{text_like}"
        else:
            base_key = f"{pkg}|{full_class_sig}|subtree:{subtree_hash(n)}"

        return base_key

    def make_key(n, idx=None, for_qa=False):
        nn = normalize_node(n)
        cls = (nn.get("className") or "").strip()
        pkg = (nn.get("pkg") or "").strip()
        view_id = (nn.get("viewId") or "").strip()
        text_like = (
            nn.get("text")
            or nn.get("desc")
            or nn.get("contentDescription")
            or nn.get("hint")
            or ""
        ).strip()

        # --- Normalización de clases ---
        parts = cls.split(".")
        norm_class = parts[-1].lower() if parts else ""
        base_class = parts[-2].lower() if len(parts) > 1 else ""
        full_class_sig = f"{base_class}.{norm_class}" if base_class else norm_class

        #key = None  # 🔹 asegurar que existe
        # 🔹 Inicializamos siempre una clave base
        key = f"{pkg}|{full_class_sig}|subtree:{subtree_hash(n)}"
        # --- Generamos el texto-hash (si existe texto) ---
        text_sig = hashlib.md5(text_like.lower().encode()).hexdigest()[:8] if text_like else ""

        # # --- NUEVA LÓGICA ---
        # # Si el nodo tiene viewId Y texto → incluimos el hash de texto en la clave
        # if view_id and text_like:
        #     key = f"{pkg}|{full_class_sig}|{view_id}|textsig:{text_like}"
        # elif view_id:
        #     key = f"{pkg}|{full_class_sig}|{view_id}"
        # # elif text_like:
        # #     key = f"{pkg}|{full_class_sig}|textsig:{text_like}"
        # elif text_like:
        #     print(f"🔍 TEXT BEFORE HASH → class={cls}, text_like={text_like!r}")
        #     # text_sig = hashlib.md5(text_like.lower().encode()).hexdigest()[:20]
        #     safe_text = text_like.replace("\n", " ").strip()[:100]
        #     key = f"{pkg}|{full_class_sig}|textsig:{safe_text}"
        #     # key = f"{pkg}|{full_class_sig}|textnode"
        # --- NUEVA LÓGICA ---
        if view_id:
            # Clave estable basada en viewId y clase → no incluimos texto
            key = f"{pkg}|{full_class_sig}|{view_id}"
        elif text_like:
            # Solo nodos sin viewId, usamos hash de texto para identificar nodo de texto único
            safe_text = text_like.replace("\n", " ").strip()[:100]
            key = f"{pkg}|{full_class_sig}|textsig:{safe_text}"
        else:
            # Nodo sin viewId ni texto → hash del subárbol
            key = f"{pkg}|{full_class_sig}|subtree:{subtree_hash(n)}"

        # else:
        #     key = f"{pkg}|{full_class_sig}|subtree:{subtree_hash(n)}"

        # --- 1. Si tiene viewId ---
        # if view_id:
        #     key = f"{pkg}|{full_class_sig}|{view_id}"

        # --- 2. Si tiene texto visible ---
        # elif text_like:
        #     # ✅ Generamos hash solo para distinguir nodos de texto únicos
        #     text_sig = hashlib.md5(text_like.lower().encode()).hexdigest()[:8]
        #     safe_text = text_like.replace("\n", " ").strip()[:50]  # acortamos texto largo solo para debug
        #     key = f"{pkg}|{full_class_sig}|textsig:{text_sig}"
            # print(f"🧩 KEY DE TEXTO: {key} (texto='{safe_text}')")

        #elif text_like:
            #print(f"🔍 TEXT BEFORE HASH → class={cls}, text_like={text_like!r}")
            #text_sig = hashlib.md5(text_like.lower().encode()).hexdigest()[:20]
            # safe_text = text_like.replace("\n", " ").strip()[:100]
            # key = f"{pkg}|{full_class_sig}|textsig:{safe_text}"
            #key = f"{pkg}|{full_class_sig}|textnode"

        # --- 3. Sin viewId ni texto ---
        # else:
        #     key = f"{pkg}|{full_class_sig}|subtree:{subtree_hash(n)}"

        # --- 4. Estado relevante ---
        state_parts = []
        if norm_class in ["checkbox", "radiobutton", "switch"]:
            state_parts.append(f"checked:{nn.get('checked')}")
        if norm_class in ["ratingbar", "seekbar"]:
            if nn.get("rating") is not None:
                state_parts.append(f"rating:{nn.get('rating')}")
            if nn.get("progress") is not None:
                state_parts.append(f"progress:{nn.get('progress')}")
        if norm_class == "tablayout":
            state_parts.append(f"selected:{nn.get('selected')}")

        if state_parts:
            key += "|" + "|".join(state_parts)

        return key


    # ---------------- index trees ----------------
    # old_idx = {make_key(n, idx): normalize_node(n) for idx, n in flatten_tree(old_tree)}
    # new_idx = {make_key(n, idx): normalize_node(n) for idx, n in flatten_tree(new_tree)}



    # ---------------- index trees ----------------
    old_idx = {}
    for idx, n in flatten_tree(old_tree):
        key = make_key(n, idx)
        #old_idx[key] = normalize_node(n)
        key = make_key(n, idx)
        old_idx[key] = normalize_node(n)
        # if "button" in (n.get("className") or "").lower():
        #     print("🧩 OLD NODE →", n.get("className"), n.get("viewId"), n.get("text"), "| key:", key)

    new_idx = {}
    for idx, n in flatten_tree(new_tree):
        key = make_key(n, idx)
        new_idx[key] = normalize_node(n)
        # if "button" in (n.get("className") or "").lower():
        #     print("🧩 NEW NODE →", n.get("className"), n.get("viewId"), n.get("text"), "| key:", key)

        try:
            print("🧩 --- DETALLE DE CLAVES Y TEXTOS ---")
            if not new_idx:
                print("⚠️ new_idx está vacío, no se generaron claves nuevas.")
            elif not old_idx:
                print("⚠️ old_idx está vacío, no se generaron claves viejas.")
            else:
                diff_count = 0
                for k in new_idx:
                    old_text = old_idx.get(k, {}).get("text", "<no existe>")
                    new_text = new_idx[k].get("text", "<sin texto>")
                    if old_text != new_text:
                        diff_count += 1
                        print(f"🔸 Key: {k}")
                        print(f"   OLD TEXT: {old_text!r}")
                        print(f"   NEW TEXT: {new_text!r}")
                if diff_count == 0:
                    print("✅ No se detectaron diferencias de texto (todos los textos coinciden).")
        except Exception as e:
            print(f"⚠️ Error al imprimir diferencias de texto: {e}")
        # ---------------- debug: ver clases reales ----------------
    old_nodes = [n for _, n in flatten_tree(old_tree)]
    new_nodes = [n for _, n in flatten_tree(new_tree)]

    text_overlap = overlap_ratio(old_nodes, new_nodes)
    print(f"📊 Text overlap ratio: {text_overlap:.2f}")

    print("Old nodes:", old_nodes)
    print("New nodes:", new_nodes)

    # print("🧱 --- OLD CLASSES ---")
    # for n in old_nodes:
    #     cls_name = n.get("className") or ""
    #     if "button" in cls_name.lower():
    #         print("   OLD:", cls_name)

    # print("🧱 --- NEW CLASSES ---")
    # for n in new_nodes:
    #     cls_name = n.get("className") or ""
    #     if "button" in cls_name.lower():
    #         print("   NEW:", cls_name)


    # print("📋 --- OLD TREE KEYS ---")
    # for k in old_idx.keys():
    #     if "button" in k:
    #         print("   ", k)

    # print("📋 --- NEW TREE KEYS ---")
    # for k in new_idx.keys():
    #     if "button" in k:
    #         print("   ", k)


    # Calcular overlap textual
    # overlap = overlap_ratio(old_nodes, new_nodes)
    # logger.info(f"📊 Text overlap ratio: {overlap:.2f}")

    added = [{"node": {"key": k, "class": v.get("className")}, "changes": {}} for k, v in new_idx.items() if k not in old_idx]
    removed = [{"node": {"key": k, "class": v.get("className")}, "changes": {}} for k, v in old_idx.items() if k not in new_idx]

    modified, ignored_changes = [], []

    print("\n🔎 COMPARANDO CLAVES ENTRE OLD_IDX y NEW_IDX...")
    print(f"   🔹 Total old: {len(old_idx)} | Total new: {len(new_idx)}")

        # --- Detectar removidos ---
    for k, v in old_idx.items():
        if k not in new_idx:
            print(f"   ❌ REMOVIDO: {k} ({v.get('className')}, text='{v.get('text', '')[:40]}')")
            removed.append({"node": {"key": k, "class": v.get("className")}, "changes": {}})

    # --- Detectar agregados ---
    for k, v in new_idx.items():
        if k not in old_idx:
            print(f"   ➕ AGREGADO: {k} ({v.get('className')}, text='{v.get('text', '')[:40]}')")
            added.append({"node": {"key": k, "class": v.get("className")}, "changes": {}})


    print("🧩 --- DETALLE COMPLETO DE TEXTOS (OLD vs NEW) ---")
    for k in sorted(set(list(old_idx.keys()) + list(new_idx.keys()))):
        old_text = old_idx.get(k, {}).get("text", "<no existe>")
        new_text = new_idx.get(k, {}).get("text", "<no existe>")
        if old_text != new_text:
            print(f"🔸 DIF TEXT → Key: {k}")
            print(f"   OLD TEXT: {old_text!r}")
            print(f"   NEW TEXT: {new_text!r}")
        else:
            print(f"✅ IGUAL → {k} → '{new_text}'")

    # ---------------- detect modifications ----------------
    for k, nn in new_idx.items():
        if k in old_idx:
            oldn = old_idx[k]
            changes = {}

            for f in TEXT_FIELDS + BOOL_FIELDS + NUM_FIELDS + OTHER_FIELDS + VISUAL_FIELDS:
                old_val, new_val = oldn.get(f), nn.get(f)
                if old_val != new_val:
                    if f in IGNORED_STATE_FIELDS and old_val is False and new_val is True:
                        ignored_changes.append((nn.get("className"), f, (old_val, new_val)))
                    else:
                        changes[f] = {"old": old_val, "new": new_val}

            if changes:
                print(f"   ✏️ MODIFICADO: {k} ({nn.get('className')})")
                print("🔍 Cambio detectado en nodo:", nn.get("className"))
                for f, vals in changes.items():
                    print(f"      • {f}: '{vals['old']}' → '{vals['new']}'")
                    print(f"   • Campo: {f} | old='{vals['old']}' → new='{vals['new']}'")

                modified.append({
                    "node": {"key": k, "class": nn.get("className")},
                    "changes": changes
                })
    print(f"\n📊 RESUMEN → removed={len(removed)}, added={len(added)}, modified={len(modified)}")
    # ---------------- report ----------------
    if ignored_changes:
        print(f"🤖 Ignorados {len(ignored_changes)} cambios esperados:")
        for c in ignored_changes[:3]:
            print("   ", c)

    print(f"🧮 Raw compare_trees → removed={len(removed)}, added={len(added)}, modified={len(modified)}")


       # ---------------- filtro de recarga visual ----------------
    total_old = len(old_idx)
    total_new = len(new_idx)
    total_changed = len(removed) + len(added)

    reload_ratio = total_changed / max(total_old, total_new, 1)

    has_loader = any(
        "progressbar" in v.get("className", "").lower()
        or "skeleton" in v.get("className", "").lower()
        or "shimmer" in v.get("className", "").lower()
        for v in new_idx.values()
    )

    # if reload_ratio > 0.4 or has_loader:
    #     print(f"⚠️ Recarga visual detectada (ratio={reload_ratio:.2f}, loader={has_loader}) → limpiando added/removed")
    #     removed, added = [], []

    # ============================================================
    # 🧩 FILTRO 2: Pares estáticos (mismo texto en added/removed)
    # ============================================================
    
    def extract_text_from_key(key: str) -> str:
        """Extrae texto visible desde la clave, soportando text: o textsig:"""
        if "textsig:" in key:
            return key.split("textsig:")[-1].strip()
        if "text:" in key:
            return key.split("text:")[-1].strip()
        return ""

    # 1️⃣ Extrae los textos antes de filtrar
    old_texts_raw = {normalize_node(n).get("text", "") for _, n in flatten_tree(old_tree) if n.get("text")}
    new_texts_raw = {normalize_node(n).get("text", "") for _, n in flatten_tree(new_tree) if n.get("text")}

    common_texts = old_texts_raw & new_texts_raw
    total_texts = old_texts_raw | new_texts_raw
    text_overlap = len(common_texts) / max(len(total_texts), 1)
    print(f"📊 Text overlap ratio (raw, pre-filter): {text_overlap:.2f}")


    same_text_removed = {extract_text_from_key(r["node"]["key"]) for r in removed if extract_text_from_key(r["node"]["key"])}
    same_text_added = {extract_text_from_key(a["node"]["key"]) for a in added if extract_text_from_key(a["node"]["key"])}

    overlap_texts = same_text_removed & same_text_added
    
    if overlap_texts:
        print(f"🧩 Filtrando pares estáticos → {len(overlap_texts)} coincidencias de texto persistente")
        removed = [r for r in removed if not any(t in r["node"]["key"] for t in overlap_texts)]
        added = [a for a in added if not any(t in a["node"]["key"] for t in overlap_texts)]

        # ============================================================
    # 🔍 PASADA GLOBAL DE CAMBIOS DE TEXTO (Red de seguridad)
    # ============================================================
    try:
        old_texts = {normalize_node(n).get("text", "") for _, n in flatten_tree(old_tree) if n.get("text")}
        new_texts = {normalize_node(n).get("text", "") for _, n in flatten_tree(new_tree) if n.get("text")}

        diff_texts = new_texts - old_texts
        removed_texts = old_texts - new_texts

        if diff_texts or removed_texts:
            print("📝 CAMBIOS GLOBALES DE TEXTO DETECTADOS:")
            for t in removed_texts:
                print(f"   ❌ Texto removido: {t!r}")
            for t in diff_texts:
                print(f"   ➕ Texto agregado: {t!r}")
        else:
            print("✅ No se detectaron cambios globales de texto.")
    except Exception as e:
        print(f"⚠️ Error al detectar cambios globales de texto: {e}")

    # ============================================================
    # ✅ Resultado final
    # ============================================================


    # ============================================================
# 🧠 DETECCIÓN DE CAMBIO DE TEXTO EN EL MISMO NODO
# ============================================================
    try:
        for k, old_node in old_idx.items():
            if k in new_idx:
                new_node = new_idx[k]
                old_text = (old_node.get("text") or "").strip()
                new_text = (new_node.get("text") or "").strip()

                if old_text and new_text and old_text != new_text:
                    # Si ya existe un modified para esa key, lo actualizamos
                    existing = next((m for m in modified if m["node"]["key"] == k), None)
                    if existing:
                        existing["changes"]["text"] = {"old": old_text, "new": new_text}
                    else:
                        modified.append({
                            "node": {"key": k, "class": new_node.get("className")},
                            "changes": {"text": {"old": old_text, "new": new_text}}
                        })
                    print(f"✏️ Cambio de texto en nodo existente: {k}")
                    print(f"    '{old_text}' → '{new_text}'")
    except Exception as e:
        print(f"⚠️ Error en detección de texto en mismo nodo: {e}")

    #return removed, added, modified
    return {
        "removed": removed,
        "added": added,
        "modified": modified,
        "text_diff": {
            "removed_texts": list(removed_texts),
            "added_texts": list(diff_texts),
            "overlap_ratio": text_overlap,  # 👈 agregado
        }
    }


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

        try:
            if build_id and str(build_id).isdigit():
                prev_build_id = str(int(build_id) - 1)
                prev_model_path = os.path.join(app_dir, tester_id, prev_build_id, "model.pkl")
        except Exception:
            prev_model_path = None    

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
    Extrae características estructurales de una jerarquía de nodos de UI,
    incluyendo diversidad textual básica para detectar botones distintos.
    """
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
        "Text": 0,
        "ButtonComposable": 0,

        # --- Híbridos ---
        "RCTView": 0,
        "RCTText": 0,
        "RCTImageView": 0,
        "FlutterView": 0,
        "WebView": 0,
        "IonContent": 0,
        "IonButton": 0,
        "IonInput": 0,
    }

    max_depth = 0
    total_nodes = len(tree)

    # --- Conjuntos para diversidad textual ---
    button_texts = set()
    text_nodes = set()

    for node in tree:
        cls = (node.get("className") or "").lower()
        text = (node.get("text") or "").strip().lower()
        depth = node.get("depth", 0)
        max_depth = max(max_depth, depth)

        for key in features.keys():
            if key.lower() in cls:
                features[key] += 1

        # recolectar textos visibles
        if "button" in cls and text:
            button_texts.add(text)
        elif "textview" in cls and text:
            text_nodes.add(text)

    # --- Métricas agregadas ---
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

    # --- Diversidad textual ---
    unique_button_texts = len(button_texts)
    unique_text_nodes = len(text_nodes)

    # --- Vector final ---
    return [
        total_nodes,            # total de nodos
        max_depth,              # profundidad máxima
        interactive_elements,   # botones, inputs, switches...
        media_elements,         # imágenes, webviews...
        layout_complexity,      # layouts estructurales
        features["RecyclerView"],
        features["ScrollView"],
        features["ComposeView"],
        features["FlutterView"],
        features["IonContent"],
        unique_button_texts,    # 🔹 nuevo: número de textos de botón distintos
        unique_text_nodes,      # 🔹 nuevo: número de textos visibles distintos
    ]


def format_screen_diff(diffs):
    lines = []
    for diff in diffs:
        # aceptar ambos formatos: {"node": {...}, "changes": {...}} o nodo plano
        if "node" in diff:
            node = diff["node"]
            changes = diff.get("changes", {})
        else:
            # diff es un normalized node o plain dict (viene de removed/added sin envolver)
            node = {"key": diff.get("viewId") or f"{diff.get('className')}|{diff.get('pkg')}", "class": diff.get("className")}
            changes = {k: {"old": None, "new": diff.get(k)} for k in diff.keys() if k not in ("className", "pkg")}  # opcional
        node_name = f"{node.get('class')} ({node.get('key')})"
        lines.append(f" - {node_name}")
        for attr, val in changes.items():
            old = val.get('old')
            new = val.get('new')
            lines.append(f"    • {attr}: {old} → {new}")
    return "\n".join(lines)

def diff_hash(removed, added, modified):
    """Genera una firma única basada en el contenido del diff."""
    concat = json.dumps([removed, added, modified], sort_keys=True)
    return hashlib.sha1(concat.encode("utf-8")).hexdigest()


def is_expected_behavior_change(node, field, old, new, expected_initial=None):
    cls = node.get("class", "")

    # CheckBox, Switch, RadioButton
    if cls.endswith(("CheckBox", "Switch", "RadioButton")) and field == "checked":
        if old in (None, "", "null") and new in (True, False):
            # Ignorar solo si coincide con estado esperado
            return expected_initial is not None and new == expected_initial
        return False  # cambios True<->False son relevantes

    # Button / MaterialButton
    if cls.endswith(("Button", "MaterialButton")) and field == "enabled":
        if old in (None, "", "null") and new in (True, False):
            return expected_initial is not None and new == expected_initial
        return False  # cambios habilitado/deshabilitado son relevantes

    # Scrollables / grids: ignorar enabled
    if cls.endswith(("ScrollView", "RecyclerView", "ListView", "GridView")):
        if field == "enabled":
            return True

    # Todo lo demás es relevante
    return False


# def overlap_ratio(removed, added):
#     def extract_texts(nodes):
#         return {
#             n["node"]["key"].split("textsig:")[-1].strip().lower()
#             for n in nodes if "textsig:" in n["node"]["key"]
#         }

#     r_texts, a_texts = extract_texts(removed), extract_texts(added)
#     if not r_texts or not a_texts:
#         return 0.0
#     common = r_texts.intersection(a_texts)
#     total = r_texts.union(a_texts)
#     return len(common) / len(total)

def overlap_ratio(old_nodes, new_nodes):
    """
    Calcula el overlap de texto entre dos listas de nodos planos.
    No depende de textsig:, trabaja directamente sobre 'text'.
    """
    def extract_texts(nodes, label):
        texts = set()
        for n in nodes:
            text = n.get("text", "")
            if text:
                texts.add(text.strip().lower())
        print(f"[{label}] Textos extraídos ({len(texts)}): {texts}")
        return texts

    r_texts = extract_texts(old_nodes, "REMOVED")
    a_texts = extract_texts(new_nodes, "ADDED")

    # ✅ Manejo de casos vacíos para evitar falsos positivos y divisiones por cero
    if not r_texts and not a_texts:
        print("⚪ Ambos sets vacíos → overlap = 1.0 (sin cambios detectados)")
        return 1.0
    elif not r_texts or not a_texts:
        print("⚠️ Uno de los sets está vacío → overlap = 0.0 (sin elementos para comparar)")
        return 0.0

    common = r_texts.intersection(a_texts)
    total = r_texts.union(a_texts)
    overlap = len(common) / len(total)
    print(f"➡️ Overlap: {len(common)}/{len(total)} = {overlap:.2f}")
    return overlap

async def analyze_and_train(event: AccessibilityEvent):
    # -------------------- Normalizar campos --------------------
    norm = _normalize_event_fields(event)
    t_id, b_id = norm.get("tester_id_norm"), norm.get("build_id_norm")
    s_name = normalize_header(event.header_text)
    app_name = event.package_name or "default_app"
    tester_id = event.tester_id or "general"
    build_id = event.build_id

    # -------------------- Árbol y firma ------------------------
    latest_tree = ensure_list(event.collect_node_tree or event.tree_data or [])
    sig = stable_signature(latest_tree)

    root_class_name = latest_tree[0].get("className") if latest_tree else ""

    # -------------------- Features enriquecidas ----------------
    struct_vec = np.array(ui_structure_features(latest_tree), dtype=float).flatten()
    sig_vec = np.array(structure_signature_features(latest_tree), dtype=float).flatten()

    timestamps = [e.timestamp for e in ensure_list(event.actions or [])]
    time_deltas = np.diff(timestamps) if len(timestamps) > 1 else [0]
    avg_dwell = float(np.mean(time_deltas)) if len(time_deltas) > 0 else 0
    num_gestos = sum(1 for e in event.actions or [] if e.type in ["tap", "scroll"])
    input_vec = np.array(input_features(event.actions or []), dtype=float).flatten()

    enriched_vector = np.concatenate([
        struct_vec,
        sig_vec,
        np.array([avg_dwell, num_gestos], dtype=float),
        input_vec
    ])

    # -------------------- Obtener snapshot previo ----------------
    prev_tree = None
    prev_enriched_vec = np.zeros_like(enriched_vector)

    IGNORED_FIELDS = {"text", "headerText", "hint", "contentDescription", "value", "progress"}

    def normalize_node(node):
        return {k: v for k, v in node.items() if k not in IGNORED_FIELDS}

    def trees_are_structurally_similar(tree_a, tree_b, threshold=0.8):
        set_a = {json.dumps(normalize_node(n), sort_keys=True) for n in tree_a}
        set_b = {json.dumps(normalize_node(n), sort_keys=True) for n in tree_b}
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return (inter / union) >= threshold if union > 0 else False
    
    def get_class_name(row):
        try:
            views = json.loads(row[0])
            if isinstance(views, list) and len(views) > 0:
                return views[0].get("className", "")
        except Exception:
            pass
        return ""


    with sqlite3.connect(DB_NAME) as conn:
        prev_rows = conn.execute("""
            SELECT collect_node_tree, header_text, signature, enriched_vector, build_id, event_type_name
            FROM accessibility_data
            WHERE LOWER(TRIM(tester_id)) = LOWER(TRIM(?))
            ORDER BY created_at DESC
            LIMIT 5
        """, (t_id,)).fetchall()

    previous_row = None
    if prev_rows:
        latest_event = getattr(event, "event_type_name", None)

        # ---------- Filtro mejorado por tipo de evento + className + signature ----------
        same_event_rows = []
        if latest_event:
            # className y signature del evento actual (más reciente)
            current_class = root_class_name or get_class_name(prev_rows[-1])
            current_signature = sig

            # 🔍 DEBUG opcional: ver coincidencias campo a campo
            logger.info("🔍 Evaluando coincidencias previas:")
            for r in prev_rows:
                logger.info({
                    "event": r[5],
                    "class": get_class_name(r),
                    "signature": (r[2][:8] if r[2] else None),
                    "match_event": r[5] == latest_event,
                    "match_class": get_class_name(r) == current_class,
                    "match_sig": r[2] == current_signature,
                })

            same_event_rows = [
                r for r in prev_rows
                if (
                    r[5] == latest_event and
                    get_class_name(r) == current_class and
                    (r[2] == current_signature or not current_signature)
                )
            ]

        # Si no hay coincidencias, usar al menos la última fila como fallback
        if not same_event_rows and prev_rows:
            same_event_rows = [prev_rows[-1]]

        # ---------- Comparación vectorial y estructural ----------
        latest_vec = enriched_vector.reshape(1, -1)
        best_sim, best_row = 0, None
        for r in same_event_rows:
            try:
                prev_vec = np.array(json.loads(r[3])).reshape(1, -1)
                sim = float(cosine_similarity(latest_vec, prev_vec)[0][0])
                if sim > best_sim:
                    best_sim, best_row = sim, r
            except Exception:
                continue

        if best_row and best_sim > 0.95:
            previous_row = best_row
            prev_tree = ensure_list(json.loads(best_row[0]))
            logger.info(f"🤝 Coincidencia alta por similitud vectorial ({best_sim:.3f})")
        else:
            latest_tree_norm = [normalize_node(n) for n in latest_tree]
            for r in same_event_rows:
                try:
                    prev_candidate = ensure_list(json.loads(r[0]))
                    if trees_are_structurally_similar(latest_tree_norm, prev_candidate):
                        previous_row = r
                        prev_tree = prev_candidate
                        logger.info("🔄 Coincidencia por estructura (layout similar, texto distinto)")
                        break
                except Exception:
                    continue
            else:
                logger.warning("⚠️ No hay coincidencia por signature, vector ni estructura.")

        if prev_tree is not None:
            try:
                prev_enriched_vec = np.array(json.loads(previous_row[3]), dtype=float)
            except Exception:
                prev_enriched_vec = np.zeros_like(enriched_vector)

    # -------------------- Comparación de árboles ----------------
    removed_all, added_all, modified_all = [], [], []

    if prev_tree:
        try:
            diff_result = compare_trees(prev_tree, latest_tree)
        except Exception as e:
            logger.error(f"❌ Error ejecutando compare_trees: {e}")
            diff_result = {"removed": [], "added": [], "modified": [], "text_diff": {}}

        removed = diff_result["removed"]
        added = diff_result["added"]
        modified = diff_result["modified"]
        text_diff = diff_result.get("text_diff", {})

        # -------------------- Convertir cambios de texto a modified ----------------
        for r in removed[:]:
            if r["node"].get("text") in text_diff.get("removed", []):
                removed.remove(r)
                modified.append({
                    "node": r["node"],
                    "changes": {"text": {"old": r["node"]["text"], "new": text_diff["added"].get(r["node"]["text"], "")}}
                })
        for a in added[:]:
            if a["node"].get("text") in text_diff.get("added", []):
                added.remove(a)

        removed_all.extend(removed)
        added_all.extend(added)
        modified_all.extend(modified)

        logger.info(f"📊 compare_trees → removed={len(removed_all)}, added={len(added_all)}, modified={len(modified_all)}")
        logger.info(f"📊 Text overlap ratio: {overlap_ratio(removed_all, added_all):.2f}")

    # -------------------- Guardar en screen_diffs ----------------
    if removed_all or added_all or modified_all:
        removed_j = json.dumps(removed_all, sort_keys=True, ensure_ascii=False)
        added_j = json.dumps(added_all, sort_keys=True, ensure_ascii=False)
        modified_j = json.dumps(modified_all, sort_keys=True, ensure_ascii=False)
        diff_signature = diff_hash(removed_all, added_all, modified_all)

        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT s.id, a.approved
                FROM screen_diffs AS s
                LEFT JOIN diff_approvals AS a ON a.diff_id = s.id
                WHERE s.diff_hash = ?
            """, (diff_signature,))
            existing = cur.fetchone()

            if existing and existing[1]:
                logger.info(f"✅ Diff {diff_signature[:8]} ya aprobado — no se inserta.")
            elif existing:
                logger.info(f"⚠️ Diff {diff_signature[:8]} ya existente sin aprobación.")
            else:
                text_diff_j = json.dumps(text_diff, ensure_ascii=False)
                cur.execute("""
                    INSERT INTO screen_diffs (tester_id, build_id, header_text, removed, added, modified, text_diff, diff_hash)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (t_id, b_id, s_name, removed_j, added_j, modified_j, text_diff_j, diff_signature))
                conn.commit()
                logger.info(f"🧩 Guardado cambio ({diff_signature[:8]}) en screen_diffs")

    # -------------------- Actualizar enriched_vector ----------------
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            UPDATE accessibility_data
            SET enriched_vector=?
            WHERE TRIM(LOWER(header_text)) LIKE '%' || TRIM(LOWER(?)) || '%'
              AND TRIM(tester_id)=TRIM(?)
        """, (json.dumps(enriched_vector.tolist(), ensure_ascii=False), s_name, t_id))
        conn.commit()

    # -------------------- Entrenamiento incremental ----------------
    asyncio.create_task(_train_model_hybrid(
        X=np.array([enriched_vector]),
        tester_id=tester_id,
        build_id=build_id,
        app_name=app_name,
        desc=f"{app_name} incremental"
    ))

    if TRAIN_GENERAL_ON_COLLECT:
        await _train_model_hybrid(
            X=np.array([enriched_vector]),
            tester_id="general",
            build_id="latest",
            app_name=app_name,
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
            logger.info(f"📝 Trace guardado: {message}")

            
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
    return hashlib.sha256(
        json.dumps(norm, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

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

        # Determinar className principal o asignar por defecto si es splash
        if normalized_nodes and isinstance(normalized_nodes, list):
            root_class_name = normalized_nodes[0].get("className", "") if isinstance(normalized_nodes[0], dict) else ""
        else:
            root_class_name = ""

        # Asignar nombre por defecto si está vacío o el árbol es muy pequeño
        if not root_class_name or len(normalized_nodes) <= 2:
            root_class_name = "SplashActivity"


        logger.info(f"[collect] Root className detectado: {root_class_name}")

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
        #input_vec = np.array(input_features(event.actions or []), dtype=float)

        timestamps = [e.timestamp for e in ensure_list(event.actions or [])]
        time_deltas = np.diff(timestamps) if len(timestamps) > 1 else [0]
        avg_dwell = float(np.mean(time_deltas)) if len(time_deltas) > 0 else 0
        num_gestos = sum(1 for e in event.actions or [] if e.type in ["tap", "scroll"])
        input_vec = np.array(input_features(event.actions or []), dtype=float).flatten()

        combined = np.concatenate([
            np.array(struct_vec, dtype=float).flatten(),
            sig_vec.flatten(),
            np.array([avg_dwell, num_gestos], dtype=float),
            input_vec.flatten()
        ])
        #enriched_vector = combined.astype(float)
        enriched_vector = np.concatenate([
            struct_vec,
            sig_vec,
            np.array([avg_dwell, num_gestos], dtype=float),
            input_vec
        ])


        cluster_id: int | None = None
        anomaly_score: float | None = None

        # justo antes del INSERT, define:
        db_class_name = root_class_name or (event.class_name or "")

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
                event.event_type_name, event.package_name, db_class_name,
                event.text, event.content_description, screens_id_val,
                event.screen_names, clean_header,
                json.dumps(normalized_nodes, ensure_ascii=False),
                signature,
                json.dumps(event.additional_info, ensure_ascii=False) if event.additional_info else None,
                json.dumps(event.tree_data, ensure_ascii=False) if event.tree_data else None,
                json.dumps(enriched_vector.tolist(), ensure_ascii=False) if enriched_vector is not None else None,
                cluster_id,
                anomaly_score
            ))
            conn.commit()
            conn.close()
            logger.info("[collect] Insert completado.")
        else:
            logger.info("[collect] Se omitió insert porque snapshot coincide con último.")
            logger.info(f"[collect] Insert completado con class_name={root_class_name}")

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


@app.get("/screen/diffs")
def get_screen_diffs(
    tester_id: Optional[str] = Query(None),
    build_id: Optional[str] = Query(None),
    screen_name: Optional[str] = Query(None),
    only_pending: bool = Query(True)
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = """
        SELECT s.id, s.tester_id, s.build_id, s.screen_name, s.header_text,
               s.removed, s.added, s.modified, s.text_diff, s.created_at, s.cluster_info
        FROM screen_diffs AS s
        LEFT JOIN diff_approvals AS a
               ON a.diff_id = s.id
        WHERE 1=1
    """
    params = []

    if only_pending:
        query += " AND a.id IS NULL"

    if tester_id is not None:
        query += " AND (s.tester_id = ? OR (s.tester_id IS NULL AND ? = ''))"
        params.extend([tester_id, tester_id])

    if build_id is not None:
        query += " AND (s.build_id = ? OR (s.build_id IS NULL AND ? = ''))"
        params.extend([build_id, build_id])

    if screen_name is not None:
        query += " AND s.screen_name = ?"
        params.append(screen_name)

    query += """
        AND (
            COALESCE(s.removed, '[]') != '[]'
            OR COALESCE(s.added, '[]') != '[]'
            OR COALESCE(s.modified, '[]') != '[]'
        )
        ORDER BY s.created_at DESC
    """

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    def safe_json_load(v):
        try:
            return json.loads(v) if v else []
        except Exception:
            return []

    # --- Función auxiliar para generar el resumen elegante ---
    from io import StringIO


    def capture_pretty_summary(removed_all, added_all, modified_all):
        """Convierte los cambios en mensajes legibles tipo IA / QA."""
        lines = []

        def format_node_text(node):
            """Usar texto real si existe, sino la key."""
            return node.get("text") or node.get("desc") or node.get("contentDescription") or node.get("hint") or node.get("key", "")

        # --- Procesar eliminados ---
        for node in removed_all:
            text = format_node_text(node)
            lines.append(f"🗑️ {node.get('class','unknown')} eliminado: “{text}”")

        # --- Procesar agregados ---
        for node in added_all:
            text = format_node_text(node)
            lines.append(f"🆕 {node.get('class','unknown')} agregado: “{text}”")

        # --- Procesar modificados ---
        for change in modified_all:
            node = change.get("node", {})
            changes = change.get("changes", {})
            if not changes:
                # Sin cambios visibles, pero mismo nodo
                text = format_node_text(node)
                lines.append(f"✏️ {node.get('class','unknown')} sin cambios visibles: “{text}”")
            else:
                for attr, vals in changes.items():
                    old = vals.get("old")
                    new = vals.get("new")
                    # Si tiene JSON largo, simplificar
                    if isinstance(old, str) and old.startswith("{"): old = "(estructura)"
                    if isinstance(new, str) and new.startswith("{"): new = "(estructura)"
                    lines.append(f"✏️ {node.get('class','unknown')} modificado ({attr}): “{old}” → “{new}”")

        if not lines:
            return "✅ Sin cambios relevantes detectados."

        return "\n".join(lines)

    # --- Construir lista de diffs ---
    diffs = []

    for row in rows:
        removed = safe_json_load(row[5])
        added = safe_json_load(row[6])
        modified = safe_json_load(row[7])
        text_diff = safe_json_load(row[8])

                # 🧠 Analizar el overlap de texto
        overlap_ratio = 0.0
        screen_status = "unknown"
        try:
            overlap_ratio = text_diff.get("overlap_ratio", 0.0)
            if overlap_ratio >= 0.98:
                screen_status = "identical"  # sin cambios visibles
            elif overlap_ratio >= 0.8:
                screen_status = "minor_changes"  # cambios pequeños
            else:
                screen_status = "different"  # pantalla distinta
        except Exception:
            pass

        # 🔍 Expande los detalles de todos los cambios
        detailed_changes = []

        def add_node_change(action: str, node: dict):
            detailed_changes.append({
                "attribute": "(entire node)",
                "old_value": None if action == "added" else json.dumps(node, ensure_ascii=False),
                "new_value": json.dumps(node, ensure_ascii=False) if action == "added" else None,
                "node_class": node.get("class"),
                "node_key": node.get("key"),
                "node_text": node.get("text", ""),
                "pkg": node.get("pkg", ""),
                "action": action
            })

        # Procesar modificados
        for change in modified:
            node = change.get("node", {})
            changes = change.get("changes", {})
            if not changes:
                add_node_change("modified_empty", node)
            for attr, vals in changes.items():
                detailed_changes.append({
                    "attribute": attr,
                    "old_value": vals.get("old"),
                    "new_value": vals.get("new"),
                    "node_class": node.get("class"),
                    "node_key": node.get("key"),
                    "node_text": node.get("text", ""),
                    "pkg": node.get("pkg", ""),
                    "action": "modified"
                })

        # Procesar agregados y eliminados
        for node in added:
            add_node_change("added", node)

        for node in removed:
            add_node_change("removed", node)

        # 🆕 Generar resumen elegante para cada diff
        summary_text = capture_pretty_summary(removed, added, modified)

        diffs.append({
            "id": row[0],
            "tester_id": row[1],
            "build_id": row[2],
            "screen_name": row[3],
            "header_text": row[4],
            "removed_count": len(removed),
            "added_count": len(added),
            "modified_count": len(modified),
            "removed": removed,
            "added": added,
            "modified": modified,
            "text_diff": text_diff,
            "text_overlap": overlap_ratio,
            "screen_status": screen_status,
            "detailed_changes": detailed_changes,
            "created_at": row[8],
            "cluster_info": json.loads(row[9]) if row[9] else {},
            "detailed_summary": summary_text  # 🆕 Nuevo campo
        })

    # ✅ Cálculo robusto de has_changes
    # has_changes = any(
    #     len(d.get("removed", [])) > 0 or
    #     len(d.get("added", [])) > 0 or
    #     len(d.get("modified", [])) > 0
    #     for d in diffs
    # )

    # ✅ Cálculo robusto de has_changes (incluye cambios de texto)
    has_changes = any(
        len(d.get("removed", [])) > 0 or
        len(d.get("added", [])) > 0 or
        len(d.get("modified", [])) > 0 or
        (d.get("text_diff", {}).get("overlap_ratio", 1.0) < 0.95)
        for d in diffs
    )

    # print(f"🧩 has_changes={has_changes} | total_diffs={len(diffs)}")

    return {
        "screen_diffs": diffs,
        "has_changes": has_changes
    }


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
    # --- 1. Leer JSON ---
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Cuerpo JSON inválido o vacío")

    # --- 2. Validar diff_id ---
    diff_id = payload.get("diff_id") or payload.get("id")
    if diff_id is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "diff_id missing"})

    try:
        diff_id_int = int(diff_id)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"status": "error", "message": "diff_id must be integer"})

    # --- 3. Guardar en DB ---
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Crear tabla si no existe
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diff_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diff_id INTEGER,
                approved INTEGER DEFAULT 1,
                approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Verificar si ya está aprobado
        cursor.execute("SELECT 1 FROM diff_approvals WHERE diff_id = ?", (diff_id_int,))
        if cursor.fetchone():
            return {"status": "already_approved", "diff_id": diff_id_int}

        # Insertar aprobación
        cursor.execute("INSERT INTO diff_approvals(diff_id, approved) VALUES (?, 1)", (diff_id_int,))
        conn.commit()

    except Exception as db_err:
        logger.exception("Error de base de datos en /approve_diff")
        raise HTTPException(status_code=500, detail=f"DB error: {db_err}")
    finally:
        conn.close()

    logger.info("✅ Diff %s aprobado vía API", diff_id_int)
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
                labels: {json.dumps(builds_sorted, ensure_ascii=False)},
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
