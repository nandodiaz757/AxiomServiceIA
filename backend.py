from fastapi import FastAPI, Query, BackgroundTasks, Request, APIRouter, HTTPException, Depends, WebSocket, Body
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, ConfigDict
import sqlite3, json, joblib, numpy as np, os, hashlib, logging, asyncio, re, unicodedata
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
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
from models_pipeline import compare_trees, _train_incremental_logic_hybrid, _train_general_logic_hybrid
import httpx
import ast
from packaging import version
import difflib
import string
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SiameseEncoder import SiameseEncoder



from stable_signature import normalize_node

# =========================================================
# CONFIGURACIÓN
# =========================================================
app = FastAPI()
router = APIRouter() 
siamese_model = None
SIM_THRESHOLD = 0.90

# Cargamos modelo siamés (una vez)


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

last_ui_structure_similarity = 0.0

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

class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, embedding_dim=64):
        super().__init__()

        # Bloque MLP simple: transforma features a embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    # ----------------------------------------------------------
    # Conversión de árbol UI a vector fijo de features
    # ----------------------------------------------------------
    def tree_to_vector(self, ui_tree):
        """
        Convierte un árbol de accesibilidad (lista de nodos JSON)
        en un vector numérico simple.
        """
        if not ui_tree or not isinstance(ui_tree, list):
            return np.zeros(128, dtype=np.float32)

        features = []
        for node in ui_tree[:50]:  # limita nodos para evitar overflow
            cls = node.get("className", "")
            txt = node.get("text", "")
            clickable = 1.0 if node.get("clickable") else 0.0
            enabled = 1.0 if node.get("enabled", True) else 0.0
            size = float(node.get("bounds", {}).get("width", 0)) * \
                   float(node.get("bounds", {}).get("height", 0))
            size = np.log1p(size) / 10.0

            # hash textual básico (reemplazable por embedding textual real)
            text_hash = (sum(ord(c) for c in txt[:10]) % 1000) / 1000.0
            cls_hash = (sum(ord(c) for c in cls[:10]) % 1000) / 1000.0

            node_vec = [clickable, enabled, size, text_hash, cls_hash]
            features.append(node_vec)

        # Flatten y normaliza tamaño
        flat = np.array(features, dtype=np.float32).flatten()
        if len(flat) < 128:
            pad = np.zeros(128 - len(flat), dtype=np.float32)
            flat = np.concatenate([flat, pad])
        else:
            flat = flat[:128]
        return flat

    # ----------------------------------------------------------
    # Encodea un árbol a embedding
    # ----------------------------------------------------------
    def encode_tree(self, ui_tree):
        vec = torch.tensor(self.tree_to_vector(ui_tree), dtype=torch.float32)
        with torch.no_grad():
            emb = self.encoder(vec)
            emb = F.normalize(emb, p=2, dim=0)  # normaliza L2
        return emb

    # ----------------------------------------------------------
    # Forward siamés: compara dos árboles y devuelve similitud
    # ----------------------------------------------------------
    def forward(self, tree_a, tree_b):
        va = torch.tensor(self.tree_to_vector(tree_a), dtype=torch.float32)
        vb = torch.tensor(self.tree_to_vector(tree_b), dtype=torch.float32)
        ea = F.normalize(self.encoder(va), p=2, dim=0)
        eb = F.normalize(self.encoder(vb), p=2, dim=0)
        sim = F.cosine_similarity(ea, eb, dim=0)
        return sim

    # ----------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------
    def save(self, path="ui_encoder.pt"):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path="ui_encoder.pt"):
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
    
@app.on_event("startup")
async def startup_event():
    """Carga inicial del modelo al arrancar el servidor."""
    global siamese_model
    from SiameseEncoder import SiameseEncoder

    siamese_model = SiameseEncoder.load("ui_encoder.pt")
    siamese_model.eval()
    print("✅ Modelo siamés cargado en memoria.")   

# === Reentrenamiento automático del modelo diff ===
@app.on_event("startup")
@repeat_every(seconds=3600)  # cada hora, ajusta el intervalo a lo que necesites
def retrain_model() -> None:
    """
    Reentrena el modelo diff con las últimas aprobaciones/rechazos
    sin necesidad de parar el servidor.
    """
    """Carga inicial del modelo al arrancar el servidor."""

    global siamese_model
    from diff_model.train_diff_model import train_and_save
    print("🧠 Iniciando reentrenamiento del modelo siamés...")

    # Entrena y guarda nuevo modelo
    train_and_save()

    #from SiameseEncoder import SiameseEncoder

    siamese_model = SiameseEncoder.load("ui_encoder.pt")
    siamese_model.eval()
    print("✅ Reentrenamiento completado y modelo actualizado en memoria.")




def _get_lock(key: str) -> asyncio.Lock:
    if key not in _model_locks:
        _model_locks[key] = asyncio.Lock()
    return _model_locks[key]

def ui_structure_similarity(tree_a, tree_b):
    """Compara estructuras jerárquicas de dos árboles de UI, ignorando texto exacto."""
    def flatten_structure(node, depth=0):
        """Aplana jerarquía en etiquetas con nivel + clase + hints."""
        nodes = []
        if isinstance(node, dict):
            cls = node.get("className", "")
            hint = node.get("hint", "")
            desc = node.get("contentDescription", "")
            text = node.get("text", "")
            tag = f"{depth}|{cls}|{hint or desc or text}"
            nodes.append(tag)
            for child in node.get("children") or []:
                nodes.extend(flatten_structure(child, depth + 1))
        elif isinstance(node, list):
            for child in node:
                nodes.extend(flatten_structure(child, depth))
        return nodes

    a_nodes = flatten_structure(tree_a)
    b_nodes = flatten_structure(tree_b)

    a_str = "\n".join(sorted(a_nodes))
    b_str = "\n".join(sorted(b_nodes))

    return difflib.SequenceMatcher(None, a_str, b_str).ratio()


def sequence_entropy(seq: list[str]) -> float:
    """Calcula la entropía de Shannon de una secuencia de elementos."""
    if not seq:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    ent = -sum((count/total) * math.log2(count/total) for count in counts.values())
    return ent


def init_metrics_table():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tester_id TEXT,
                build_id TEXT,
                total_events INTEGER DEFAULT 0,
                total_changes INTEGER DEFAULT 0,
                total_added INTEGER DEFAULT 0,
                total_removed INTEGER DEFAULT 0,
                total_modified INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tester_id, build_id)
            )
        """)
        conn.commit()


# Actualizar métricas
def update_metrics(tester_id: str, build_id: str, has_changes: bool,
                   added_count: int = 0, removed_count: int = 0, modified_count: int = 0):
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO metrics_changes (
                tester_id, build_id, total_events, total_changes,
                total_added, total_removed, total_modified
            )
            VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(tester_id, build_id)
            DO UPDATE SET
                total_events = total_events + 1,
                total_changes = total_changes + EXCLUDED.total_changes,
                total_added = total_added + EXCLUDED.total_added,
                total_removed = total_removed + EXCLUDED.total_removed,
                total_modified = total_modified + EXCLUDED.total_modified,
                last_updated = CURRENT_TIMESTAMP
        """, (
            tester_id,
            build_id,
            1 if has_changes else 0,
            added_count,
            removed_count,
            modified_count
        ))
        conn.commit()

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
            text_overlap REAL DEFAULT 0,
            overlap_ratio REAL DEFAULT 0,
            ui_structure_similarity REAL DEFAULT 0,      
            cluster_id INTEGER DEFAULT -1,  
            screen_status TEXT DEFAULT 'unknown',  
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
        CREATE TABLE IF NOT EXISTS login_codes (
            codigo TEXT PRIMARY KEY,
            usuario_id TEXT NOT NULL,
            generado_en INTEGER NOT NULL,
            expira_en INTEGER NOT NULL,
            usos_permitidos INTEGER NOT NULL,
            usos_actuales INTEGER DEFAULT 0,
            activo INTEGER DEFAULT 1  
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS pagos (
            pago_id TEXT PRIMARY KEY,    
            membresia_id TEXT NOT NULL,     
            usuario_id TEXT NOT NULL,
            proveedor TEXT NOT NULL,           
            proveedor_id TEXT,                  
            monto INTEGER NOT NULL,
            moneda TEXT DEFAULT 'USD',
            estado TEXT DEFAULT 'PENDIENTE',  
            transaccion_id TEXT   
            cantidad_codigos INTEGER NOT NULL,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fecha_confirmacion INTEGER 
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS membresias (
            membresia_id TEXT PRIMARY KEY,
            usuario_id TEXT NOT NULL,
            tipo_plan TEXT NOT NULL,
            cantidad_codigos INTEGER NOT NULL,
            fecha_inicio INTEGER NOT NULL,
            fecha_fin INTEGER NOT NULL 
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS pagos_log  (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pago_id TEXT NOT NULL,
            evento TEXT NOT NULL,             -- Ej: "CREADO", "CONFIRMADO", "FALLIDO", "CODIGO_GENERADO"
            descripcion TEXT,                 -- Detalles del evento
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(pago_id) REFERENCES pagos(pago_id)
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS transacciones (
            transaccion_id TEXT PRIMARY KEY,
            proveedor TEXT NOT NULL,          -- Stripe, PayPal, Banco
            proveedor_id TEXT,                -- ID de la transacción en el proveedor
            monto REAL NOT NULL,
            moneda TEXT DEFAULT 'USD',
            estado TEXT DEFAULT 'PENDIENTE',  -- PENDIENTE, CONFIRMADA, FALLIDA
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    # ✅ NUEVA TABLA: métricas de cambios
    c.execute("""
        CREATE TABLE IF NOT EXISTS metrics_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tester_id TEXT,
            build_id TEXT,
            total_events INTEGER DEFAULT 0,
            total_changes INTEGER DEFAULT 0,
            total_added INTEGER DEFAULT 0,      
            total_removed INTEGER DEFAULT 0,    
            total_modified INTEGER DEFAULT 0,   
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tester_id, build_id)
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


    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow"
    )
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
    # print("Flattened nodes:", flat)
    return flat

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

# def diff_hash(removed, added, modified):
#     """Genera una firma única basada en el contenido del diff."""
#     concat = json.dumps([removed, added, modified], sort_keys=True)
#     return hashlib.sha1(concat.encode("utf-8")).hexdigest()

def diff_hash(removed, added, modified, text_diff=None):
    """Genera una firma única basada en los cambios detectados (estructura + texto)."""
    def normalize(x):
        return sorted(x, key=lambda v: json.dumps(v, sort_keys=True)) if isinstance(x, list) else x

    concat = json.dumps(
        [normalize(removed), normalize(added), normalize(modified), text_diff or {}],
        sort_keys=True
    )
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
    t_id = str(norm.get("tester_id_norm") or "").strip()
    b_id = str(norm.get("build_id_norm") or "").strip()
    s_name = normalize_header(event.header_text)
    event_type_ref = normalize_header(event.event_type_name)
    app_name = event.package_name or "default_app"
    tester_id = event.tester_id or "general"
    build_id = event.build_id
    header_text = event.header_text or ""
    global siamese_model
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

    # Valor por defecto, para evitar errores si hay excepciones antes de asignarlo
    has_changes = False  

    # -------------------- Verificación rápida de cambio de header --------------------
    try:
        curr_header = (s_name or "").strip().lower()
        prev_header = None

        with sqlite3.connect(DB_NAME) as conn:
            # Buscar el último header distinto para el mismo tester
            row = conn.execute("""
                SELECT header_text
                FROM accessibility_data
                WHERE LOWER(TRIM(tester_id)) = LOWER(TRIM(?))
                AND LOWER(TRIM(header_text)) != LOWER(TRIM(?))
                AND LOWER(TRIM(event_type_name)) = LOWER(TRIM(?))
                ORDER BY created_at DESC
                LIMIT 1
            """, (t_id, curr_header, event_type_ref)).fetchone()

            if row:
                prev_header = (row[0] or "").strip().lower()
        print(f"curr_header: '{curr_header}' | prev_header: '{prev_header}'")
        if prev_header and prev_header != curr_header:
            print(f"⚠️ Header cambió (detección temprana): '{prev_header}' → '{curr_header}'")
            header_changed = {"before": prev_header, "after": curr_header}
            has_changes = True
        else:
            header_changed = None
            print("✅ Header sin cambios (verificación temprana).")

    except Exception as e:
        header_changed = None
        logger.warning(f"⚠️ Error en verificación temprana de header_text: {e}")

    # -------------------- Obtener snapshot previo ----------------
    prev_tree = None
    IGNORED_FIELDS = {"hint", "contentDescription", "value", "progress"}

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
   
# normalizar latest_tree: asegurar lista de dicts, eliminar Nones
    def sanitize_tree(tree):
        if not tree:
            return []
        cleaned = []
        for n in tree:
            # aceptamos dicts; también aceptamos strings JSON (por si llegan así)
            if n is None:
                continue
            if isinstance(n, str):
                try:
                    obj = json.loads(n)
                    if isinstance(obj, dict):
                        cleaned.append(obj)
                    elif isinstance(obj, list):
                        # si vino un lista serializada, extenderla
                        cleaned.extend([x for x in obj if isinstance(x, dict)])
                except Exception:
                    continue
            elif isinstance(n, dict):
                cleaned.append(n)
            # ignorar otros tipos
        return cleaned

    latest_tree = sanitize_tree(latest_tree)
    if not latest_tree:
        logger.warning("⚠️ latest_tree vacío o todos sus nodos son None/invalid — usando embedding neutro")
        # crear embedding neutro (mismo tamaño que modelo devuelve)
        try:
            emb_dim = siamese_model.embedding_dim  # define esto en la clase SiameseEncoder
            emb_curr = np.zeros((1, emb_dim), dtype=float)
        except Exception:
            emb_curr = np.zeros((1, 64), dtype=float)  # fallback si no conoces dim
    else:
        try:
            with torch.no_grad():
                emb_tensor = siamese_model.encode_tree(latest_tree)  # debe devolver tensor 1D o 2D
            # asegurar formato numpy (1, -1)
            emb_curr = emb_tensor.cpu().numpy().reshape(1, -1)
        except Exception as e:
            # log detallado y fallback neutro pero no abortar proceso
            logger.exception(f"Error generando embedding: {e} -- fallback embedding neutro usado")
            try:
                emb_dim = getattr(siamese_model, "embedding_dim", 64)
                emb_curr = np.zeros((1, emb_dim), dtype=float)
            except Exception:
                emb_curr = np.zeros((1, 64), dtype=float)

    with sqlite3.connect(DB_NAME) as conn:
        prev_rows = conn.execute("""
            SELECT collect_node_tree, header_text, signature, enriched_vector, build_id, event_type_name
            FROM accessibility_data
            WHERE LOWER(TRIM(tester_id)) = LOWER(TRIM(?))
              AND LOWER(TRIM(build_id)) != LOWER(TRIM(?))
            ORDER BY created_at DESC
            LIMIT 5
        """, (t_id, b_id)).fetchall()

    # esto es nuevo tocar o eliminar 
        if not prev_rows:
            logger.info("ℹ️ No hay snapshots previos para comparar.")
            has_changes = True
            prev_tree = None
        else:
            # -------------------- 4. Calcular similitud aprendida --------------------
            best_sim, best_row = 0.0, None
            for row in prev_rows:
                try:
                    prev_tree = ensure_list(json.loads(row[0]))
                    with torch.no_grad():
                        emb_prev = siamese_model.encode_tree(prev_tree)
                    emb_prev = emb_prev.cpu().numpy().reshape(1, -1)

                    sim_torch = torch.nn.functional.cosine_similarity(
                        torch.tensor(emb_curr, dtype=torch.float32),
                        torch.tensor(emb_prev, dtype=torch.float32),
                        dim=1
                    )
                    print("Similitud (torch):", sim_torch.mean().item())

                    sim = float(cosine_similarity(emb_curr, emb_prev)[0][0])
                    # sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
                    # print("Similitud:", sim.item())
                    # sim = float(cosine_similarity(emb_curr, emb_prev)[0][0])
                    if sim > best_sim:
                        best_sim, best_row = sim, row
                except Exception:
                    continue

            if best_sim > SIM_THRESHOLD and best_row:
                logger.info(f"🤝 Coincidencia detectada por modelo siamés (sim={best_sim:.3f})")
                prev_tree = ensure_list(json.loads(best_row[0]))
            else:
                logger.warning("⚠️ No se encontró coincidencia fuerte, usar estructura directa.")
                prev_tree = ensure_list(json.loads(prev_rows[0][0]))    
    
    removed_all, added_all, modified_all = [], [], []
    text_diff = {}
    diff_result = {}
    
    # esto es nuevo  tocar o eliminar  
    if prev_tree:
        try:
            diff_result = compare_trees(
                prev_tree, latest_tree,
                app_name=app_name,
                tester_id=tester_id,
                build_id=build_id,
                screen_id=event.screens_id or event.header_text or "unknown_screen"
            )

            has_changes = bool(
                diff_result.get("removed") or diff_result.get("added") or diff_result.get("modified") or
                diff_result.get("text_diff", {}).get("overlap_ratio", 1.0) < 0.9 or
                diff_result.get("has_changes")
            )

        except Exception as e:
            logger.error(f"Error comparando árboles: {e}")
            has_changes = True
    else:
        has_changes = True

    # -------------------- 6. Guardar diff + enriquecimiento --------------------
    with sqlite3.connect(DB_NAME) as conn:
        emb_json = json.dumps(emb_curr.tolist())
        conn.execute("""
            UPDATE accessibility_data
            SET enriched_vector=?
            WHERE TRIM(LOWER(header_text)) LIKE '%' || TRIM(LOWER(?)) || '%'
              AND TRIM(tester_id)=TRIM(?)
        """, (emb_json, s_name, t_id))
        conn.commit()

    # -------------------- 7. Entrenamiento incremental --------------------
    # opcional: puedes volver a entrenar el encoder con nuevos ejemplos
    asyncio.create_task(_train_incremental_logic_hybrid(
        enriched_vector=emb_curr.flatten(),
        tester_id=tester_id,
        build_id=build_id,
        app_name=app_name,
        screen_id=event.screens_id or s_name or "unknown_screen"
    ))

    if TRAIN_GENERAL_ON_COLLECT:
        await _train_incremental_logic_hybrid(
            enriched_vector=emb_curr.flatten(),
            tester_id="general",
            build_id="latest",
            app_name=app_name,
            screen_id=event.screens_id or s_name or "unknown_screen"
        )

    # -------------------- 9. Guardar en screen_diffs --------------------
    try:
        if has_changes:
            removed_all = diff_result.get("removed", [])
            added_all = diff_result.get("added", [])
            modified_all = diff_result.get("modified", [])
            text_diff = diff_result.get("text_diff", {})
            header_changed = text_diff.get("header_changed", None)

            # --- Generar firma hash del diff ---
            diff_signature = diff_hash(removed_all, added_all, modified_all, text_diff)

            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()

                # Buscar si ya existe este diff
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
                    # Enriquecer text_diff si falta overlap_ratio o structure_similarity
                    if "overlap_ratio" not in text_diff:
                        text_diff["overlap_ratio"] = diff_result.get("text_diff", {}).get("overlap_ratio", 1.0)

                    if "structure_similarity" not in diff_result:
                        try:
                            diff_result["structure_similarity"] = ui_structure_similarity(prev_tree, latest_tree)
                        except Exception:
                            diff_result["structure_similarity"] = 1.0

                    text_overlap = text_diff.get("overlap_ratio", 1.0)
                    overlap_ratio = text_overlap
                    ui_structure_sim_value = diff_result.get("structure_similarity", 1.0)

                    # Determinar estado textual
                    if ui_structure_sim_value > 0.9 and overlap_ratio > 0.8:
                        screen_status = "identical"
                    elif overlap_ratio > 0.6:
                        screen_status = "minor_changes"
                    else:
                        screen_status = "different"

                    # Serializar
                    removed_j = json.dumps(removed_all, sort_keys=True, ensure_ascii=False)
                    added_j = json.dumps(added_all, sort_keys=True, ensure_ascii=False)
                    modified_j = json.dumps(modified_all, sort_keys=True, ensure_ascii=False)
                    text_diff_j = json.dumps(text_diff, ensure_ascii=False)


                    print("\n--- DEBUG INSERT INTO screen_diffs ---")
                    print(f"tester_id={t_id}, build_id={b_id}, screen_name={s_name}")
                    print(f"has_changes={has_changes}, removed={len(removed_all)}, added={len(added_all)}, modified={len(modified_all)}")
                    print(f"diff_hash={diff_signature[:10]}, text_overlap={text_diff.get('overlap_ratio')}")
                    print(f"existing={existing}")
                    print("-------------------------------\n")


                    cur.execute("""
                        INSERT OR IGNORE INTO screen_diffs (
                            tester_id, build_id, screen_name, header_text,
                            removed, added, modified, text_diff, diff_hash,
                            text_overlap, overlap_ratio, ui_structure_similarity, screen_status
                        )
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        t_id, b_id, s_name, header_text,
                        removed_j, added_j, modified_j, text_diff_j,
                        diff_signature, text_overlap, overlap_ratio,
                        ui_structure_sim_value, screen_status
                    ))

                    conn.commit()
                    logger.info(f"🧩 Guardado cambio ({diff_signature[:8]}) en screen_diffs")

        else:
            logger.info(f"🧩 Sin cambios detectados para {s_name}")
    except Exception as e:
        logger.exception(f"❌ Error guardando diff en screen_diffs: {e}")


    # -------------------- 8. Retornar métricas --------------------
    added_count = len(diff_result.get("added", []))
    removed_count = len(diff_result.get("removed", []))
    modified_count = len(diff_result.get("modified", []))
    return has_changes, added_count, removed_count, modified_count


def _insert_diff_trace(tester_id, build_id, screen_normalized, message):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        exists = c.execute("""
            SELECT 1 FROM diff_trace
            WHERE tester_id=? AND build_id=? AND screen_name=? AND message=?
            LIMIT 1
        """, (tester_id, build_id, screen_normalized, message)).fetchone()
        if not exists:
            c.execute("""
                INSERT INTO diff_trace (tester_id, build_id, screen_name, message)
                VALUES (?, ?, ?, ?)
            """, (tester_id, build_id, screen_normalized, message))
            conn.commit()
            logger.info(f"📝 Trace guardado: {message}")

 


def update_diff_trace(tester_id: str, build_id: str, screen: str, changes: list) -> None:
    """
    Actualiza la tabla diff_trace:
      - Si hay cambios, borra registros "No hay cambios" y agrega cada cambio.
      - Si no hay cambios, asegura que quede un único registro "No hay cambios".
    """
    screen_normalized = normalize_header(screen)

    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()

        if changes:
            # Eliminar registros "No hay cambios" para este tester/pantalla/build
            c.execute("""
                DELETE FROM diff_trace
                WHERE tester_id=? AND build_id=? AND screen_name=? AND message='No hay cambios'
            """, (tester_id, build_id, screen_normalized))

            # Insertar cada cambio, ignorando duplicados automáticamente
            for ch in changes:
                c.execute("""
                    INSERT OR IGNORE INTO diff_trace (tester_id, build_id, screen_name, message)
                    VALUES (?, ?, ?, ?)
                """, (tester_id, build_id, screen_normalized, ch))

        else:
            # Borrar otros mensajes distintos de "No hay cambios"
            c.execute("""
                DELETE FROM diff_trace
                WHERE tester_id=? AND build_id=? AND screen_name=? AND message <> 'No hay cambios'
            """, (tester_id, build_id, screen_normalized))

            # Asegurar que exista solo un registro "No hay cambios"
            c.execute("""
                INSERT OR IGNORE INTO diff_trace (tester_id, build_id, screen_name, message)
                VALUES (?, ?, ?, 'No hay cambios')
            """, (tester_id, build_id, screen_normalized))

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

@app.post("/collect")
async def collect_event(event: AccessibilityEvent, background_tasks: BackgroundTasks):
    logger.debug("Raw request: %s", event.model_dump())
    try:
        # -------------------- Normalización --------------------
        raw_nodes = event.collect_node_tree or event.tree_data or []
        normalized_nodes = normalize_tree(raw_nodes)
        signature = stable_signature(raw_nodes)
        norm = _normalize_event_fields(event)

        tester_norm = norm.get("tester_id_norm")
        build_norm = norm.get("build_id_norm")
        screen_name = event.screen_names or ""
        header_text = event.header_text or ""
        screens_id_val = event.screens_id or norm.get("screensId") or None

        # Class raíz
        if normalized_nodes and isinstance(normalized_nodes, list):
            root_class_name = normalized_nodes[0].get("className", "") if isinstance(normalized_nodes[0], dict) else ""
        else:
            root_class_name = ""
        if not root_class_name or len(normalized_nodes) <= 2:
            root_class_name = "SplashActivity"

        print(f"[collect] Root className detectado: {root_class_name}")
        print(f"[collect] tester={tester_norm} build={build_norm} screen={screen_name}")

        # -------------------- Estado inicial --------------------
        has_changes = False
        prev_build_name = None
        is_new_record = True

        # -------------------- Buscar último snapshot --------------------
        last = last_hash_for_screen(tester_norm, screen_name)
        logger.debug(f"[collect] last_hash={last} current_hash={screens_id_val}")

        prev_snapshot = None
        if last:
            with sqlite3.connect(DB_NAME) as conn:
                conn.row_factory = sqlite3.Row
                prev_snapshot = conn.execute("""
                    SELECT class_name, build_id, collect_node_tree
                    FROM accessibility_data
                    WHERE header_text = ? AND class_name = ?
                    ORDER BY id DESC LIMIT 1
                """, (header_text, root_class_name)).fetchone()

        if prev_snapshot:
            prev_build_name = prev_snapshot["build_id"]
            prev_nodes = json.loads(prev_snapshot["collect_node_tree"] or "[]")
            is_new_record = build_norm != prev_build_name

            # Ejecutar análisis y obtener cambios
            has_changes, added_count, removed_count, modified_count = await analyze_and_train(event)

            update_metrics(
                tester_norm,
                build_norm,
                has_changes,
                added_count=added_count,
                removed_count=removed_count,
                modified_count=modified_count
            )
            #return {"has_changes": has_changes}


        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            existing = cursor.execute("""
                SELECT 1 FROM accessibility_data
                WHERE tester_id=? AND build_id=? AND signature=?
            """, (tester_norm, build_norm, signature)).fetchone()

        do_insert = (is_new_record or has_changes) and not existing

        if do_insert:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO accessibility_data (
                        tester_id, build_id, timestamp, event_type, event_type_name,
                        package_name, class_name, text, content_description, screens_id,
                        screen_names, header_text, collect_node_tree, signature,
                        additional_info, tree_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tester_norm, build_norm, event.timestamp, event.event_type,
                    event.event_type_name, event.package_name, root_class_name,
                    event.text, event.content_description, screens_id_val,
                    event.screen_names, header_text,
                    json.dumps(normalized_nodes, ensure_ascii=False),
                    signature,
                    json.dumps(event.additional_info or {}, ensure_ascii=False),
                    json.dumps(event.tree_data or [], ensure_ascii=False)
                ))
                conn.commit()
            logger.info("[collect] Insert completado (nuevo build o cambios detectados).")
            return {"status": "success", "inserted": True, "has_changes": has_changes}

        # Si no hay cambios ni nuevo build
        return {"status": "skipped", "inserted": False, "has_changes": has_changes}

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
    app_name: str = Query(..., description="Nombre de la aplicación"),
    batch_size: int = Query(1000, ge=1, description="Tamaño máximo de muestras para entrenar"),
    min_samples: int = Query(2, ge=1, description="Mínimo de muestras por pantalla para poder entrenar"),
    # update_general: bool = Query(True, description="Forzar entrenamiento general híbrido")
    update_general: bool = Query(True, description="Usar modelo general como base")
):    
    """
    Endpoint para disparar el entrenamiento general híbrido por aplicación.
    Llama a _train_general_logic_hybrid con los parámetros especificados.
    """
    await _train_general_logic_hybrid(
        app_name=app_name,
        batch_size=batch_size,
        min_samples=min_samples,
        update_general=update_general
    )

    return {
        "status": "success",
        "message": f"Entrenamiento general híbrido disparado para app '{app_name}'",
        "params": {
            "batch_size": batch_size,
            "min_samples": min_samples,
             "update_general": update_general
        }
    }


@app.post("/train/incremental")
async def trigger_incremental_train(
    app_name: str = Query(..., description="Nombre de la aplicación"),
    tester_id: str = Query(..., description="Identificador del tester"),
    build_id: str = Query(..., description="ID del build o versión de la app"),
    screen_id: str = Query(..., description="ID de la pantalla actual"),
    min_samples: int = Query(2, ge=1, description="Número mínimo de muestras para entrenar"),
    use_general_as_base: bool = Query(True, description="Usar modelo general como base"),
    enriched_vector: Optional[List[float]] = Body(None, description="Vector enriquecido actual")
):
    """
    Endpoint para disparar el entrenamiento incremental híbrido de una pantalla específica.
    """
    if enriched_vector is None:
        return {
            "status": "error",
            "message": "Se requiere 'enriched_vector' para el entrenamiento incremental."
        }

    # Llamar al entrenamiento incremental híbrido
    await _train_incremental_logic_hybrid(
        enriched_vector=np.array(enriched_vector, dtype=float),
        tester_id=tester_id,
        build_id=build_id,
        app_name=app_name,
        screen_id=screen_id,
        min_samples=min_samples,
        use_general_as_base=use_general_as_base
    )

    return {
        "status": "success",
        "message": f"Entrenamiento incremental híbrido ejecutado para {app_name}/{tester_id}/{build_id}/{screen_id}",
        "params": {
            "min_samples": min_samples,
            "use_general_as_base": use_general_as_base
        }
    }


def extract_numeric_version(v: str) -> str:
    """Extrae el número de versión (por ejemplo, 'v1.2.3-beta' → '1.2.3')."""
    if not v:
        return None
    match = re.search(r"(\d+(?:\.\d+){0,2})", v)
    return match.group(1) if match else None


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

    if screen_name is not None:
        query += """
            AND (
                LOWER(TRIM(s.header_text)) LIKE LOWER(TRIM(?))
                OR (TRIM(s.header_text) = '' AND ? = '')
                OR (s.header_text IS NULL AND ? = '')
            )
        """
        like_pattern = f"%{screen_name.strip()}%"
        params.extend([like_pattern, screen_name, screen_name])

    query += " ORDER BY s.created_at DESC"

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    # 🧮 Filtro de build_id semántico (menor o igual)
    if build_id is not None:
        build_ver_str = extract_numeric_version(str(build_id))
        if build_ver_str:
            build_ver = version.parse(build_ver_str)
            filtered_rows = []
            for r in rows:
                r_build_str = extract_numeric_version(str(r[2]))
                if not r_build_str:
                    continue  # ignora los registros sin versión válida
                try:
                    if version.parse(r_build_str) <= build_ver:
                        filtered_rows.append(r)
                except Exception:
                    continue
            rows = filtered_rows

    def safe_json_load(v):
        try:
            return json.loads(v) if v else []
        except Exception:
            return []

    # --- Función auxiliar para generar el resumen elegante ---
    from io import StringIO


    def capture_pretty_summary(removed_all, added_all, modified_all, text_diff):
        lines = []

        def format_node_text(node):
            return (
                node.get("text")
                or node.get("desc")
                or node.get("contentDescription")
                or node.get("hint")
                or node.get("key", "")
            )

        # --- Eliminados ---
        for node in removed_all:
            text = format_node_text(node)
            lines.append(f"🗑️ {node.get('class','unknown')} eliminado: “{text}”")

        # --- Agregados ---
        for node in added_all:
            text = format_node_text(node)
            lines.append(f"🆕 {node.get('class','unknown')} agregado: “{text}”")

        # --- Modificados ---
        for change in modified_all:
            node = change.get("node", {})
            changes = change.get("changes", {})

            if not changes:
                text = format_node_text(node)
                lines.append(f"✏️ {node.get('class','unknown')} sin cambios visibles: “{text}”")
                continue

            for attr, vals in changes.items():
                # Manejar si vals no es un dict (por ejemplo, str, bool, int)
                if isinstance(vals, dict):
                    old = vals.get("old")
                    new = vals.get("new")
                else:
                    old = new = vals
                    new = vals

                # Simplificar estructuras grandes
                if isinstance(old, str) and old.startswith("{"): old = "(estructura)"
                if isinstance(new, str) and new.startswith("{"): new = "(estructura)"

                lines.append(f"✏️ {node.get('class','unknown')} modificado ({attr}): “{old}” → “{new}”")

        # --- Si no hay líneas ---
        if not lines:
            if isinstance(text_diff, dict) and "header_changed" in text_diff:
                before = text_diff["header_changed"].get("before", "")
                after = text_diff["header_changed"].get("after", "")
                return f"⚠️ Texto modificado: {before} → {after}"
            else:
                return "✅ Sin cambios visibles."

        return "\n".join(lines)


    # --- Construir lista de diffs ---
    diffs = []

    for row in rows:
        removed = safe_json_load(row[5])
        added = safe_json_load(row[6])
        modified = safe_json_load(row[7])
        text_diff = safe_json_load(row[8])

                # 🧠 Analizar el overlap de texto
        # Inicializar valores
        overlap_ratio = 0.0
        screen_status = "unknown"
        changes_list = []

        # Obtener overlap_ratio si existe en text_diff
        if isinstance(text_diff, dict):
            overlap_ratio = text_diff.get("overlap_ratio", 0.0)  # fallback a 0.0 si no existe

        # Determinar estado semántico considerando cambios estructurales
        if len(removed) == 0 and len(added) == 0 and len(modified) == 0:
            screen_status = "identical"      # No hay cambios → idéntica
        elif overlap_ratio >= 0.8:
            screen_status = "minor_changes"  # Cambios menores
        else:
            screen_status = "different"      # Cambios grandes o sin overlap suficiente

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
                if isinstance(vals, dict):
                    old_value = vals.get("old")
                    new_value = vals.get("new")
                else:
                    # Si vals es string (o cualquier otro tipo), lo tratamos como valor antiguo/nuevo
                    old_value = vals
                    new_value = vals

                detailed_changes.append({
                    "attribute": attr,
                    "old_value": old_value,
                    "new_value": new_value,
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


        for node in removed:
            changes_list.append(f"Removed: {node.get('class','unknown')} ({node.get('text','')})")

        for node in added:
            changes_list.append(f"Added: {node.get('class','unknown')} ({node.get('text','')})")

        for change in modified:
            node = change.get("node", {})
            changes = change.get("changes", {})
            if not changes:
                changes_list.append(f"Modified (no details): {node.get('class','unknown')} ({node.get('text','')})")
            for attr, vals in changes.items():
                if isinstance(vals, dict):
                    old = vals.get("old")
                    new = vals.get("new")
                else:
                    old = new = vals
                changes_list.append(f"Modified: {node.get('class','unknown')} {attr}: {old} → {new}")
    

        # update_diff_trace(
        #     tester_id=tester_id,
        #     build_id=build_id,
        #     screen=row[3],  # screen_name
        #     changes=changes_list
        # )
    

        # 🆕 Generar resumen elegante para cada diff
        summary_text = capture_pretty_summary(removed, added, modified, text_diff)

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
            "created_at": row[9],
            "cluster_info": json.loads(row[10]) if row[10] else {},
            "detailed_summary": summary_text  # 🆕 Nuevo campo
        })

    # ✅ Cálculo robusto de has_changes
    #print("DEBUG diffs:", diffs)
    # Tomamos has_changes directo de compare_trees
    for d in diffs:
        d["has_changes"] = any([
            len(d.get("removed", [])) > 0,
            len(d.get("added", [])) > 0,
            len(d.get("modified", [])) > 0,
            bool(d.get("text_diff", {}))
        ])

    # Si quieres un indicador global
    # has_changes = any(d["has_changes"] for d in diffs)
    has_changes = any(diff.get("has_changes", False) for diff in diffs)

    print(f"🧩 has_changes={has_changes} | total_diffs={len(diffs)}")

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
    

@router.post("/create_code")
async def create_code(usuario_id: str, duracion_dias: int = 30, usos_permitidos: int = 1):
    # Generar el código (formato similar al de Android)
    chars = string.ascii_uppercase + string.digits
    prefix_char = random.choice(chars)
    device_prefix = ''.join(random.choices("0123456789ABCDEF", k=2))
    random_part = str(random.randint(0, 9999)).zfill(4)
    codigo = f"{prefix_char}{device_prefix}{random_part}"

    generado_en = int(time.time())
    expira_en = generado_en + duracion_dias * 24 * 3600

    # Guardar en SQLite
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO login_codes     
            (codigo, usuario_id, generado_en, expira_en, usos_permitidos, usos_actuales, activo)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,  (codigo, usuario_id, generado_en, expira_en, usos_permitidos, 0, 1))
        conn.commit()

    return {"codigo": codigo, "expira_en": expira_en, "duracion_dias": duracion_dias}

@router.post("/validate_code")
async def validate_code(request: Request):
    data = await request.json()
    codigo = data.get("codigo", "").strip().upper()

    if not codigo:
        return {"valid": False, "reason": "Código vacío"}

    now = int(time.time())

    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        row = c.execute("""
            SELECT * FROM login_codes
            WHERE codigo = ? AND activo = 1
        """, (codigo,)).fetchone()

        if not row:
            return {"valid": False, "reason": "Código no encontrado"}

        # Validar expiración
        if now > row["expira_en"]:
            return {"valid": False, "reason": "Código expirado"}

        # Validar usos permitidos
        if row["usos_actuales"] >= row["usos_permitidos"]:
            return {"valid": False, "reason": "Límite de usos alcanzado"}

        # Incrementar uso
        c.execute("""
            UPDATE login_codes
            SET usos_actuales = usos_actuales + 1
            WHERE codigo = ?
        """, (codigo,))
        conn.commit()

        restante = row["expira_en"] - now

    return {
        "valid": True,
        "codigo": codigo,
        "usuario_id": row["usuario_id"],
        "expira_en": row["expira_en"],
        "restante_en_segundos": restante,
        "usos_restantes": row["usos_permitidos"] - (row["usos_actuales"] + 1)
    }

@router.post("/confirm_payment")
async def confirm_payment(request: Request):
    data = await request.json()
    codigo = data.get("codigo")
    payment_token = data.get("payment_token")  # del proveedor

    # 1️⃣ Validar que el código existe y no está activo
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        row = c.execute("SELECT * FROM login_codes WHERE codigo = ?", (codigo,)).fetchone()
        if not row:
            return {"success": False, "reason": "Código no encontrado"}
        if row["activo"] == 1:
            return {"success": False, "reason": "Código ya pagado"}

    # 2️⃣ Verificar pago con proveedor externo (Stripe, PayPal, etc.)
    pago_exitoso = verificar_pago_externo(payment_token)  # función que implementas según el proveedor

    if pago_exitoso:
        # 3️⃣ Activar el código
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("UPDATE login_codes SET activo = 1 WHERE codigo = ?", (codigo,))
            conn.commit()
        return {"success": True, "codigo": codigo}
    else:
        return {"success": False, "reason": "Pago no validado"}


    

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


@app.get("/metrics/changes")
async def get_change_metrics(tester_id: str = None):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        q = """
            SELECT tester_id, build_id, total_events, total_changes,
                   total_added, total_removed, total_modified,
                   ROUND(100.0 * total_changes / total_events, 2) AS change_rate,
                   last_updated
            FROM metrics_changes
        """
        if tester_id:
            q += " WHERE tester_id = ? ORDER BY last_updated DESC"
            rows = cur.execute(q, (tester_id,)).fetchall()
        else:
            q += " ORDER BY last_updated DESC"
            rows = cur.execute(q).fetchall()
        return {"metrics": [dict(r) for r in rows]}


