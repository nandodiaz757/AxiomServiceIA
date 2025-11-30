from fastapi import FastAPI, Query, BackgroundTasks, Request, APIRouter, HTTPException, Depends, WebSocket, Body, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, ConfigDict
import sqlite3, json, numpy as np, os, hashlib, logging, asyncio, re, unicodedata
from joblib import dump, load
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
from diff_model.predict_diff import router as diff_router
from datetime import datetime, timedelta  
from contextlib import asynccontextmanager
from email_service import send_email
from reset_service import generate_code, validate_code
import random, time
from PIL import Image
import asyncio
from collections import Counter
import math
import uuid
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
from contextvars import ContextVar
from slugify import slugify
import logging


from stable_signature import normalize_node

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)



# 🔹 NUEVA IMPORTACIÓN: ConfigManager para gestión centralizada
from config_manager import get_config, init_config

# 🔹 NUEVA IMPORTACIÓN: Sistema de retroalimentación incremental
try:
    from incremental_feedback_system import (
        IncrementalFeedbackSystem,
        check_approved_diff_pattern,
        record_diff_decision
    )
except ImportError as e:
    logger.warning(f"⚠️  No se pudo importar incremental_feedback_system: {e}")

# 🔹 NUEVA IMPORTACIÓN: Motor de análisis de flujos
try:
    from FlowAnalyticsEngine import FlowAnalyticsEngine
    logger.info("✅ FlowAnalyticsEngine importado correctamente")
except ImportError as e:
    logger.warning(f"⚠️  No se pudo importar FlowAnalyticsEngine: {e}")
    FlowAnalyticsEngine = None

# =========================================================
# CONFIGURACIÓN
# =========================================================

app = FastAPI()
router = APIRouter() 
siamese_model = None
SIM_THRESHOLD = 0.90
FLOW_MODEL_DIR = "models/flows"
FLOW_MODELS = {}
# Cargamos modelo siamés (una vez)
last_train: dict[str, float] = {}

event_queue: asyncio.Queue = asyncio.Queue()
BATCH_SIZE = 10           # número máximo de eventos por batch
BATCH_INTERVAL = 2        # tiempo máximo de espera en segundos antes de procesar batch
last_processed = {}
DEBOUNCE_TIME = 2

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

class TrainModeRequest(BaseModel):
    train_general: bool
    app_name: str
    tester_id: str | None = None
    build_id: str | None = None
    screen_id: str | None = None
    enriched_vector: list[float] | None = None


semantic_screen_id_ctx = ContextVar("screen_id_final", default=None)
semantic_screen_id_ctf = ContextVar("screen_id_final", default=None)

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

class BaselineMarkRequest(BaseModel):
    app_name: str
    tester_id: str
    build_id: str    

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
        #vec = torch.tensor(self.tree_to_vector(ui_tree), dtype=torch.float32)
        vec = torch.from_numpy(np.array(self.tree_to_vector(ui_tree), dtype=np.float32))
        with torch.no_grad():
            emb = self.encoder(vec)
            emb = F.normalize(emb, p=2, dim=0)  # normaliza L2
        return emb
    
    def load_app_flows(app_name: str):
        path = os.path.join(FLOW_MODEL_DIR, f"{app_name}_flows.joblib")
        if os.path.exists(path):
            FLOW_MODELS[app_name] = load(path)
            logger.info(f"💾 Modelo de flujos cargado en memoria para {app_name}")
        else:
            FLOW_MODELS[app_name] = {}
            logger.info(f"⚠️ No hay modelo previo para {app_name}, se inicia vacío")

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
    
def extract_model_key(semantic_screen_id: str) -> str:
    """
    Devuelve una key corta y safe para usar como nombre de carpeta.
    - Toma solo 1 o 2 primeras palabras del root semántico.
    - Quita flags.
    - Limpia caracteres ilegales en Windows.
    """
    # 1. separar por "_"
    parts = semantic_screen_id.split("_")

    # 2. cortar donde empiezan los flags (cuando detectamos '=')
    clean_parts = []
    for p in parts:
        if "=" in p:  # flag detectado
            break
        clean_parts.append(p)

    # 3. mantener máximo 2 palabras de contexto
    clean_parts = clean_parts[:2]

    # fallback si está vacío
    if not clean_parts:
        clean_parts = ["screen"]

    model_key = "_".join(clean_parts)

    # 4. limpiar caracteres ilegales para Windows
    illegal = '<>:"/\\|?*'

    for ch in illegal:
        model_key = model_key.replace(ch, "")

    return model_key


def extract_model_key(semantic_screen_id: str) -> str:
    """
    Devuelve una key corta y safe para usar como nombre de carpeta.
    - Toma solo 1 o 2 primeras palabras del root semántico.
    - Quita flags.
    - Limpia caracteres ilegales en Windows.
    """
    # 1. separar por "_"
    parts = semantic_screen_id.split("_")

    # 2. cortar donde empiezan los flags (cuando detectamos '=')
    clean_parts = []
    for p in parts:
        if "=" in p:  # flag detectado
            break
        clean_parts.append(p)

    # 3. mantener máximo 2 palabras de contexto
    clean_parts = clean_parts[:2]

    # fallback si está vacío
    if not clean_parts:
        clean_parts = ["screen"]

    model_key = "_".join(clean_parts)

    # 4. limpiar caracteres ilegales para Windows
    illegal = '<>:"/\\|?*'

    for ch in illegal:
        model_key = model_key.replace(ch, "")

    return model_key


def extract_short_screen_id(semantic_screen_id: str, max_base_len: int = 20) -> str:
    """
    Genera un ID corto y seguro.
    Formato: <shortname>_<hash8>
    Siempre corto, siempre estable.
    """

    if not semantic_screen_id:
        return "unknown_" + hashlib.md5("empty".encode()).hexdigest()[:8]

    # --- 1. Tomar solo letras y números del inicio ---
    base = re.findall(r'[a-zA-Z0-9]+', semantic_screen_id)
    base = base[0] if base else "screen"

    # --- 2. Limitar longitud máxima del nombre base ---
    base = base[:max_base_len]

    # --- 3. Hash estable de TODO el semantic id ---
    h = hashlib.md5(semantic_screen_id.encode()).hexdigest()[:8]

    # --- 4. Construir id final ---
    return f"{base}_{h}"

def is_baseline_build(app_name, tester_id, build_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT 1 FROM baseline_metadata
        WHERE app_name = ? AND tester_id = ? AND build_id = ?
        LIMIT 1
    """, (app_name, tester_id, build_id))
    row = c.fetchone()
    conn.close()
    return row is not None

def sanitize_screen_id_for_fs(screen_id: str) -> str:
    """Sanitiza el screen_id para usarlo en paths del file system."""
    import re, hashlib
        
    # 1. Remover caracteres ilegales para Windows
    screen_id = re.sub(r'[<>:"/\\|?*]', '_', screen_id)

    # 2. Limitar longitud (Windows falla >240 chars de path)
    if len(screen_id) > 120:  # margen seguro
        h = hashlib.sha1(screen_id.encode()).hexdigest()[:8]
        screen_id = screen_id[:110] + "_h" + h

    return screen_id


# =====================================================
# Sanitizador para filesystem (nuevo en versión 3.0)
# =====================================================
def sanitize_for_fs(name: str) -> str:
    """
    Convierte un screen_id en un nombre seguro para usar como nombre de archivo.
    - Reemplaza caracteres no permitidos
    - Limita longitud
    """
    if not name:
        return "unknown"

    # Solo permitir letras, números, guion y underscore.
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name)

    # Truncar si es demasiado largo (Windows tiene límite de path)
    return safe[:120]

# =====================================================
# Funciones auxiliares previas: slugify / dynamic / canonical
# =====================================================

NOISE_PREFIXES = {"¡", "!", "(", "["}
DYNAMIC_PATTERNS = [
    r"https?://",
    r"\d{2,}",
    r"%",
]

def slugify(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/_-]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text

def looks_dynamic(txt):
    if not txt:
        return False
    t = txt.lower()
    if any(t.startswith(p) for p in NOISE_PREFIXES):
        return True
    for pat in DYNAMIC_PATTERNS:
        if re.search(pat, t):
            return True
    return False
        # ...existing code...
# =====================================================
# Canonical mapping
# =====================================================
CANONICAL = {
    "home": {"home", "inicio", "home_page"},
    "account": {"account", "cuenta", "perfil", "profile"},
    "favorites": {"favorites", "favoritos", "mis_favoritos"},
    "offers": {"offers", "ofertas", "promociones"},
    "search": {"search", "buscar", "busqueda"},
    "cart": {"cart", "carrito", "checkout", "pagar"},
    "orders": {"orders", "pedidos", "mis_pedidos"},
    "login": {"login", "iniciar_sesion", "sign_in"},
    "settings": {"settings", "ajustes", "configuracion"},
    "social": {"social", "sociales"},
    "help": {"help", "ayuda", "soporte"},
}

_CANON_LOOKUP = {}
for k, s in CANONICAL.items():
    for v in s:
        _CANON_LOOKUP[v] = k

def canonical_map(text):
    if not text:
        return None
    t = slugify(text).replace("_", "")
    for k in _CANON_LOOKUP:
        if k in t:
            return _CANON_LOOKUP[k]
    return None

# =====================================================
# Plantilla de features (igual que antes)
# =====================================================
DEFAULT_FEATURES = {
    "Button": 0, "MaterialButton": 0, "ImageButton": 0, "EditText": 0,
    "TextView": 0, "ImageView": 0, "CheckBox": 0, "RadioButton": 0, "Switch": 0,
    "Spinner": 0, "SeekBar": 0, "ProgressBar": 0, "RecyclerView": 0,
    "ListView": 0, "ScrollView": 0, "LinearLayout": 0, "RelativeLayout": 0,
    "ConstraintLayout": 0, "FrameLayout": 0, "CardView": 0,
    "ComposeView": 0, "Text": 0, "ButtonComposable": 0,
    "RCTView": 0, "RCTText": 0, "RCTImageView": 0,
    "FlutterView": 0, "WebView": 0,
    "IonContent": 0, "IonButton": 0, "IonInput": 0,
}

def looks_dynamic(text: str) -> bool:
    if not text:
        return False
    # Detectar fechas, horas, contadores, temporizadores, cantidades cambiantes
    return any(ch.isdigit() for ch in text) and len(text) >= 4


def canonical_map(text: str):
    # Mapa de equivalencias para tabs o headers comunes
    MAP = {
        "home": "home",
        "inicio": "home",
        "profile": "profile",
        "perfil": "profile",
        "settings": "settings",
        "ajustes": "settings",
        "menu": "menu",
        "more": "menu"
    }
    t = slugify(text or "").lower()
    return MAP.get(t)


def sanitize_for_fs(text: str) -> str:
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad else c for c in text)

# =====================================================
# SCREEN-ID GENERATOR v4.0 (estable y compacto)
# =====================================================
def build_screen_id(raw_nodes, *, return_components=False):
    if not raw_nodes:
        sid = "unknown"
        return (sid, sanitize_for_fs(sid), {}) if return_components else sid

    # -------------------------------
    # (1) Recolección de features
    # -------------------------------
    texts = []
    nav = []
    ui = {"tabs": False, "bottomnav": False, "modal": False, "form": False,
          "list": False, "scroll": False}
    frameworks = {"compose": False, "flutter": False, "reactnative": False,
                  "webview": False, "ionic": False, "native": False}

    for idx, node in enumerate(raw_nodes):
        cls = (node.get("className") or "").lower()
        sig = (node.get("signature") or "").lower()
        txt = (node.get("text") or node.get("desc") or "") or ""
        txt = txt.strip()

        # Textos candidatos (no dinámicos)
        if txt and not looks_dynamic(txt):
            texts.append((idx, txt))

        # Navegación inferior / tabs
        slug = slugify(txt).replace("_", "")
        if slug in {"home", "profile", "settings", "menu"}:
            nav.append((idx, txt))

        # Framework detector
        if "compose" in cls:
            frameworks["compose"] = True
        if "flutter" in cls or "flutter" in sig:
            frameworks["flutter"] = True
        if "rct" in cls or "reactnative" in sig:
            frameworks["reactnative"] = True
        if "webview" in cls:
            frameworks["webview"] = True
        if "ion" in cls:
            frameworks["ionic"] = True

        # UI patterns
        if any(k in cls for k in ["scroll", "lazycolumn", "recycler", "listview"]):
            ui["scroll"] = True
        if "edittext" in cls or "input" in cls:
            ui["form"] = True
        if "dialog" in cls or "alert" in cls:
            ui["modal"] = True
        if "tab" in cls:
            ui["tabs"] = True
        if "recycler" in cls or "listview" in cls:
            ui["list"] = True

    # Fallback -> native
    if not any(frameworks.values()):
        frameworks["native"] = True

    # Bottom nav detection
    bottom_tab = None
    if len(nav) >= 3:
        ui["bottomnav"] = True
        bottom_tab = nav[-1][1]

    # -------------------------------
    # (2) Header semántico estable
    # -------------------------------
    header = None
    if texts:
        # CSR (candidate stable ranking)
        scored = []
        for idx, txt in texts:
            score = 0
            score += max(0, 10 - idx // 20)  # más arriba = más score
            score += min(5, len(txt) // 10)  # textos más largos valen más
            scored.append((score, txt))
        scored.sort(reverse=True)
        header = scored[0][1]

    if not header:
        header = bottom_tab or "screen"

    sem_key = canonical_map(header) or slugify(header)

    # -------------------------------
    # (3) Hierarchical path
    # -------------------------------
    parts = []

    if ui["bottomnav"] and bottom_tab:
        bcanon = canonical_map(bottom_tab) or slugify(bottom_tab)
        parts.append(bcanon)
        if sem_key != bcanon:
            parts.append(sem_key)
    else:
        parts.append(sem_key)

    if ui["list"]:
        parts.append("list")
    elif ui["form"]:
        parts.append("form")

    hierarchical = "/".join(parts)

    # -------------------------------
    # (4) Firmas de flags
    # -------------------------------
    ui_flags = "|".join(f"{k}={str(ui[k]).lower()}" for k in ui)
    fw_flags = "|".join(f"{k}={str(frameworks[k]).lower()}" for k in frameworks)

    signature = f"{hierarchical}|{ui_flags}|{fw_flags}"
    short_hash = hashlib.sha1(signature.encode()).hexdigest()[:8]

    screen_id = f"{hierarchical}|{ui_flags}|{fw_flags}|h={short_hash}"
    screen_id_fs = sanitize_for_fs(screen_id)

    if not return_components:
        return screen_id

    return (
        screen_id,
        screen_id_fs,
        {
            "header": header,
            "semantic_key": sem_key,
            "hierarchical": hierarchical,
            "ui": ui,
            "frameworks": frameworks,
            "canonical_signature": signature,
            "short_hash": short_hash,
        }
    )

def clean_old_diff_records(days: int = 90):
    cutoff = datetime.utcnow() - timedelta(days=days)
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
    logger.info(f"🧹 Limpieza de registros antiguos completada (>{days} días).")


def load_siamese_model(path: str = "ui_encoder.pt"):
    global siamese_model
    from SiameseEncoder import SiameseEncoder
    siamese_model = SiameseEncoder.load(path)
    siamese_model.eval()
    print("✅ Modelo siamés cargado en memoria.")

    
def retrain_diff_model():
    from diff_model.train_diff_model import train_and_save
    #print("🧠 Iniciando reentrenamiento del modelo siamés...")
    train_and_save()
    load_siamese_model()  # recarga el modelo actualizado
    #print("✅ Reentrenamiento completado y modelo actualizado en memoria.")


# ==============================
# TAREAS PERIÓDICAS
# ==============================

#@repeat_every(seconds=3600, wait_first=True)
@repeat_every(seconds=3600, wait_first=False)
async def periodic_retrain():
    lock = _get_lock("diff_model")
    
    """Reentrenamiento del modelo cada hora."""
    async with lock:
        await asyncio.to_thread(retrain_diff_model)


@repeat_every(seconds=86400, wait_first=True)
async def daily_cleanup():
    """Limpieza de registros cada 24 horas."""
    await asyncio.to_thread(clean_old_diff_records, days=90)


# ==========================================
# LIFESPAN HANDLER (startup + shutdown)
# ==========================================
# 🔹 Variable global para feedback system
feedback_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    global feedback_system
    logger.info("🚀 Iniciando FastAPI y cargando modelo siamés...")
    load_siamese_model()
    
    # 🔹 Inicializar sistema de retroalimentación
    try:
        feedback_system = IncrementalFeedbackSystem(db_name="feedback_model.db")
        logger.info("✅ Sistema de retroalimentación incremental inicializado")
    except Exception as e:
        logger.warning(f"⚠️  No se pudo inicializar feedback_system: {e}")
        feedback_system = None

    yield  # Aquí la app ya está corriendo

    # --- Shutdown ---
    logger.info("🛑 Apagando aplicación y limpiando tareas...")
    # worker_task.cancel()


# ==============================
# APP PRINCIPAL
# ==============================

app = FastAPI(lifespan=lifespan)

# 🔹 INICIALIZAR CONFIG MANAGER AL STARTUP
config = init_config()


def send_slack_alert(webhook_url: Optional[str] = None, title: str = "", payload: dict = None):
    """
    Envía una alerta formateada a Slack usando incoming webhook.
    Si webhook_url no se proporciona, intenta obtenerlo de la configuración.
    Incluye retry automático basado en config.
    """
    if payload is None:
        payload = {}
    
    # Obtener webhook de config si no se proporciona
    if not webhook_url:
        webhook_url = config.get_webhook_url("slack")
    
    # Verificar si Slack está habilitado
    if not config.is_notification_enabled("slack") or not webhook_url:
        logger.info("⚠️ Slack notifications disabled or webhook URL not configured")
        return False
    
    try:
        msg = {
            "text": f"*{title}*\nTester: {payload.get('tester_id')}  Build: {payload.get('build_id')}  Severity: {payload.get('severity', 0):.2f}",
            "attachments": [
                {
                    "fields": [
                        {"title": "Severity", "value": f"{payload.get('severity', 0):.2f}", "short": True},
                        {"title": "Diffs", "value": str(payload.get('diff_count', 0)), "short": True},
                        {"title": "Timestamp", "value": str(datetime.now().isoformat()), "short": True}
                    ],
                    "color": "danger" if payload.get('severity', 0) > 0.7 else "warning"
                }
            ]
        }

        import requests
        headers = {"Content-Type": "application/json"}
        timeout = config.get("notifications.slack.timeout", 5)
        retry_count = config.get("notifications.slack.retry_count", 2)
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    webhook_url,
                    headers=headers,
                    data=json.dumps(msg),
                    timeout=timeout
                )
                if response.status_code == 200:
                    logger.info(f"✅ Slack alert sent successfully")
                    return True
                else:
                    logger.warning(f"⚠️ Slack alert failed: {response.status_code}")
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.info(f"🔄 Retrying Slack alert ({attempt + 1}/{retry_count})...")
                    time.sleep(1)
                else:
                    raise
        
        return False
        
    except Exception as e:
        logger.exception(f"❌ Error sending Slack alert: {e}")
        return False


def send_teams_alert(webhook_url: Optional[str] = None, title: str = "", payload: dict = None):
    """
    Envía una Adaptive Card a Microsoft Teams usando incoming webhook.
    Si webhook_url no se proporciona, intenta obtenerlo de la configuración.
    Incluye retry automático basado en config.
    """
    if payload is None:
        payload = {}
    
    # Obtener webhook de config si no se proporciona
    if not webhook_url:
        webhook_url = config.get_webhook_url("teams")
    
    # Verificar si Teams está habilitado
    if not config.is_notification_enabled("teams") or not webhook_url:
        logger.info("⚠️ Teams notifications disabled or webhook URL not configured")
        return False
    
    try:
        severity = payload.get('severity', 0)
        theme_color = "E81123" if severity > 0.7 else "FFB900"  # Rojo o Ámbar
        
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": title,
            "themeColor": theme_color,
            "sections": [
                {
                    "activityTitle": title,
                    "activitySubtitle": f"Timestamp: {datetime.now().isoformat()}",
                    "facts": [
                        {"name": "Tester", "value": str(payload.get('tester_id', 'N/A'))},
                        {"name": "Build", "value": str(payload.get('build_id', 'N/A'))},
                        {"name": "Severity", "value": f"{severity:.2f}"},
                        {"name": "Diffs", "value": str(payload.get('diff_count', 0))}
                    ]
                }
            ]
        }

        import requests
        headers = {"Content-Type": "application/json"}
        timeout = config.get("notifications.teams.timeout", 5)
        retry_count = config.get("notifications.teams.retry_count", 2)
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    webhook_url,
                    headers=headers,
                    data=json.dumps(card),
                    timeout=timeout
                )
                if response.status_code == 200:
                    logger.info(f"✅ Teams alert sent successfully")
                    return True
                else:
                    logger.warning(f"⚠️ Teams alert failed: {response.status_code}")
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.info(f"🔄 Retrying Teams alert ({attempt + 1}/{retry_count})...")
                    time.sleep(1)
                else:
                    raise
        
        return False
        
    except Exception as e:
        logger.exception(f"❌ Error sending Teams alert: {e}")
        return False


def send_jira_issue(
    project_key: Optional[str] = None,
    summary: str = "",
    description: str = "",
    issue_type: Optional[str] = None
):
    """
    Crea un issue en Jira usando REST API.
    Obtiene credenciales de la configuración o variables de entorno.
    Incluye retry automático basado en config.
    """
    # Obtener configuración
    jira_base_url = config.get("notifications.jira.base_url") or os.environ.get('JIRA_BASE_URL')
    jira_token = config.get("notifications.jira.api_token") or os.environ.get('JIRA_API_TOKEN')
    project_key = project_key or config.get("notifications.jira.project_key", "QA")
    issue_type = issue_type or config.get("notifications.jira.issue_type", "Task")
    
    # Verificar si Jira está habilitado
    if not config.is_notification_enabled("jira") or not jira_base_url or not jira_token:
        logger.info("⚠️ Jira notifications disabled or credentials not configured")
        return None

    try:
        api_endpoint = jira_base_url.rstrip('/') + '/rest/api/3/issue'
        auth = ('token', jira_token) if jira_token else None

        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type}
            }
        }

        import requests
        headers = {"Content-Type": "application/json"}
        timeout = config.get("notifications.jira.timeout", 10)
        retry_count = config.get("notifications.jira.retry_count", 2)
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    api_endpoint,
                    auth=auth,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout
                )
                if response.status_code in (200, 201):
                    data = response.json()
                    issue_key = data.get('key')
                    logger.info(f"✅ Created Jira issue {issue_key}")
                    return issue_key
                else:
                    logger.warning(f"⚠️ Jira API error: {response.status_code}")
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.info(f"🔄 Retrying Jira issue creation ({attempt + 1}/{retry_count})...")
                    time.sleep(1)
                else:
                    raise
        
        return None
        
    except Exception as e:
        logger.exception(f"❌ Error creating Jira issue: {e}")
        return None


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
    
        # 🔹 Asegurar columnas nuevas para retroalimentación incremental (retrocompatible)
        try:
            c.execute("ALTER TABLE screen_diffs ADD COLUMN IF NOT EXISTS diff_priority TEXT DEFAULT 'high'")
            c.execute("ALTER TABLE screen_diffs ADD COLUMN IF NOT EXISTS similarity_to_approved REAL DEFAULT 0.0")
            c.execute("ALTER TABLE screen_diffs ADD COLUMN IF NOT EXISTS approved_before INTEGER DEFAULT 0")
        except Exception as e:
            logger.warning(f"⚠️  No se pudieron agregar columnas (puede que ya existan): {e}")

        conn.commit()
        conn.close()
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
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
            global_signature TEXT,
            partial_signature TEXT,
            scroll_type TEXT,
            signature TEXT,
            version TEXT,
            collect_node_tree TEXT,
            additional_info TEXT,
            tree_data TEXT,
            is_baseline INTEGER DEFAULT 0,   
            enriched_vector TEXT,
            cluster_id INTEGER,
            is_stable INTEGER DEFAULT 0,
            anomaly_score REAL,
            session_key TEXT,
            embedding_vector  TEXT, 
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
            diff_priority TEXT DEFAULT 'high',
            text_diff TEXT,  
            text_overlap REAL DEFAULT 0,
            overlap_ratio REAL DEFAULT 0,
            ui_structure_similarity REAL DEFAULT 0,      
            cluster_id INTEGER DEFAULT -1,  
            screen_status TEXT DEFAULT 'unknown',  
            similarity_to_approved REAL DEFAULT 0.0,
            approved_before INTEGER DEFAULT 0,
            approval_status TEXT,   
            rejection_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS metrics_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_group TEXT NOT NULL,   -- screen / build / global
            metric_name TEXT NOT NULL,
            metric_value TEXT
        )
    """)
    c.execute("""
		CREATE TABLE IF NOT EXISTS diff_items (
		  id INTEGER PRIMARY KEY,
		  diff_id INTEGER NOT NULL REFERENCES screen_diffs(id),
		  action TEXT NOT NULL,          
		  node_class TEXT,
		  node_key TEXT,
		  node_text TEXT,
		  changes_json TEXT,            
		  raw_json TEXT,               
		  label TEXT,                   
		  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);
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
        CREATE TABLE IF NOT EXISTS baseline_metadata (
            app_name TEXT,
            tester_id TEXT,
            build_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (app_name, tester_id, build_id)
        );
    """)
    # Crear la tabla
    c.execute("""
    CREATE TABLE IF NOT EXISTS active_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  -- <--- aquí va el nombre
        plan_name TEXT UNIQUE NOT NULL,
        description TEXT NOT NULL,
        active INTEGER DEFAULT 1,
        currency TEXT CHECK(currency IN ('USD', 'COP', 'EUR')) NOT NULL DEFAULT 'USD',
        rate_usd REAL DEFAULT 1.0,       
        rate_cop REAL DEFAULT 4100.0,   
        rate_eur REAL DEFAULT 0.94,      
        rate REAL DEFAULT 1.0,
        price REAL NOT NULL,
        no_associate REAL DEFAULT 0.0,  -- costo por código sin plan activo
        max_tester INTEGER NOT NULL,      
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Insertar los planes por defecto
    default_plans = [
        ('BASIC', 'Basic Plan', 'USD', 1.0, 1, 1),
        ('STANDARD', 'Standard Plan', 'USD', 1.0, 25, 5)
    ]

    for plan in default_plans:
        c.execute("""
            INSERT OR IGNORE INTO active_plans (plan_name, description, currency, rate, price, max_tester)
            VALUES (?, ?, ?, ?, ?, ?)
        """, plan)


    c.execute("""
        CREATE TABLE IF NOT EXISTS login_codes (
            codigo TEXT PRIMARY KEY,
            usuario_id TEXT NOT NULL,
            plan_id INTEGER,  
            generado_en INTEGER NOT NULL,
            expira_en INTEGER NOT NULL,
            usos_permitidos INTEGER NOT NULL,
            usos_actuales INTEGER DEFAULT 0,
            pago_confirmado INTEGER DEFAULT 0,  
            activo INTEGER DEFAULT 1,  
            is_paid INTEGER DEFAULT 0,
            FOREIGN KEY (plan_id) REFERENCES active_plans(id)  
        );
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id TEXT PRIMARY KEY,
            nombre TEXT,
            rol TEXT CHECK(rol IN ('owner', 'tester')) DEFAULT 'tester',
            plan_id INTEGER,
            FOREIGN KEY(plan_id) REFERENCES active_plans(id)
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
            rejection_reason TEXT, 
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
    
    collect_node_tree: Optional[List[Dict[str, Any]]] = Field(
    None, alias="collectNodeTree"
    )
    
    # Datos adicionales para enriquecer el modelo (libres)
    additional_info: Optional[Dict[str, Any]] = Field(None, alias="additionalInfo")
    tree_data: Optional[Dict[str, Any]] = Field(None, alias="treeData")
    session_key: Optional[str] = None

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


def build_stable_list(tree):
    """Aplana el árbol en una lista ordenada con firmas estables."""
    result = []

    def walk(node, depth=0):
        if not isinstance(node, dict):
            return

        cls = node.get("className", "")
        text = node.get("text") or ""
        desc = node.get("desc") or ""
        viewId = node.get("viewId") or ""

        # firma estable del nodo
        signature = f"{depth}|{cls}|{viewId}|{text or desc}"

        result.append(signature)

        children = node.get("children") or []
        for child in children:
            walk(child, depth + 1)

    if isinstance(tree, list):
        for n in tree:
            walk(n)
    else:
        walk(tree)

    return result


def compare_ui_orders(tree_a, tree_b):
    """Compara el orden relativo y detecta reordenamientos sin usar coordenadas."""
    a_list = build_stable_list(tree_a)
    b_list = build_stable_list(tree_b)

    sm = difflib.SequenceMatcher(None, a_list, b_list)

    result = {
        "missing_in_b": [],
        "new_in_b": [],
        "reordered": [],
        "same_structure_score": sm.ratio()
    }

    # detectar elementos que faltan
    for x in a_list:
        if x not in b_list:
            result["missing_in_b"].append(x)

    # detectar elementos nuevos
    for x in b_list:
        if x not in a_list:
            result["new_in_b"].append(x)

    # detectar reordenamientos
    opcodes = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "replace" or tag == "delete" or tag == "insert":
            result["reordered"].append({
                "from_a": a_list[i1:i2],
                "to_b": b_list[j1:j2]
            })

    return result


def detect_scroll_type(nodes: List[Dict]) -> str:
    for n in nodes:
        cls = n.get("className", "")
        if "ScrollView" in cls or "RecyclerView" in cls:
            return "vertical"
        if "HorizontalScrollView" in cls or "ViewPager" in cls:
            return "horizontal"
    return "none"

def stable_signatures(nodes: List[Dict], header_text: str = "") -> Dict[str, str]:
    """Genera firmas global y parcial para distinguir scrolls y pantallas."""
    if not nodes:
        return {"global_signature": "none", "partial_signature": "none"}

    norm = normalize_tree(nodes)
    root = norm[0] if isinstance(norm[0], dict) else {}

    # --- Global signature (identifica pantalla base) ---
    root_class = root.get("className", "")
    main_id = root.get("resourceId", "")
    global_data = f"{root_class}|{main_id}"
    global_signature = hashlib.sha256(global_data.encode("utf-8")).hexdigest()

    # --- Partial signature (lo visible actualmente + pantalla) ---
    visible_nodes = [n for n in norm if n.get("visibleToUser")]
    partial_data = {
        "header_text": header_text,  # <-- ahora incluido
        "nodes": visible_nodes
    }
    partial_signature = hashlib.sha256(
        json.dumps(partial_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return {
        "global_signature": global_signature,
        "partial_signature": partial_signature
    }
   

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
        #print(f"✅ Código guardado para {email}")
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


async def _safe_training_wrapper(
    X,
    tester_id: str,
    build_id: str,
    app_name: str,
    screen_id: str,
    desc: str,
    lock: asyncio.Lock,
    train_fn,   # <- función de entrenamiento a ejecutar
):
    """
    Wrapper seguro para realizar reentrenamientos evitando condiciones de carrera.
    Protege el entrenamiento con un lock global por modelo.
    """
    async with lock:  # 🔒 garantiza exclusividad de entrenamiento
        try:
            logger.warning(f"🔄 [{desc}] Entrenamiento iniciado para {app_name}/{tester_id}/{build_id}")

            # 👇 Ejecuta el entrenamiento real de forma async (por si es CPU-heavy)
            await train_fn(
                enriched_vector=X.flatten(),
                tester_id=tester_id,
                build_id=build_id,
                app_name=app_name,
                screen_id=screen_id,
            )

            logger.warning(f"✅ [{desc}] Entrenamiento completado para {app_name}/{tester_id}/{build_id}")

        except Exception as e:
            logger.error(f"❌ [{desc}] Error durante reentrenamiento: {e}", exc_info=True)


async def ensure_model_dimensions(
    kmeans,
    X,
    tester_id,
    build_id,
    app_name="default_app",
    screen_id="unknown_screen",
    desc=""
):
    try:
        # 🛑 1. Modelo aún no entrenado → entrenar ahora
        if not hasattr(kmeans, "cluster_centers_"):
            logger.warning(
                f"[{desc}] KMeans aún no entrenado — iniciando entrenamiento inicial"
            )

            lock_key = f"{app_name}:initial_train"
            lock = _get_lock(lock_key)

            if not lock.locked():
                asyncio.create_task(
                    _safe_training_wrapper(
                        X=X,
                        tester_id=tester_id,
                        build_id=build_id,
                        app_name=app_name,
                        screen_id=screen_id,
                        desc=f"initial_train {desc}",
                        lock=lock,
                        train_fn=_train_incremental_logic_hybrid,
                    )
                )
            return False

        # 🧩 2. Comparar dimensiones reales
        expected_features = kmeans.cluster_centers_.shape[1]
        current_features = X.shape[1]

        if current_features != expected_features:
            logger.warning(
                f"[{desc}] Dimensión inconsistente: modelo={expected_features}, nuevo={current_features}. Reentrenando..."
            )

            lock_key = f"{app_name}:diff_model"
            lock = _get_lock(lock_key)

            if lock.locked():
                logger.warning(
                    f"[{desc}] Skip: reentrenamiento ya en proceso para {lock_key}"
                )
                return False

            asyncio.create_task(
                _safe_training_wrapper(
                    X=X,
                    tester_id=tester_id,
                    build_id=build_id,
                    app_name=app_name,
                    screen_id=screen_id,
                    desc=f"retrain {desc}",
                    lock=lock,
                    train_fn=_train_incremental_logic_hybrid,
                )
            )

            return False

        # ✅ Todo bien
        return True

    except Exception as e:
        logger.warning(
            f"[{desc}] Error validando dimensiones ({app_name}/{tester_id}/{build_id}): {e}"
        )
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
        #print(f"[{label}] Textos extraídos ({len(texts)}): {texts}")
        return texts

    r_texts = extract_texts(old_nodes, "REMOVED")
    a_texts = extract_texts(new_nodes, "ADDED")

    # ✅ Manejo de casos vacíos para evitar falsos positivos y divisiones por cero
    if not r_texts and not a_texts:
        #print("⚪ Ambos sets vacíos → overlap = 1.0 (sin cambios detectados)")
        return 1.0
    elif not r_texts or not a_texts:
        #print("⚠️ Uno de los sets está vacío → overlap = 0.0 (sin elementos para comparar)")
        return 0.0

    common = r_texts.intersection(a_texts)
    total = r_texts.union(a_texts)
    overlap = len(common) / len(total)
    #print(f"➡️ Overlap: {len(common)}/{len(total)} = {overlap:.2f}")
    return overlap


# normalizar latest_tree: asegurar lista de dicts, eliminar Nones

def sanitize_tree(tree):
    if not tree:
        return []

    cleaned = []

    for n in tree:

        if isinstance(n, dict):
            # Aceptar nodos si tienen ALGO útil
            if any(n.get(k) for k in [
                "className", "text", "contentDescription", "desc", "hint"
            ]):
                cleaned.append(n)
            continue

        if isinstance(n, str):
            try:
                obj = json.loads(n)
                if isinstance(obj, dict):
                    if any(obj.get(k) for k in [
                        "className", "text", "contentDescription", "desc", "hint"
                    ]):
                        cleaned.append(obj)

                elif isinstance(obj, list):
                    cleaned.extend([
                        x for x in obj if isinstance(x, dict) and any(
                            x.get(k) for k in [
                                "className", "text", "contentDescription", "desc", "hint"
                            ]
                        )
                    ])
            except:
                continue

    return cleaned

async def analyze_and_train(event: AccessibilityEvent):
    # -------------------- Normalizar campos --------------------
    norm = _normalize_event_fields(event)
    t_id = str(norm.get("tester_id_norm") or "").strip()
    b_id = str(norm.get("build_id_norm") or "").strip()
    s_name = event.header_text 
    # s_name = normalize_header(event.header_text)
    event_type_ref = normalize_header(event.event_type_name)
    app_name = event.package_name or "default_app"
    tester_id = event.tester_id or "general"
    build_id = event.build_id
    header_text = event.header_text or ""
    global siamese_model
    flow_trees = FLOW_MODELS.get(app_name, {})
    raw_tree = event.collect_node_tree or event.tree_data or []
        # -------------------- Obtener snapshot previo ----------------
    prev_tree = None
    IGNORED_FIELDS = {"hint", "contentDescription", "value", "progress"}
   
    # -------------------- Árbol y firma ------------------------
    latest_tree = sanitize_tree(ensure_list(raw_tree))

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
        np.array([avg_dwell, num_gestos]),
        input_vec
    ])

    removed_all, added_all, modified_all = [], [], []
    text_diff = {}
    diff_result = {}

    # Valor por defecto, para evitar errores si hay excepciones antes de asignarlo
    has_changes = False  

    # -------------------- Verificación rápida de cambio de header --------------------
    try:
        curr_header = (s_name or "")
        # curr_header = (s_name or "").strip().lower()
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
                prev_id = row[0]          # ID del registro anterior
                prev_header = (row[1] or "").strip().lower()      # header_text
            else:
                prev_id = None
                prev_header = None

            # if row: 
            #     prev_header = (row[0] or "").strip().lower()


        # print(f"curr_header: '{curr_header}' | prev_header: '{prev_header}'")
        if prev_header and prev_header != curr_header:
            # print(f"⚠️ Header cambió (detección temprana): '{prev_header}' → '{curr_header}'")
            header_changed = {"before": prev_header, "after": curr_header}
            has_changes = True
        else:
            header_changed = None
            # print("✅ Header sin cambios (verificación temprana).")

    except Exception as e:
        header_changed = None
        logger.warning(f"⚠️ Error en verificación temprana de header_text: {e}")


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
            logger.exception(f"Error generando embedding: {e} -- usando vector neutro")
            # emb_curr = np.zeros((1, siamese_model.embedding_dim))

        # emb_curr = ensure_model_dimensions(emb_curr)
            # Fallback seguro
            try:
                emb_dim = getattr(siamese_model, "embedding_dim", 64)

            except Exception:
                emb_dim = 64 

            emb_curr = np.zeros((1, emb_dim), dtype=float)

            # 🔍 VALIDAR DIMENSIONES DEL MODELO
        valid_dimensions = await ensure_model_dimensions(
            kmeans=kmeans_model,
            X=emb_curr,
            tester_id=t_id,
            build_id=b_id,
            app_name=app_name,
            screen_id=semantic_screen_id_ctx.get(),
            desc="embedding_validation"
        )    

    # -------------------- (Opcional) Guardar embedding --------------------
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            UPDATE accessibility_data
            SET embedding_vector=?
            WHERE id=?
        """, (json.dumps(emb_curr.tolist()), prev_id))
        conn.commit()

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
            #print("ℹ️ No hay snapshots previos para comparar.")
            has_changes = True
            prev_tree = None
        else:
            # -------------------- 4. Calcular similitud aprendida --------------------
            best_sim, best_row = 0.0, None
            for row in prev_rows:
                try:
                    prev_tree = sanitize_tree(ensure_list(json.loads(row[0])))

                    #prev_tree = ensure_list(json.loads(row[0]))
                    with torch.no_grad():
                        emb_prev = siamese_model.encode_tree(prev_tree)
                    emb_prev = emb_prev.cpu().numpy().reshape(1, -1)

                    # Validar ambas dimensiones están alineadas
                    if emb_curr.shape[1] != emb_prev.shape[1]:
                        logger.warning(f"Dimensiones incompatibles: emb_curr={emb_curr.shape[1]}, emb_prev={emb_prev.shape[1]}")
                        continue

                    sim_torch = torch.nn.functional.cosine_similarity(
                        torch.tensor(emb_curr, dtype=torch.float32),
                        torch.tensor(emb_prev, dtype=torch.float32),
                        dim=1
                    )
                    print("Similitud (torch):", sim_torch.mean().item())

                    sim = float(cosine_similarity(emb_curr, emb_prev)[0][0])

                    if sim > best_sim:
                        best_sim, best_row = sim, row
                except Exception:
                    continue

            if best_sim > SIM_THRESHOLD and best_row:
                logger.info(f"🤝 Coincidencia detectada por modelo siamés (sim={best_sim:.3f})")
                #prev_tree = ensure_list(json.loads(best_row[0]))
                prev_tree = sanitize_tree(ensure_list(json.loads(best_row[0])))
            else:
                logger.warning("⚠️ No se encontró coincidencia fuerte, usar estructura directa.")
                prev_tree = ensure_list(json.loads(prev_rows[0][0]))    
    

    # esto es nuevo  tocar o eliminar  
    if prev_tree:
        try:
            def clean_nodes(tree):
                IGNORED = {"hint", "contentDescription", "value", "progress"}
                cleaned = []
                for n in tree:
                    if not isinstance(n, dict):
                        continue
                    c = {k: v for k, v in n.items() if k not in IGNORED}
                    # ignorar nodos vacíos o inválidos
                    if not c.get("className") and not c.get("text"):
                        continue
                    cleaned.append(c)
                return cleaned

            prev_tree_clean = clean_nodes(prev_tree)
            latest_tree_clean = clean_nodes(latest_tree)


            diff_result = compare_trees(
                prev_tree_clean,
                latest_tree_clean,
                app_name=app_name,
                tester_id=tester_id,
                build_id=build_id,
                screen_id=event.screens_id or event.header_text or "unknown_screen"
            )
            # --- (A) Similaridad estructural si no viene del diff ---
              # ------------------- STRUCTURE SIMILARITY -------------------
            if "structure_similarity" not in diff_result:
                try:
                    diff_result["structure_similarity"] = ui_structure_similarity(prev_tree, latest_tree)
                except Exception:
                    diff_result["structure_similarity"] = 1.0

            # ------------------- NUEVO: DIFF DE ORDEN -------------------
            try:
                order_info = compare_ui_orders(prev_tree, latest_tree)

                diff_result["order_missing"] = order_info["missing_in_b"]
                diff_result["order_new"] = order_info["new_in_b"]
                diff_result["order_reordered"] = order_info["reordered"]
                diff_result["order_score"] = order_info["same_structure_score"]

            except Exception as e:
                logger.error(f"Error comparando orden UI: {e}")
                diff_result["order_score"] = 1.0


            # has_changes = bool(
            #     diff_result.get("removed") or diff_result.get("added") or diff_result.get("modified") or
            #     diff_result.get("text_diff", {}).get("overlap_ratio", 1.0) < 0.9 or
            #     diff_result.get("has_changes")
            # )

            # AHORA SÍ: incluir orden en el análisis final
            has_changes = bool(
                diff_result.get("removed")
                or diff_result.get("added")
                or diff_result.get("modified")
                or diff_result.get("order_missing")
                or diff_result.get("order_new")
                or diff_result.get("order_reordered")
                or diff_result.get("text_diff", {}).get("overlap_ratio", 1.0) < 0.9
                or diff_result.get("structure_similarity", 1.0) < 0.9
                or diff_result.get("order_score", 1.0) < 0.9
                or diff_result.get("has_changes")
            )
            
            # 🔹 NUEVA SECCIÓN: Retroalimentación Incremental
            mark_as_low_priority = False
            similarity_to_approved = 0.0
            
            if has_changes and feedback_system is not None:
                try:
                    approval_info = check_approved_diff_pattern(
                        diff_signature=diff_signature,
                        app_name=app_name,
                        tester_id=t_id,
                        feedback_system=feedback_system
                    )
                    logger.info(f"📊 Análisis de retroalimentación: {approval_info['reason']}")
                    
                    # Guardar el estado para usar al insertar
                    mark_as_low_priority = not approval_info['should_show']
                    similarity_to_approved = approval_info['similarity_score']
                except Exception as e:
                    logger.warning(f"⚠️  Error en check_approved_diff_pattern: {e}")
                    mark_as_low_priority = False
                    similarity_to_approved = 0.0

        except Exception as e:
            logger.error(f"Error comparando árboles: {e}")
            has_changes = True
            mark_as_low_priority = False  # 🔹 NUEVA LÍNEA
            similarity_to_approved = 0.0   # 🔹 NUEVA LÍNEA

    else:
        has_changes = True
        mark_as_low_priority = False  # 🔹 NUEVA LÍNEA
        similarity_to_approved = 0.0   # 🔹 NUEVA LÍNEA


    # 🔹 Recalcular enriched_vector (después de emb_curr y fuera del if)
    # enriched_vector = np.concatenate([
    #     struct_vec,
    #     sig_vec,
    #     np.array([avg_dwell, num_gestos], dtype=float),
    #     input_vec,
    #     emb_curr.flatten()
    # ])  

    if (
        enriched_vector is not None 
        and hasattr(enriched_vector, "size") 
        and enriched_vector.size > 0
    ):    

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
    asyncio.create_task(_train_incremental_logic_hybrid(
        enriched_vector=enriched_vector,
        tester_id=tester_id,
        build_id=build_id,
        app_name=app_name,
        #screen_id=event.screens_id or s_name or "unknown_screen",
        screen_id=semantic_screen_id_ctx.get(),
        use_general_as_base=True
    ))


    if TRAIN_GENERAL_ON_COLLECT:
        await _train_general_logic_hybrid(
            app_name=app_name,
            batch_size=500,
            min_samples=3,
            update_general=True
        )

         # -------------------- Entrenamiento general con intervalo --------------------
    if TRAIN_GENERAL_ON_COLLECT:
        global last_train
        now = time.time()
        key = f"{app_name}_{semantic_screen_id_ctx.get()}"
        last = last_train.get(key, 0)
        MIN_INTERVAL = 300  # 5 minutos entre entrenamientos generales

        if now - last > MIN_INTERVAL:
            last_train[key] = now
            await _train_general_logic_hybrid(
                app_name=app_name,
                batch_size=500,
                min_samples=3,
                update_general=True
            )
        else:
            logger.debug(f"⏳ Skip general training for {key}, last run {now - last:.1f}s ago")


# -------------------- 9. Guardar en screen_diffs --------------------
    try:
        if has_changes:

            removed_all = diff_result.get("removed", [])
            added_all = diff_result.get("added", [])
            modified_all = diff_result.get("modified", [])
            text_diff = diff_result.get("text_diff", {})



            # --- mantener lo tuyo ---
            header_changed = text_diff.get("header_changed", None)

            diff_vector = np.array([
                len(removed_all),
                len(added_all),
                len(modified_all),
                text_diff.get("overlap_ratio", 1.0),
                diff_result.get("structure_similarity", 1.0)
            ])

            if "structure_similarity" not in diff_result:
                try:
                    diff_result["structure_similarity"] = ui_structure_similarity(prev_tree, latest_tree)
                except Exception:
                    diff_result["structure_similarity"] = 1.0  

            # --- Generar firma hash del diff (como tenías) ---
            diff_signature = diff_hash(removed_all, added_all, modified_all, text_diff)

            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()

                # Verificar si YA existe este diff_hash
                cur.execute("""
                    SELECT id FROM screen_diffs WHERE diff_hash = ?
                """, (diff_signature,))
                existing = cur.fetchone()

                # if existing:
                #     logger.info(f"⚠️ Diff {diff_signature[:8]} ya existe — NO insertamos screen_diffs NI diff_items")
                #     return   # ⬅️ AQUÍ SE CORTA TODO
                
                if existing:
                    logger.info(f"⚠️ Diff {diff_signature[:8]} ya existe — NO insertamos screen_diffs NI diff_items")
                    # devolver métricas coherentes
                    added_count = len(added_all) if 'added_all' in locals() else 0
                    removed_count = len(removed_all) if 'removed_all' in locals() else 0
                    modified_count = len(modified_all) if 'modified_all' in locals() else 0
                    return has_changes, added_count, removed_count, modified_count

                # ---- calcular valores ----
                text_overlap = text_diff.get("overlap_ratio", 1.0)
                ui_sim = diff_result.get("structure_similarity", 1.0)

                # estado textual
                if text_overlap >= 0.9 and ui_sim >= 0.9:
                    screen_status = "identical"
                elif text_overlap >= 0.6 and ui_sim >= 0.6:
                    screen_status = "minor_changes"
                else:
                    screen_status = "different"

            
                # serializar
                removed_j = json.dumps(removed_all, ensure_ascii=False)
                added_j = json.dumps(added_all, ensure_ascii=False)
                modified_j = json.dumps(modified_all, ensure_ascii=False)
                text_diff_j = json.dumps(text_diff, ensure_ascii=False)

            break_insert = False

            if (
                len(removed_all) == 0
                and len(added_all) == 0
                and len(modified_all) == 0
                and (not text_diff or text_diff.get("overlap_ratio", 1.0) >= 0.99)

            ):                 
                logger.info("🔍 Diff vacío — NO se inserta en screen_diffs.")
                break_insert = True
                has_changes = False  # No hubo cambios
                added_count = 0
                removed_count = 0
                modified_count = 0

                return has_changes, added_count, removed_count, modified_count


                # Insertar diff
            if not break_insert:
                # Determinar prioridad basada en retroalimentación
                diff_priority = 'low' if mark_as_low_priority else 'high'

                cur.execute("""
                    INSERT INTO screen_diffs (
                        tester_id, build_id, screen_name, header_text,
                        removed, added, modified, text_diff, diff_hash,
                        text_overlap, overlap_ratio, ui_structure_similarity, screen_status,
                        diff_priority, similarity_to_approved
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    t_id, b_id, s_name, header_text,
                    removed_j, added_j, modified_j, text_diff_j,
                    diff_signature, text_overlap, text_overlap,
                    ui_sim, screen_status,
                    diff_priority, similarity_to_approved
                ))

                conn.commit()

                # obtener diff_id real
                cur.execute("SELECT id FROM screen_diffs WHERE diff_hash = ?", (diff_signature,))
                diff_id = cur.fetchone()[0]

                # 🔹 Registrar decisión inicial en el feedback system (si está disponible)
                try:
                    if feedback_system is not None:
                        record_diff_decision(
                            diff_hash=diff_signature,
                            diff_signature=diff_signature,
                            app_name=app_name,
                            tester_id=t_id,
                            build_version=b_id,
                            decision='show' if diff_priority == 'high' else 'low_priority',
                            user_approved=True,
                            feedback_system=feedback_system
                        )
                except Exception as e:
                    logger.warning(f"⚠️  No se pudo registrar decision inicial en feedback_system: {e}")

                # ---- dedupe ----
                def dedupe_nodes(items):
                    seen = set()
                    unique = []
                    for item in items:
                        key = item.get("node", {}).get("key", "")
                        if key not in seen:
                            seen.add(key)
                            unique.append(item)
                    return unique

                added_all = dedupe_nodes(added_all)

                # ---- insertar diff_items ----
                for item in added_all:
                    node = item.get("node", {})
                    cur.execute("""
                        INSERT INTO diff_items (diff_id, action, node_class, node_key, node_text, raw_json)
                        VALUES (?,?,?,?,?,?)
                    """, (
                        diff_id, "added",
                        node.get("class"), node.get("key"),
                        node.get("text"),
                        json.dumps(item, ensure_ascii=False)
                    ))

                conn.commit()

                logger.info(f"🧩 Guardado cambio ({diff_signature[:8]}) en screen_diffs")

        else:
            logger.info(f"🧩 Sin cambios detectados para {s_name}")

    except Exception as e:
        logger.exception(f"❌ Error guardando diff en screen_diffs: {e}")


    # --------------------------------------
    # 🔍 VALIDACIÓN DE FLUJOS DE NAVEGACIÓN
    # --------------------------------------
    from FlowValidator import (
        validate_flow_sequence,
        update_flow_trees_incremental,
        build_flow_trees_from_db,
        get_sequence_from_db
    )

    ENABLE_FLOW_VALIDATION = True  # puedes apagarlo según config

    if ENABLE_FLOW_VALIDATION:
        try:
            # 1️⃣ Actualiza el modelo de flujos con la nueva sesión
            update_flow_trees_incremental(app_name, event.session_key)

            # 2️⃣ Recupera la secuencia completa de esa sesión
            seq = get_sequence_from_db(event.session_key)
            if len(seq) < 2:
                logger.info(f"🔹 Secuencia demasiado corta para validar flujo: {seq}")
            else:

                flow_trees = FLOW_MODELS.get(app_name, {})
                
                result = validate_flow_sequence(flow_trees, seq)
                if result["valid"]:
                    logger.info(f"✅ Flujo válido: {seq}")
                else:
                    logger.warning(f"⚠️ Flujo anómalo ({result['reason']}) → {seq}")

        except Exception as e:
            logger.error(f"Error al validar flujo: {e}")



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
    """Filtra claves estables sin destruir los tipos."""
    return {k: node.get(k, None) for k in SAFE_KEYS}     

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
    # logger.debug("Raw request: %s", event.model_dump())

    try:
        # -------------------- 1) Obtener nodos RAW (los originales) --------------------
        raw_nodes = event.collect_node_tree or event.tree_data or []

        semantic_screen_id = build_screen_id(raw_nodes)

        screen_id_short = extract_short_screen_id(semantic_screen_id)

        screen_id_final = sanitize_screen_id_for_fs(screen_id_short)

        semantic_screen_id_ctx.set(screen_id_final)

        #model_key = extract_model_key(semantic_screen_id)


        # 1. Guardar el ID sanitizado
        semantic_screen_id_fs = sanitize_screen_id_for_fs(semantic_screen_id)
        semantic_screen_id_ctf.set(semantic_screen_id_fs)

        # 2. Guardar también la versión "short"
        screen_id_short = extract_short_screen_id(semantic_screen_id)
        semantic_screen_id_ctf.set(screen_id_short)

        # 3. Recuperar EL VALOR REAL
        #semantic_screen_id_value = semantic_screen_id_ctf.get()

        # 4. Ahora sí llamar a extract_model_key()
       # model_key2 = extract_model_key(semantic_screen_id_value)

        # -------------------- 2) Firmas estables (aquí adentro ya se normaliza) -------
        signatures = stable_signatures(raw_nodes, header_text=event.header_text or "")
        global_signature = signatures["global_signature"]
        partial_signature = signatures["partial_signature"]

        # -------------------- 3) Normalizar solo campos del evento --------------------
        norm = _normalize_event_fields(event)
        tester_norm = norm.get("tester_id_norm")
        build_norm = norm.get("build_id_norm")
        screen_name = event.screen_names or ""
        header_text = event.header_text or ""
        screens_id_val = semantic_screen_id  
        #screens_id_val = event.screens_id or norm.get("screensId") or None

        # -------------------- 4) Detectar scroll --------------------
        scroll_type = detect_scroll_type(raw_nodes)

        baseline_flag = is_baseline_build(
            app_name=event.package_name,
            tester_id=tester_norm,
            build_id=build_norm
        )

        # -------------------- 5) Session key ------------------------
        base = tester_norm or getattr(event, "tester_id", "anon")
        minute_block = int(time.time() // 60)
        event.session_key = getattr(event, "session_key", None) or f"{base}_{minute_block}"

        # -------------------- 6) Detectar class raíz desde RAW -------------------------
        if raw_nodes:
            root_class_name = raw_nodes[0].get("className", "") if isinstance(raw_nodes[0], dict) else ""
        else:
            root_class_name = ""

        if not root_class_name and screen_name == "" and header_text == "":
            root_class_name = "SplashActivity"

        # -------------------- 7) Verificar si ya existe --------------------
        do_insert = True
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()

            existing = cursor.execute(
                """
                SELECT 1 FROM accessibility_data
                WHERE tester_id=? AND build_id=? AND global_signature=? AND partial_signature=?
                """,
                (tester_norm, build_norm, global_signature, partial_signature),
            ).fetchone()

            if existing:
                do_insert = False

            # -------------------- 8) Insertar solo si hay cambios --------------------
            if do_insert:
                cursor.execute(
                    """
                    INSERT INTO accessibility_data (
                        tester_id, build_id, is_baseline, timestamp, event_type, event_type_name,
                        package_name, class_name, text, content_description, screens_id,
                        screen_names, header_text, collect_node_tree, global_signature,
                        partial_signature, scroll_type, additional_info, tree_data,
                        session_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tester_norm,
                        build_norm,
                        1 if baseline_flag else 0, 
                        event.timestamp,
                        event.event_type,
                        event.event_type_name,
                        event.package_name,
                        root_class_name,
                        event.text,
                        event.content_description,
                        screens_id_val,
                        event.screen_names,
                        header_text,
                        # 🔥 GUARDA RAW NODES, NO NORMALIZADOS
                        json.dumps(raw_nodes, ensure_ascii=False),

                        global_signature,
                        partial_signature,
                        scroll_type,
                        json.dumps(event.additional_info or {}, ensure_ascii=False),
                        json.dumps(event.tree_data or [], ensure_ascii=False),
                        event.session_key,
                    ),
                )
                conn.commit()

            inserted_id = cursor.lastrowid
            # logger.info("[collect] Insert completado (nuevo build o cambios detectados).")

        # -------------------- 9) Encolar para análisis --------------------
        await event_queue.put((event, do_insert, partial_signature))

        # -------------------- 10) Entrenar modelo incremental --------------------
        has_changes, added_count, removed_count, modified_count = await analyze_and_train(event)

        # -------------------- 11) Respuesta --------------------
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "inserted": do_insert,
                "session_key": event.session_key,
                "scroll_type": scroll_type,
                "global_signature": global_signature,
                "partial_signature": partial_signature
            },
        )

    except Exception as e:
        logger.error(f"Error en /collect: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
 

async def worker_process_queue():
    while True:
        batch = []
        try:
            item = await event_queue.get()
            batch.append(item)

            # Recolecta más eventos por un corto periodo
            start = time.time()
            while len(batch) < BATCH_SIZE and (time.time() - start) < BATCH_INTERVAL:
                try:
                    batch.append(event_queue.get_nowait())
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)

            for ev, inserted, signature in batch:
                try:
                    norm = _normalize_event_fields(ev)
                    tester_norm = norm.get("tester_id_norm")
                    build_norm = norm.get("build_id_norm")
                    screen_name = ev.screen_names

                    # Debounce por pantalla
                    key = (tester_norm, screen_name)
                    now = time.time()
                    last_time = last_processed.get(key, 0)
                    if now - last_time < DEBOUNCE_TIME:
                        logger.debug(f"[worker] Skipping debounce: {key}")
                        continue
                    last_processed[key] = now

                    # -------------------- Solo analizar si insertó o cambio de snapshot --------------------
                    if not inserted:
                        # Revisar si la última versión en DB es idéntica
                        with sqlite3.connect(DB_NAME) as conn:
                            existing_sig = conn.execute(
                                "SELECT signature FROM accessibility_data WHERE tester_id=? AND build_id=? ORDER BY created_at DESC LIMIT 1",
                                (tester_norm, build_norm)
                            ).fetchone()
                            if existing_sig and existing_sig[0] == signature:
                                logger.debug(f"[worker] No cambios reales detectados para {tester_norm}/{screen_name}")
                                continue

                    # -------------------- Ejecutar análisis y métricas --------------------
                    has_changes, added_count, removed_count, modified_count = await analyze_and_train(ev)
                    if has_changes:
                        update_metrics(
                            tester_norm,
                            build_norm,
                            has_changes,
                            added_count=added_count,
                            removed_count=removed_count,
                            modified_count=modified_count
                        )
                        logger.info(f"[worker] Procesado y métricas actualizadas: {tester_norm}/{screen_name}")

                except Exception as e:
                    logger.error(f"[worker] Error procesando evento: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[worker] Error general en worker: {e}", exc_info=True)

        await asyncio.sleep(0.01)

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
        #screen_id=screen_id,
        screen_id=semantic_screen_id_ctx.get(),
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


class CIRequest(BaseModel):
    tester_id: str
    build_id: str
    baseline_build_id: Optional[str] = None
    screens: Optional[List[str]] = None
    threshold: float = 0.7
    max_results: int = 20


@app.post("/api/ci/check-diff")
def ci_check_diff(req: CIRequest):
    """Quick CI gating endpoint.

    Returns pass/fail and a severity score (0.0-1.0). Uses most recent diffs for the
    given tester/build. If severity >= threshold the result is "fail".
    """
    try:
        # lightweight local imports to avoid top-level dependency churn
        from typing import Optional, List
        import sqlite3

        # Verify subscription for the tester (blocks unpaid users)
        if not is_subscription_active(req.tester_id):
            return JSONResponse(status_code=403, content={
                "status": "forbidden",
                "message": "Subscription inactive or not found."
            })

        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        params = [req.tester_id, req.build_id]
        query = """
            SELECT id, screen_name, overlap_ratio, ui_structure_similarity, diff_priority, created_at
            FROM screen_diffs
            WHERE tester_id=? AND build_id=?
        """

        if req.screens:
            placeholders = ",".join("?" for _ in req.screens)
            query += f" AND screen_name IN ({placeholders})"
            params.extend(req.screens)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(req.max_results)

        rows = cur.execute(query, params).fetchall()
        conn.close()

        if not rows:
            return {"status": "pass", "severity": 0.0, "message": "No diffs detected", "diffs": []}

        severities = []
        diffs = []
        for r in rows:
            diff_id, screen_name, overlap_ratio, ui_sim, diff_priority, created_at = r
            overlap = overlap_ratio if overlap_ratio is not None else 1.0
            try:
                sev = 1.0 - float(overlap)
            except Exception:
                sev = 0.0
            severities.append(sev)
            diffs.append({
                "id": diff_id,
                "screen_name": screen_name,
                "overlap_ratio": overlap,
                "severity": sev,
                "diff_priority": diff_priority
            })

        max_sev = max(severities)
        status = "fail" if max_sev >= float(req.threshold) else "pass"

        result = {"status": status, "severity": max_sev, "diffs": diffs}

        # Send notifications on failure (non-blocking)
        try:
            if status == "fail":
                message = f"UI regression detected for tester={req.tester_id} build={req.build_id} severity={max_sev:.2f}"
                # Use environment-configured webhooks if available
                slack_url = globals().get('SLACK_WEBHOOK_URL') or os.environ.get('SLACK_WEBHOOK_URL')
                teams_url = globals().get('TEAMS_WEBHOOK_URL') or os.environ.get('TEAMS_WEBHOOK_URL')

                payload = {
                    "tester_id": req.tester_id,
                    "build_id": req.build_id,
                    "severity": max_sev,
                    "diff_count": len(diffs),
                    "diffs": diffs[:5]
                }

                # fire-and-forget style: send but don't block the response
                if slack_url:
                    try:
                        send_slack_alert(slack_url, message, payload)
                    except Exception:
                        logger.exception("Failed sending Slack alert")

                if teams_url:
                    try:
                        send_teams_alert(teams_url, message, payload)
                    except Exception:
                        logger.exception("Failed sending Teams alert")
        except Exception:
            logger.exception("Error sending CI notifications")

        return result

    except Exception as e:
        logger.exception(f"Error in /api/ci/check-diff: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/screen/diffs")
def get_screen_diffs(
    tester_id: Optional[str] = Query(None),
    build_id: Optional[str] = Query(None),
    header_text: Optional[str] = Query(None),
    only_pending: bool = Query(True),
    only_approved: bool = Query(False),
    only_rejected: bool = Query(False)
):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 🔹 QUERY MEJORADA: incluye JOIN a diff_rejections
    query = """
        SELECT 
            s.id, 
            s.tester_id, 
            s.build_id, 
            s.screen_name, 
            s.header_text,
            s.removed, 
            s.added, 
            s.modified, 
            s.text_diff, 
            s.created_at, 
            s.cluster_info,
            CASE 
                WHEN a.id IS NOT NULL THEN 'approved'
                WHEN r.id IS NOT NULL THEN 'rejected'
                ELSE 'pending'
            END AS approval_status,
            a.created_at AS approved_at,
            r.created_at AS rejected_at,
            r.rejection_reason
        FROM screen_diffs AS s
        LEFT JOIN diff_approvals AS a ON a.diff_id = s.id
        LEFT JOIN diff_rejections AS r ON r.diff_id = s.id
        WHERE 1=1
    """
    params = []

    # 🔹 FILTRO DE ESTADO MEJORADO
    if only_pending:
        query += " AND a.id IS NULL AND r.id IS NULL"
    elif only_approved:
        query += " AND a.id IS NOT NULL"
    elif only_rejected:
        query += " AND r.id IS NOT NULL"

    # 🔹 FILTRO TESTER_ID SIMPLIFICADO
    if tester_id is not None and tester_id != "":
        query += " AND s.tester_id = ?"
        params.append(tester_id)

    # 🔹 FILTRO HEADER_TEXT MEJORADO
    if header_text is not None and header_text != "":
        query += " AND LOWER(TRIM(s.header_text)) LIKE LOWER(TRIM(?))"
        params.append(f"%{header_text.strip()}%")

    query += " ORDER BY s.created_at DESC LIMIT 1000"

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
    traces_to_batch = []

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
        elif overlap_ratio >= 1:
            screen_status = "no_changes"  # Sin Cambios    
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

        # Construir changes_list para trace
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

        # Guardar trace para batch después del loop
        traces_to_batch.append({
            "tester_id": tester_id,
            "build_id": build_id,
            "screen": row[3],
            "changes": changes_list
        })

        # Determinar estado de aprobación
        approval_status = row[11] if len(row) > 11 else "pending"  # De la query CASE statement
        approved_at = row[12] if len(row) > 12 else None
        rejected_at = row[13] if len(row) > 13 else None
        rejection_reason = row[14] if len(row) > 14 else None

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
            "approval": {
                "status": approval_status,
                "approved_at": approved_at,
                "rejected_at": rejected_at,
                "rejection_reason": rejection_reason,
                "is_pending": approval_status == "pending"
            }
        })

    # ✅ Calcular metadata de aprobaciones
    approval_counts = {
        "pending": sum(1 for d in diffs if d["approval"]["status"] == "pending"),
        "approved": sum(1 for d in diffs if d["approval"]["status"] == "approved"),
        "rejected": sum(1 for d in diffs if d["approval"]["status"] == "rejected")
    }

    # ✅ Batch update de traces (fuera del loop)
    for trace in traces_to_batch:
        try:
            update_diff_trace(
                tester_id=trace["tester_id"],
                build_id=trace["build_id"],
                screen=trace["screen"],
                changes=trace["changes"]
            )
        except Exception as e:
            print(f"Error updating trace: {e}")

    # ✅ Cálculo robusto de has_changes
    for d in diffs:
        d["has_changes"] = any([
            len(d.get("removed", [])) > 0,
            len(d.get("added", [])) > 0,
            len(d.get("modified", [])) > 0,
            bool(d.get("text_diff", {}))
        ])

    has_changes = any(diff.get("has_changes", False) for diff in diffs)

    return {
        "screen_diffs": diffs,
        "metadata": {
            "pending": approval_counts["pending"],
            "approved": approval_counts["approved"],
            "rejected": approval_counts["rejected"],
            "total_diffs": len(diffs),
            "total_changes": sum(d["detailed_changes"].__len__() for d in diffs),
            "has_changes": has_changes
        },
        "request_filters": {
            "only_pending": only_pending,
            "only_approved": only_approved,
            "only_rejected": only_rejected,
            "tester_id": tester_id,
            "build_id": build_id
        }
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
                rejection_reason TEXT,       
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
    #print(f"📩 Petición recibida desde Android: {req.email}")
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
        #print(f"🔑 Password for {req.email} updated to: {req.new_password}")
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
    import random, string, time, sqlite3

    chars = string.ascii_uppercase + string.digits
    prefix_char = random.choice(chars)
    device_prefix = ''.join(random.choices("0123456789ABCDEF", k=2))
    random_part = str(random.randint(0, 9999)).zfill(4)
    codigo = f"{prefix_char}{device_prefix}{random_part}"

    generado_en = int(time.time())
    expira_en = generado_en + duracion_dias * 24 * 3600

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # 1️⃣ Verificar si el usuario tiene plan activo
        plan = c.execute("""
            SELECT id, plan_name, price, no_associate, max_tester
            FROM active_plans
            WHERE active = 1
            ORDER BY id ASC LIMIT 1
        """).fetchone()

        if plan:
            plan_id = plan["id"]
            price = plan["price"]
            max_tester = plan["max_tester"]

            # Verificar si el usuario no ha superado su límite mensual (30 códigos)
            codigos_mes = c.execute("""
                SELECT COUNT(*) AS total FROM login_codes
                WHERE usuario_id = ? AND plan_id = ? AND activo = 1
            """, (usuario_id, plan_id)).fetchone()["total"]

            if codigos_mes >= 30:
                return {"success": False, "reason": "Límite de 30 códigos alcanzado este mes."}

            costo = price
            plan_tipo = plan["plan_name"]

        else:
            # Usuario sin plan activo
            plan = c.execute("""
                SELECT id, no_associate FROM active_plans
                WHERE plan_name = 'BASIC'
            """).fetchone()
            if not plan:
                return {"success": False, "reason": "No hay plan base configurado."}

            plan_id = plan["id"]
            costo = plan["no_associate"]
            plan_tipo = "NO_PLAN"

        # 2️⃣ Guardar el código generado
        c.execute("""
            INSERT INTO login_codes (codigo, usuario_id, plan_id, generado_en, expira_en, usos_permitidos, usos_actuales, activo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (codigo, usuario_id, plan_id, generado_en, expira_en, usos_permitidos, 0, 1))
        conn.commit()

    return {
        "success": True,
        "codigo": codigo,
        "plan": plan_tipo,
        "costo_usd": costo,
        "expira_en": expira_en,
        "duracion_dias": duracion_dias
    }


@router.post("/validate_code")
async def validate_code(request: Request):
    import time, sqlite3
    data = await request.json()
    codigo = data.get("codigo", "").strip().upper()

    if not codigo:
        return {"valid": False, "reason": "Código vacío"}

    now = int(time.time())

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        row = c.execute("""
            SELECT l.*, a.plan_name
            FROM login_codes l
            LEFT JOIN active_plans a ON a.id = l.plan_id
            WHERE l.codigo = ? AND l.activo = 1 AND l.is_paid = 1
        """, (codigo,)).fetchone()

        if not row:
            return {"valid": False, "reason": "Código no encontrado"}

        if now > row["expira_en"]:
            return {"valid": False, "reason": "Código expirado"}

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
        "plan": row["plan_name"],
        "expira_en": row["expira_en"],
        "restante_en_segundos": restante,
        "usos_restantes": row["usos_permitidos"] - (row["usos_actuales"] + 1)
    }

def verificar_pago_externo(payment_token: str) -> bool:
    """
    Mock temporal para simular la validación con un proveedor de pagos.
    - Si el token contiene 'OK', se aprueba el pago.
    - Si contiene 'FAIL', se rechaza.
    - En otros casos, se aprueba aleatoriamente (80% éxito).
    """
    if not payment_token:
        return False
    if "OK" in payment_token.upper():
        return True
    if "FAIL" in payment_token.upper():
        return False
    # Modo aleatorio de prueba
    return random.random() < 0.8

@router.post("/confirm_payment")
async def confirm_payment(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body vacío o JSON inválido")

    codigo = data.get("codigo")
    payment_token = data.get("payment_token")
    usuario_id = data.get("usuario_id")  # 👈 agregado

    if not codigo or not payment_token or not usuario_id:
        raise HTTPException(status_code=400, detail="Faltan parámetros requeridos")

    # 1️⃣ Verificar que el código existe y pertenece al usuario
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        row = c.execute("""
            SELECT * FROM login_codes 
            WHERE codigo = ? AND usuario_id = ?
        """, (codigo, usuario_id)).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Código no encontrado o no pertenece al usuario")
        if row["activo"] == 1:
            raise HTTPException(status_code=400, detail="Código ya está activo o pagado")

    # 2️⃣ Verificar pago (mock temporal)
    pago_exitoso = verificar_pago_externo(payment_token)

    if not pago_exitoso:
        return {"success": False, "reason": "Pago rechazado por el proveedor (mock)"}

    # 3️⃣ Activar el código solo si pertenece al usuario correcto
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE login_codes
            SET activo = 1
            WHERE codigo = ? AND usuario_id = ?
        """, (codigo, usuario_id))
        conn.commit()

    return {
        "success": True,
        "codigo": codigo,
        "usuario_id": usuario_id,
        "message": "✅ Pago confirmado y código activado correctamente"
    }


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


# =====================================================
# ENDPOINTS DE CONFIGURACIÓN
# =====================================================

@app.get("/api/config")
def get_current_config():
    """
    Retorna la configuración actual (con valores sensibles enmascarados).
    Útil para debugging y auditoría.
    """
    try:
        config_dict = config.to_dict()
        return JSONResponse(
            content={
                "status": "ok",
                "config": config_dict,
                "file_path": str(config.config_path)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/config/notifications")
def get_notifications_config():
    """
    Retorna solo la configuración de notificaciones.
    """
    try:
        notifications_config = config.get_section("notifications")
        # Enmascarar valores sensibles
        safe_config = config._mask_sensitive_values(notifications_config.copy())
        
        return JSONResponse(
            content={
                "status": "ok",
                "slack": {
                    "enabled": config.is_notification_enabled("slack"),
                    "has_webhook": bool(config.get_webhook_url("slack"))
                },
                "teams": {
                    "enabled": config.is_notification_enabled("teams"),
                    "has_webhook": bool(config.get_webhook_url("teams"))
                },
                "jira": {
                    "enabled": config.is_notification_enabled("jira"),
                    "has_credentials": bool(config.get("notifications.jira.api_token"))
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.post("/api/config/test-slack")
def test_slack_notification():
    """
    Envía un mensaje de prueba a Slack para verificar la configuración.
    """
    try:
        if not config.is_notification_enabled("slack"):
            return JSONResponse(
                status_code=400,
                content={"error": "Slack notifications are disabled"}
            )
        
        webhook_url = config.get_webhook_url("slack")
        if not webhook_url:
            return JSONResponse(
                status_code=400,
                content={"error": "Slack webhook URL not configured"}
            )
        
        test_payload = {
            "tester_id": "test-user",
            "build_id": "test-build-v1.0",
            "severity": 0.5,
            "diff_count": 3
        }
        
        success = send_slack_alert(
            webhook_url=webhook_url,
            title="🧪 Test Alert from AxiomServiceIA",
            payload=test_payload
        )
        
        if success:
            return JSONResponse(
                content={"status": "ok", "message": "Test message sent to Slack"}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to send test message to Slack"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/config/test-teams")
def test_teams_notification():
    """
    Envía un mensaje de prueba a Teams para verificar la configuración.
    """
    try:
        if not config.is_notification_enabled("teams"):
            return JSONResponse(
                status_code=400,
                content={"error": "Teams notifications are disabled"}
            )
        
        webhook_url = config.get_webhook_url("teams")
        if not webhook_url:
            return JSONResponse(
                status_code=400,
                content={"error": "Teams webhook URL not configured"}
            )
        
        test_payload = {
            "tester_id": "test-user",
            "build_id": "test-build-v1.0",
            "severity": 0.5,
            "diff_count": 3
        }
        
        success = send_teams_alert(
            webhook_url=webhook_url,
            title="🧪 Test Alert from AxiomServiceIA",
            payload=test_payload
        )
        
        if success:
            return JSONResponse(
                content={"status": "ok", "message": "Test message sent to Teams"}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to send test message to Teams"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/config/reload")
def reload_config():
    """
    Recarga la configuración desde el archivo config.yaml.
    Útil para aplicar cambios sin reiniciar el servidor.
    """
    try:
        config.reload()
        return JSONResponse(
            content={
                "status": "ok",
                "message": "Configuration reloaded successfully"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to reload config: {str(e)}"}
        )


@app.get("/api/config/ci")
def get_ci_config():
    """
    Retorna solo la configuración de CI.
    """
    try:
        ci_config = config.get_section("ci")
        return JSONResponse(
            content={
                "status": "ok",
                "ci": ci_config
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/config/ml")
def get_ml_config():
    """
    Retorna solo la configuración de modelos ML.
    """
    try:
        ml_config = config.get_section("ml")
        return JSONResponse(
            content={
                "status": "ok",
                "ml": ml_config
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.get("/api/config/health")
def config_health_check():
    """
    Realiza un health check de la configuración.
    Verifica que todos los servicios necesarios estén correctamente configurados.
    """
    try:
        health_status = {
            "status": "ok",
            "checks": {
                "slack": {
                    "enabled": config.is_notification_enabled("slack"),
                    "configured": bool(config.get_webhook_url("slack")),
                    "ready": config.is_notification_enabled("slack") and bool(config.get_webhook_url("slack"))
                },
                "teams": {
                    "enabled": config.is_notification_enabled("teams"),
                    "configured": bool(config.get_webhook_url("teams")),
                    "ready": config.is_notification_enabled("teams") and bool(config.get_webhook_url("teams"))
                },
                "jira": {
                    "enabled": config.is_notification_enabled("jira"),
                    "configured": bool(config.get("notifications.jira.api_token")),
                    "ready": config.is_notification_enabled("jira") and bool(config.get("notifications.jira.api_token"))
                },
                "database": {
                    "path": config.get("database.path", "unknown"),
                    "exists": os.path.exists(config.get("database.path", ""))
                }
            }
        }
        
        # Determinar estado general
        notification_services = [
            health_status["checks"]["slack"]["ready"],
            health_status["checks"]["teams"]["ready"],
            health_status["checks"]["jira"]["ready"]
        ]
        
        if any(notification_services):
            health_status["overall"] = "healthy"
        else:
            health_status["overall"] = "degraded"
            health_status["warning"] = "At least one notification service should be configured"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "overall": "unhealthy",
                "error": str(e)
            }
        )


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

@app.get("/qa_report/{tester_id}", response_class=HTMLResponse)
def qa_report(tester_id: str, builds: Optional[int] = 5):

    import json
    import sqlite3
    from collections import defaultdict

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 🔍 Tomar los últimos builds y todas sus pantallas
    c.execute("""
        SELECT build_id, header_text, removed, added, modified, anomaly_score, cluster_id
        FROM screen_diffs
        WHERE tester_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (tester_id, builds * 50))

    rows = c.fetchall()
    conn.close()

    # Estructuras
    builds_dict = defaultdict(list)
    cluster_map = defaultdict(list)

    for build_id, screen, removed_raw, added_raw, modified_raw, anomaly_score, cluster_id in rows:

        removed = json.loads(removed_raw) if removed_raw else []
        added = json.loads(added_raw) if added_raw else []
        modified = json.loads(modified_raw) if modified_raw else []

        entry = {
            "screen": screen,
            "removed": removed,
            "added": added,
            "modified": modified,
            "anomaly_score": anomaly_score or 0,
            "cluster_id": cluster_id or "-"
        }

        builds_dict[build_id].append(entry)
        cluster_map[entry["cluster_id"]].append(entry)

    builds_sorted = sorted(builds_dict.keys())

    # ======== HTML =========
    html = """
    <html>
    <head>
        <title>QA Report Avanzado</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

        <style>
            body { font-family: Arial; padding:20px; }

            .section-title {
                margin-top: 40px;
                font-size: 24px;
                border-bottom: 2px solid #ccc;
                padding-bottom: 5px;
            }

            table { width:100%; border-collapse:collapse; margin-bottom:20px; }
            th, td { border:1px solid #ddd; padding:6px; }

            .removed-row { background:#fdd; }
            .added-row { background:#dfd; }
            .modified-row { background:#ffd; }

            .change-box {
                margin:4px; padding:6px; border-radius:4px;
                background:#fafafa; border:1px solid #ccc;
            }

            .similarity-low { color:red; font-weight:bold; }
            .similarity-mid { color:orange; font-weight:bold; }
            .similarity-high { color:green; font-weight:bold; }

            details { margin-bottom:5px; }
        </style>
    </head>
    <body>
    """

    html += f"<h1>Reporte Avanzado QA — Tester <b>{tester_id}</b></h1>"

    # ============================
    #   📊 SECCIÓN 1 — GRAFICO GENERAL
    # ============================

    removed_series = [sum(len(s["removed"]) for s in builds_dict[b]) for b in builds_sorted]
    added_series   = [sum(len(s["added"]) for s in builds_dict[b]) for b in builds_sorted]
    modified_series= [sum(len(s["modified"]) for s in builds_dict[b]) for b in builds_sorted]
    anomaly_avg    = [
        sum(s["anomaly_score"] for s in builds_dict[b]) / max(1, len(builds_dict[b]))
        for b in builds_sorted
    ]

    html += """
    <div class="section-title">📈 Tendencias Generales</div>
    <canvas id="chart1" height="120"></canvas>
    <script>
        new Chart(document.getElementById('chart1'), {
            type: 'line',
            data: {
                labels: """ + json.dumps(builds_sorted) + """,
                datasets: [
                    {label:'Removed', data:""" + json.dumps(removed_series) + """, borderColor:'red'},
                    {label:'Added', data:""" + json.dumps(added_series) + """, borderColor:'green'},
                    {label:'Modified', data:""" + json.dumps(modified_series) + """, borderColor:'orange'},
                    {label:'Anomaly Avg', data:""" + json.dumps(anomaly_avg) + """, borderColor:'blue'}
                ]
            }
        });
    </script>
    """

    # ============================
    #   🧬 SECCIÓN 2 — CLUSTERS
    # ============================

    html += """
    <div class='section-title'>🧬 Clusters Detectados</div>
    """

    for cluster_id, screens in cluster_map.items():
        color = f"hsl({hash(cluster_id) % 360}, 60%, 70%)"
        html += f"<h3 style='color:{color}'>Cluster {cluster_id} ({len(screens)} pantallas)</h3>"

        html += "<ul>"
        for s in screens:
            html += f"<li>{s['screen']}</li>"
        html += "</ul>"

    # ============================
    #   📄 SECCIÓN 3 — DETALLE POR BUILD
    # ============================

    html += """
    <div class="section-title">📄 Cambios por Build</div>
    """

    for build_id in builds_sorted:

        html += f"<h2>Build {build_id}</h2>"
        html += """
        <table>
        <tr><th>Pantalla</th><th>Removed</th><th>Added</th><th>Modified</th><th>Detalle</th></tr>
        """

        for s in builds_dict[build_id]:

            html += f"""
            <tr>
                <td>{s['screen']}</td>
                <td>{len(s['removed'])}</td>
                <td>{len(s['added'])}</td>
                <td>{len(s['modified'])}</td>
                <td>
                    <details>
                        <summary>Ver detalles</summary>
                        <div class='change-box'>
            """

            # REMOVED
            if s["removed"]:
                html += "<h4>Removed</h4>"
                for item in s["removed"]:
                    html += f"<div>- {item['node']['key']}</div>"

            # ADDED
            if s["added"]:
                html += "<h4>Added</h4>"
                for item in s["added"]:
                    html += f"<div>+ {item['node']['key']}</div>"

            # MODIFIED
            if s["modified"]:
                html += "<h4>Modified</h4>"
                for item in s["modified"]:
                    ch = item["changes"]

                    sim = float(ch.get("similarity", 0))
                    if sim < 0.4:
                        cls = "similarity-low"
                    elif sim < 0.7:
                        cls = "similarity-mid"
                    else:
                        cls = "similarity-high"

                    html += f"""
                        <div class='change-box'>
                            <b>{item['node']['class']}</b><br>
                            <span class='{cls}'>Similarity: {sim:.2f}</span>
                            <br>Old: {ch['text']['old']}
                            <br>New: {ch['text']['new']}
                        </div>
                    """

            html += "</div></details></td></tr>"

        html += "</table>"

    html += "</body></html>"

    return HTMLResponse(content=html)


app.include_router(diff_router)
app.include_router(router, prefix="/api")


@app.post("/train/toggle-general")
async def toggle_general_train(app_name: str, tester_id: str, build_id: str, enabled: bool):
    """
    Activa/desactiva el entrenamiento general automático para este tester/app/build
    """

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Actualizar tabla de configuración del tester
    c.update_general_flag(tester_id, app_name, build_id, enabled)
    return {"status": "ok", "train_general_enabled": enabled}

@app.post("/baseline/mark")
async def mark_baseline(req: BaselineMarkRequest):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        DELETE FROM baseline_metadata
        WHERE app_name = ? AND tester_id = ?
    """, (req.app_name, req.tester_id))

    c.execute("""
        INSERT INTO baseline_metadata (app_name, tester_id, build_id)
        VALUES (?, ?, ?)
    """, (req.app_name, req.tester_id, req.build_id))

    conn.commit()
    conn.close()

    return {"status": "ok", "message": "Baseline marcada correctamente"}


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


@app.post("/update_rates")
async def update_rates(usd_to_cop: float, usd_to_eur: float):
    # Validaciones básicas
    if usd_to_cop <= 0 or usd_to_eur <= 0:
        raise HTTPException(status_code=400, detail="Las tasas deben ser mayores que cero.")

    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            c = conn.cursor()
            c.execute("""
                UPDATE active_plans
                SET rate_cop = ?, rate_eur = ?, updated_at = CURRENT_TIMESTAMP
            """, (usd_to_cop, usd_to_eur))
            conn.commit()

        return {
            "success": True,
            "message": "Tasas actualizadas correctamente.",
            "usd_to_cop": usd_to_cop,
            "usd_to_eur": usd_to_eur
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar tasas: {str(e)}")
    
    
@app.post("/train/sync-mode")
async def sync_train_mode(payload: TrainModeRequest):
    """
    Endpoint intermedio que se llama desde Android.
    Si train_general=True -> ejecuta entrenamiento general.
    Si train_general=False -> ejecuta entrenamiento incremental.
    """

    if payload.train_general:
        # 🔹 Modo general activado
        await _train_general_logic_hybrid(
            app_name=payload.app_name,
            batch_size=1000,
            min_samples=1,
            update_general=True
        )
        return {
            "status": "success",
            "mode": "general",
            "message": f"Entrenamiento general iniciado para {payload.app_name}"
        }

    else:
        # 🔹 Modo incremental
        if (
            not payload.tester_id
            or not payload.build_id
            or not payload.screen_id
            or not payload.enriched_vector
        ):
            return {
                "status": "error",
                "message": "Faltan parámetros para entrenamiento incremental"
            }

        await _train_incremental_logic_hybrid(
            enriched_vector=np.array(payload.enriched_vector, dtype=float),
            tester_id=payload.tester_id,
            build_id=payload.build_id,
            app_name=payload.app_name,
            # screen_id=payload.screen_id,
            screen_id=semantic_screen_id_ctx.get(),
            min_samples=2,
            use_general_as_base=True
        )

        return {
            "status": "success",
            "mode": "incremental",
            "message": f"Entrenamiento incremental ejecutado para {payload.app_name}/{payload.screen_id}"
        }


def calculate_all_metrics():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Limpia el resumen viejo
        cur.execute("DELETE FROM metrics_summary")

        # ----------------------------
        #   A) MÉTRICAS POR PANTALLA
        # ----------------------------

        # % de cambio total
        q1 = """
            SELECT screen_signature,
                   SUM(total_added + total_removed + total_modified) AS total_changes,
                   COUNT(*) AS appearances,
                   ROUND(100.0 * SUM(total_added + total_removed + total_modified) / COUNT(*), 2) AS change_rate
            FROM changes
            GROUP BY screen_signature
        """
        for r in cur.execute(q1).fetchall():
            cur.execute("""
                INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
                VALUES (?, ?, ?)
            """, ("screen", f"change_rate_{r['screen_signature']}", r['change_rate']))

        # Severidad
        q2 = """
            SELECT screen_signature,
                   ROUND(AVG(total_modified),2) AS severity
            FROM changes GROUP BY screen_signature
        """
        for r in cur.execute(q2).fetchall():
            cur.execute("""
                INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
                VALUES (?, ?, ?)
            """, ("screen", f"severity_{r['screen_signature']}", r['severity']))

        # Frecuencia aparición de pantalla
        q3 = """
            SELECT screen_signature, COUNT(*) AS freq
            FROM collect GROUP BY screen_signature
        """
        for r in cur.execute(q3).fetchall():
            cur.execute("""
                INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
                VALUES (?, ?, ?)
            """, ("screen", f"frequency_{r['screen_signature']}", r['freq']))

        # -------------------------
        #   B) MÉTRICAS POR BUILD
        # -------------------------

        # Build con más cambios
        q4 = """
            SELECT build_id, SUM(total_changes) AS changes
            FROM metrics_changes
            GROUP BY build_id
            ORDER BY changes DESC LIMIT 1
        """
        r = cur.execute(q4).fetchone()
        if r:
            cur.execute("""
                INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
                VALUES ('build', 'most_changes_build', ?)
            """, (f"{r['build_id']} ({r['changes']} cambios)",))

        # Build más inestable (tasa de cambios)
        q5 = """
            SELECT build_id,
                   ROUND(100.0 * SUM(total_changes) / SUM(total_events), 2) AS instability
            FROM metrics_changes
            GROUP BY build_id ORDER BY instability DESC LIMIT 1
        """
        r = cur.execute(q5).fetchone()
        if r:
            cur.execute("""
                INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
                VALUES ('build', 'most_unstable_build', ?)
            """, (f"{r['build_id']} ({r['instability']}%)",))

        # Ratio cambios/pantalla
        q6 = """
            SELECT build_id,
                   ROUND(SUM(total_changes) * 1.0 / COUNT(DISTINCT screen_signature), 2) AS ratio
            FROM changes GROUP BY build_id
        """
        for r in cur.execute(q6).fetchall():
            cur.execute("""
                INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
                VALUES ('build', ?, ?)
            """, (f"change_screen_ratio_{r['build_id']}", r['ratio']))

        # ------------------------------
        #   C) MÉTRICAS GLOBAL QA LEAD
        # ------------------------------

        # Ranking pantallas más inestables (top 5)
        q7 = """
            SELECT screen_signature,
                   ROUND(AVG(total_added + total_removed + total_modified),2) AS avg_changes
            FROM changes GROUP BY screen_signature
            ORDER BY avg_changes DESC LIMIT 5
        """
        rank = ", ".join([f"{row['screen_signature']}({row['avg_changes']})"
                          for row in cur.execute(q7).fetchall()])
        cur.execute("""
            INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
            VALUES ('global', 'top_instable_screens', ?)
        """, (rank,))

        # Guardar fecha actualización
        cur.execute("""
            INSERT INTO metrics_summary(metric_group, metric_name, metric_value)
            VALUES ('global', 'last_run', ?)
        """, (datetime.now().isoformat(),))

        conn.commit()

    return {"status": "ok", "msg": "all metrics calculated"}


@app.get("/metrics/all")
async def metrics_all():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM metrics_summary").fetchall()
        return {"metrics": [dict(r) for r in rows]}       
    

# ============================================================
# 🔹 NUEVOS ENDPOINTS: SISTEMA DE RETROALIMENTACIÓN
# ============================================================

@app.post("/diff/{diff_id}/approve")
async def approve_diff(diff_id: int):
    """
    Registra aprobación de un diff.
    El modelo aprenderá a no mostrar similares en el futuro.
    """
    global feedback_system
    if feedback_system is None:
        return {"error": "Feedback system not initialized", "status": 503}
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            
            # Obtener detalles del diff
            c.execute("""
                SELECT diff_hash, tester_id, app_name, build_id, screen_name
                FROM screen_diffs WHERE id = ? LIMIT 1
            """, (diff_id,))
            
            diff_row = c.fetchone()
            if not diff_row:
                return {"error": "Diff not found", "status": 404}
            
            diff_hash, tester, app, build, screen = diff_row
            
            # Registrar aprobación en feedback system
            record_diff_decision(
                diff_hash=diff_hash,
                diff_signature=diff_hash,
                app_name=app,
                tester_id=tester,
                build_version=build,
                decision='approved',
                user_approved=True,
                feedback_system=feedback_system
            )
            
            # Actualizar BD
            c.execute("""
                UPDATE screen_diffs 
                SET diff_priority = 'low', approved_before = 1
                WHERE id = ?
            """, (diff_id,))
            
            conn.commit()
        
        logger.info(f"✅ Diff {diff_id} aprobado por tester {tester}")
        return {
            "success": True,
            "message": f"Diff approved - modelo mejorado",
            "diff_id": diff_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error aprobando diff: {e}")
        return {"error": str(e), "status": 500}


@app.post("/diff/{diff_id}/reject")
async def reject_diff(diff_id: int):
    """
    Registra rechazo de un diff (falso positivo).
    El modelo aprenderá a no mostrar similares.
    """
    global feedback_system
    if feedback_system is None:
        return {"error": "Feedback system not initialized", "status": 503}
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            
            c.execute("""
                SELECT diff_hash, tester_id, app_name, build_id
                FROM screen_diffs WHERE id = ? LIMIT 1
            """, (diff_id,))
            
            diff_row = c.fetchone()
            if not diff_row:
                return {"error": "Diff not found", "status": 404}
            
            diff_hash, tester, app, build = diff_row
            
            # Registrar rechazo
            record_diff_decision(
                diff_hash=diff_hash,
                diff_signature=diff_hash,
                app_name=app,
                tester_id=tester,
                build_version=build,
                decision='rejected',
                user_approved=False,
                feedback_system=feedback_system
            )
            
            # Marcar como baja prioridad (falso positivo)
            c.execute("""
                UPDATE screen_diffs 
                SET diff_priority = 'low'
                WHERE id = ?
            """, (diff_id,))
            
            conn.commit()
        
        logger.info(f"❌ Diff {diff_id} rechazado por tester {tester} - falso positivo")
        return {
            "success": True,
            "message": "Diff marcado como falso positivo",
            "diff_id": diff_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error rechazando diff: {e}")
        return {"error": str(e), "status": 500}


@app.get("/learning-insights/{app_name}/{tester_id}")
async def get_learning_insights(app_name: str, tester_id: str):
    """
    Retorna métricas de aprendizaje del modelo para un tester.
    Muestra cómo está mejorando la precisión.
    """
    global feedback_system
    if feedback_system is None:
        return {"error": "Feedback system not initialized", "status": 503}
    
    try:
        insights = feedback_system.get_learning_insights(app_name, tester_id)
        return {
            "app_name": app_name,
            "tester_id": tester_id,
            "insights": insights,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo insights: {e}")
        return {"error": str(e), "status": 500}


# ============================================================
# 🔹 ENDPOINTS: FLOW ANALYTICS ENGINE
# ============================================================

# Instancia global del motor de análisis de flujos
flow_analytics_engine = None

@app.on_event("startup")
async def init_flow_analytics():
    """Inicializar FlowAnalyticsEngine al arrancar la app."""
    global flow_analytics_engine
    try:
        from FlowAnalyticsEngine import FlowAnalyticsEngine
        flow_analytics_engine = FlowAnalyticsEngine(app_name="default_app")
        logger.info("✅ FlowAnalyticsEngine inicializado correctamente")
    except Exception as e:
        logger.warning(f"⚠️  No se pudo inicializar FlowAnalyticsEngine: {e}")
        flow_analytics_engine = None


@app.post("/flow-analyze/{app_name}/{tester_id}")
async def analyze_tester_flows(app_name: str, tester_id: str, request: Request):
    """
    Analiza los flujos de un tester y genera reporte con calidad, anomalías y sugerencias.
    
    Body esperado (opcional):
    {
        "session_key": "tester_123_minute_block_xyz",
        "flow_sequence": ["home", "profile", "settings"]
    }
    
    Retorna:
    {
        "tester_id": "...",
        "app_name": "...",
        "quality_score": 85.5,
        "anomaly_rate": 0.12,
        "total_flows_analyzed": 42,
        "suggestions": [...],
        "report": "Texto del reporte"
    }
    """
    global flow_analytics_engine
    
    if flow_analytics_engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "FlowAnalyticsEngine not initialized"}
        )
    
    try:
        payload = await request.json()
    except:
        payload = {}
    
    try:
        # Generar reporte de flujos para el tester
        report = flow_analytics_engine.generate_tester_flow_report(
            app_name=app_name,
            tester_id=tester_id
        )
        
        logger.info(f"📊 Reporte de flujos generado para {tester_id} en {app_name}")
        
        return {
            "success": True,
            "app_name": app_name,
            "tester_id": tester_id,
            "report": report,
            "status": "ok"
        }
        
    except Exception as e:
        logger.error(f"❌ Error analizando flujos: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/flow-dashboard/{app_name}")
async def get_flow_analytics_dashboard(app_name: str):
    """
    Dashboard global de análisis de flujos:
    - Distribución de pantallas por frecuencia
    - Puntos de interrupción (donde los flujos se desvían)
    - Anomalías totales detectadas
    - Recomendaciones generales
    
    Retorna:
    {
        "app_name": "...",
        "dashboard": {
            "total_flows": 500,
            "unique_screens": 25,
            "interruption_hotspots": [...],
            "anomalies_summary": {...},
            "recommendations": [...]
        }
    }
    """
    global flow_analytics_engine
    
    if flow_analytics_engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "FlowAnalyticsEngine not initialized"}
        )
    
    try:
        dashboard = flow_analytics_engine.get_flow_analytics_dashboard(app_name)
        
        logger.info(f"📈 Dashboard de flujos recuperado para {app_name}")
        
        return {
            "success": True,
            "app_name": app_name,
            "dashboard": dashboard,
            "status": "ok"
        }
        
    except Exception as e:
        logger.error(f"❌ Error generando dashboard: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ============================================================
# 🤖 QA AI DASHBOARD - ANÁLISIS INTELIGENTE DE CAMBIOS
# ============================================================

from qa_ai_dashboard import qa_ai_router
app.include_router(qa_ai_router)


@app.get("/flow-anomalies/{tester_id}")
async def get_flow_anomalies(
    tester_id: str,
    limit: int = Query(50, ge=1, le=500),
    severity: Optional[str] = Query(None, regex="^(low|medium|high)$")
):
    """
    Recupera historial de anomalías detectadas para un tester.
    
    Query params:
    - tester_id: ID del tester (requerido)
    - limit: Máximo número de registros (1-500, default=50)
    - severity: Filtrar por severidad (low|medium|high, opcional)
    
    Retorna:
    {
        "tester_id": "...",
        "anomalies": [
            {
                "id": 1,
                "app_name": "com.example.app",
                "flow_sequence": ["home", "profile", "error"],
                "deviation_point": "profile",
                "deviation_reason": "Botón inactivo detectado",
                "severity": "high",
                "timestamp": "2024-01-15T10:30:00"
            },
            ...
        ],
        "total": 15
    }
    """
    global flow_analytics_engine
    
    if flow_analytics_engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "FlowAnalyticsEngine not initialized"}
        )
    
    try:
        # Recuperar anomalías del motor
        anomalies = flow_analytics_engine.get_anomaly_history(
            tester_id=tester_id,
            limit=limit,
            severity_filter=severity
        )
        
        logger.info(f"📋 {len(anomalies)} anomalías recuperadas para {tester_id}")
        
        return {
            "success": True,
            "tester_id": tester_id,
            "anomalies": anomalies,
            "total": len(anomalies),
            "status": "ok"
        }
        
    except Exception as e:
        logger.error(f"❌ Error recuperando anomalías: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
