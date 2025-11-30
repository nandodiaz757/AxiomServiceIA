# 🏗️ Arquitectura Completa: Axiom + Automatización

## Visión General

**Axiom Automation Integration** es un sistema que permite que **tests automatizados ejecuten en paralelo con validaciones automáticas de flujos y accesibilidad**, sin modificar el código de tus tests existentes.

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                  AUTOMATION TESTERS                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │ Selenium Python  │  │ Selenide Java    │  │ JUnit/... │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───┘ │
│           │                     │                    │       │
│           └─────────────────────┼────────────────────┘       │
│                                 │                            │
│                    AxiomTestSession (SDK)                    │
│                   (Cliente HTTP + Métodos)                   │
└─────────────────────────────────────────────────────────────┘
                                   │
                    HTTP REST API (JSON)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│            AXIOM BACKEND SERVICE (FastAPI)                  │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │        automation_endpoints.py                      │   │
│  │  • POST /api/automation/session/create             │   │
│  │  • POST /api/automation/session/{id}/start         │   │
│  │  • POST /api/automation/session/{id}/event         │   │
│  │  • POST /api/automation/session/{id}/validation    │   │
│  │  • POST /api/automation/session/{id}/end           │   │
│  │  • GET  /api/automation/sessions                   │   │
│  │  • GET  /api/automation/stats                      │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │        session_manager.py                          │   │
│  │                                                    │   │
│  │  • SessionManager (Singleton)                      │   │
│  │    - create_session()                             │   │
│  │    - start_session()                              │   │
│  │    - process_event()      ◄── Validación         │   │
│  │    - end_session()        ◄── Reporte            │   │
│  │    - add_validation()                             │   │
│  │                                                    │   │
│  │  • Persistencia en BD (SQLite)                    │   │
│  │    - test_sessions                                │   │
│  │    - session_events                               │   │
│  │    - session_validations                          │   │
│  │    - session_reports                              │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│             ┌─────────────────────────┐                    │
│             │  FlowValidator          │                    │
│             │  (Análisis en Tiempo    │                    │
│             │   Real de Flujos)       │                    │
│             └─────────────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│          APP UNDER TEST (Rappi / Web App)                  │
│                                                             │
│  • Envía eventos de accesibilidad (normales)              │
│  • Axiom los procesa de forma transparente                │
│  • Los combina con eventos del tester automatizado        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Flujo de Datos

### 1. Crear Sesión

```
Cliente (Selenium/Selenide)
    │
    │ POST /api/automation/session/create
    │ {
    │   test_name: "Login Flow",
    │   tester_id: "selenium_bot_01",
    │   build_id: "1.0.0",
    │   expected_flow: ["login_screen", "home_screen"]
    │ }
    │
    ▼
SessionManager
    │
    ├─ Crear objeto TestSession
    │  • session_id: "A1B2C3D4"
    │  • status: CREATED
    │
    ├─ Guardar en memoria (dict)
    │
    └─ Persistir en BD
       └─ INSERT INTO test_sessions
              session_id, test_name, status, ...
    
    │
    ▼
Response → sessionId: "A1B2C3D4"
```

### 2. Iniciar Sesión

```
Cliente: POST /api/automation/session/A1B2C3D4/start
    │
    ▼
SessionManager.start_session()
    │
    ├─ Cambiar status: CREATED → RUNNING
    ├─ Set started_at = now()
    └─ UPDATE en BD
    
    │
    ▼
Response → status: "running"
```

### 3. Registrar Eventos (Durante el Test)

```
┌─ Cliente ejecuta test y navega a login_screen
│
├─ Cliente: POST /api/automation/session/A1B2C3D4/event
│                   { screen_name: "login_screen" }
│
├─ App también envía evento normal a /collect
│  (Axiom recibe ambos)
│
▼
SessionManager.process_event()
    │
    ├─ expected_flow[0] = "login_screen"  ✅ MATCH
    ├─ Incrementar flow_position: 0 → 1
    ├─ events_validated: 0 → 1
    │
    ├─ Guardar en session_events (BD)
    │  INSERT INTO session_events
    │      event_id, screen_name, validation_result='match', ...
    │
    └─ Llamar callbacks registrados
    
    │
    ▼
Response → {
  validation_result: "match",
  message: "✅ Evento coincide: login_screen (pos 1/2)"
}
```

### 4. Validaciones Adicionales

```
Cliente: POST /api/automation/session/A1B2C3D4/validation
    {
      validation_name: "Login fields visible",
      rule: { email_field: true, password_field: true },
      passed: true
    }
    │
    ▼
SessionManager.add_validation()
    │
    ├─ Guardar en session_validations (BD)
    │
    └─ Retornar confirmación
```

### 5. Finalizar Sesión

```
Cliente: POST /api/automation/session/A1B2C3D4/end
    {
      success: true,
      final_status: "completed"
    }
    │
    ▼
SessionManager.end_session()
    │
    ├─ status: RUNNING → COMPLETED
    ├─ ended_at = now()
    │
    ├─ Calcular métricas:
    │  • duration_seconds = ended_at - started_at
    │  • flow_completion_percentage = flow_position / expected_flow.length * 100
    │  • errors_count = len(validation_errors)
    │  • success = (errors_count == 0)
    │
    ├─ Guardar reporte en BD
    │  INSERT INTO session_reports
    │      report_id, session_id, summary={...}, ...
    │
    └─ Retornar reporte completo
    
    │
    ▼
Response → {
  session_id: "A1B2C3D4",
  status: "completed",
  duration_seconds: 45.23,
  events_received: 8,
  events_validated: 8,
  flow_completion_percentage: 100,
  validation_errors: [],
  success: true
}
```

---

## 🔄 Validación en Tiempo Real

### Algoritmo de Validación

```python
def process_event(session, screen_name):
    # 1. Obtener pantalla esperada en posición actual
    expected = expected_flow[flow_position]
    
    # 2. Comparar (normalizado)
    if normalize(screen_name) == normalize(expected):
        # ✅ MATCH - Pantalla correcta en orden correcto
        result = MATCH
        flow_position += 1
        events_validated += 1
        
    elif screen_name in expected_flow:
        # ⚠️ UNEXPECTED - Pantalla esperada pero en orden incorrecto
        result = UNEXPECTED
        errors.append({
            type: "unexpected_screen",
            received: screen_name,
            expected: expected
        })
        
    else:
        # ❌ ANOMALY - Pantalla no en flujo esperado
        result = ANOMALY
        errors.append({
            type: "anomaly_screen",
            received: screen_name
        })
    
    return result
```

### Ejemplos de Validación

```
Flujo esperado: ["login", "home", "cart", "checkout"]

Test 1: Flujo correcto
  1. "login"     ✅ MATCH (pos 1/4)
  2. "home"      ✅ MATCH (pos 2/4)
  3. "cart"      ✅ MATCH (pos 3/4)
  4. "checkout"  ✅ MATCH (pos 4/4)
  Resultado: ✅ 100% completado

Test 2: Pantalla extra
  1. "login"     ✅ MATCH (pos 1/4)
  2. "home"      ✅ MATCH (pos 2/4)
  3. "settings"  ❌ ANOMALY (no en flujo)
  4. "cart"      ✅ MATCH (pos 3/4)
  5. "checkout"  ✅ MATCH (pos 4/4)
  Resultado: ⚠️ 100% pero 1 anomalía

Test 3: Orden incorrecto
  1. "login"     ✅ MATCH (pos 1/4)
  2. "checkout"  ⚠️ UNEXPECTED (esperado: home)
  3. "home"      ✅ MATCH (pos 2/4)
  4. "cart"      ✅ MATCH (pos 3/4)
  5. "checkout"  ✅ MATCH (pos 4/4)
  Resultado: ⚠️ 100% pero orden incorrecto

Test 4: Flujo incompleto
  1. "login"     ✅ MATCH (pos 1/4)
  2. "home"      ✅ MATCH (pos 2/4)
  [test termina]
  Resultado: ❌ 50% completado (faltan: cart, checkout)
```

---

## 💾 Modelo de Datos

### Tablas en BD

#### `test_sessions`
```sql
CREATE TABLE test_sessions (
    session_id TEXT PRIMARY KEY,
    test_name TEXT,
    tester_id TEXT,
    build_id TEXT,
    app_name TEXT,
    expected_flow TEXT,  -- JSON ["login", "home"]
    status TEXT,         -- created, running, completed, failed, error
    events_received INTEGER,
    events_validated INTEGER,
    flow_position INTEGER,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    screen_sequence TEXT,  -- JSON ["login", "home", "cart"]
    validation_errors TEXT,  -- JSON [{type: ..., ...}]
    metadata TEXT  -- JSON {browser: "Chrome", ...}
)
```

#### `session_events`
```sql
CREATE TABLE session_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT,
    screen_name TEXT,
    header_text TEXT,
    event_type TEXT,
    validation_result TEXT,  -- match, unexpected, anomaly
    expected TEXT,
    actual TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES test_sessions
)
```

#### `session_validations`
```sql
CREATE TABLE session_validations (
    validation_id TEXT PRIMARY KEY,
    session_id TEXT,
    validation_name TEXT,
    rule TEXT,  -- JSON
    passed INTEGER,  -- 0/1
    error_message TEXT,
    evaluated_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES test_sessions
)
```

#### `session_reports`
```sql
CREATE TABLE session_reports (
    report_id TEXT PRIMARY KEY,
    session_id TEXT,
    summary TEXT,  -- JSON completo del reporte
    total_events INTEGER,
    matched_events INTEGER,
    unexpected_events INTEGER,
    flow_completion_percentage REAL,
    generated_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES test_sessions
)
```

---

## 🔌 Integración con Código Existente

### Backend.py - Cambios Necesarios

En el archivo principal `backend.py`, agregar:

```python
# ============================================
# 1. IMPORTAR NUEVOS MÓDULOS
# ============================================

from automation_endpoints import router as automation_router
from automation_endpoints import setup_automation_routes
from session_manager import init_session_manager, get_session_manager


# ============================================
# 2. EN LA FUNCIÓN LIFESPAN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("🚀 Iniciando Axiom Backend...")
    
    # Inicializar config manager (existente)
    config = init_config()
    
    # ✨ NUEVO: Inicializar session manager
    session_mgr = init_session_manager()
    setup_automation_routes()
    
    # ... resto del código ...
    
    yield
    
    # SHUTDOWN
    logger.info("🛑 Apagando Axiom Backend...")


# ============================================
# 3. REGISTRAR ROUTER
# ============================================

app = FastAPI(lifespan=lifespan)

# Rutas de config (existentes)
@app.get("/api/config")
...

# ✨ NUEVO: Rutas de automatización
app.include_router(automation_router)

# Rutas de colección (existentes)
@app.post("/collect")
...
```

### Integración con /collect

El endpoint `/collect` existente **continúa funcionando igual**, pero ahora:

```python
@app.post("/collect")
async def collect_event(event: AccessibilityEvent, background_tasks: BackgroundTasks):
    
    # Lógica existente...
    
    # ✨ NUEVO: Si hay una sesión de automatización activa, también procesar ahí
    try:
        session_mgr = get_session_manager()
        # Buscar si hay sesión activa para este build/app
        for sid, session in session_mgr.sessions.items():
            if (session.app_name == event.package_name and 
                session.build_id == event.build_id and
                session.status == SessionStatus.RUNNING):
                
                # Procesar evento también en la sesión
                await session_mgr.process_event(
                    session_id=sid,
                    screen_name=event.header_text,
                    header_text=event.header_text,
                    additional_data={"app_event": True}
                )
    except Exception as e:
        logger.debug(f"Info: No hay sesión activa de automatización: {e}")
    
    # Continuar con lógica normal...
```

---

## 📦 Archivos Creados

```
AxiomServiceIA/
├── session_manager.py              ← Gestor de sesiones
├── axiom_test_client.py            ← Cliente SDK Python
├── automation_endpoints.py          ← Endpoints FastAPI
├── examples/
│   ├── selenium_example.py         ← Ejemplo Selenium
│   ├── RappiFlowTest.java          ← Ejemplo Selenide + TestNG
│   ├── AxiomTestSession.java       ← Cliente SDK Java
│   └── TestResult.java             ← Clase de resultados
├── AUTOMATION_INTEGRATION_GUIDE.md ← Documentación completa
└── ARCHITECTURE.md                 ← Este archivo
```

---

## 🔍 Casos de Uso

### Caso 1: Test de E2E Simple

```python
# test_login.py
from axiom_test_client import AxiomTestSession
from selenium import webdriver

session = AxiomTestSession(
    test_name="Login Test",
    expected_flow=["login_screen", "home_screen"]
)
session.create()
session.start()

driver = webdriver.Chrome()
driver.get("https://app.example.com/login")
session.record_event("login_screen")

# ... login logic ...

driver.get("https://app.example.com/home")
session.record_event("home_screen")

result = session.end()
# Reporte automático con validación de flujo
```

### Caso 2: Suite Completa con Múltiples Tests

```
┌─ Test Suite
│
├─ test_login.py
│   ├─ Session A (expected: ["login", "home"])
│   ├─ Session B (expected: ["login", "home"])
│   └─ Session C (expected: ["login", "home"])
│
├─ test_checkout.py
│   ├─ Session D (expected: ["home", "cart", "checkout"])
│   └─ Session E (expected: ["home", "cart", "checkout"])
│
└─ Axiom Backend gestiona 5 sesiones en paralelo
   • Recibe eventos de todos los tests
   • Valida flujos independientemente
   • Genera 5 reportes
```

### Caso 3: Integración CI/CD

```yaml
# GitHub Actions / GitLab CI
jobs:
  automation_tests:
    steps:
      - name: Start Axiom Backend
        run: python -m uvicorn backend:app &
      
      - name: Wait for backend
        run: sleep 5
      
      - name: Run Selenium Tests
        run: pytest tests/ --axiom-server=http://localhost:8000
      
      - name: Generate Report
        run: python scripts/generate_axiom_report.py
      
      - name: Upload Results
        run: aws s3 cp axiom_report.json s3://bucket/reports/
```

---

## 🚨 Manejo de Errores

### Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| Connection refused | Axiom no corriendo | `python -m uvicorn backend:app` |
| Session not found | ID incorrecto/expirado | Verificar session_id |
| Flow mismatch | Orden incorrecto | Verificar expected_flow |
| Timeout | Backend lento | Aumentar timeout en cliente |

### Recuperación

```python
try:
    session.record_event("screen")
except ConnectionError:
    logger.warn("Lost connection, retrying...")
    time.sleep(2)
    session.record_event("screen")  # Reintentar
```

---

## 📈 Métricas y Monitoreo

### Endpoints de Stats

```bash
# Ver estadísticas generales
GET /api/automation/stats

{
  "total_sessions": 42,
  "active_sessions": 3,
  "completed_sessions": 35,
  "failed_sessions": 4,
  "total_events": 512,
  "avg_flow_completion": 98.5
}
```

### Dashboard Potencial (Futuro)

```
┌─────────────────────────────────────┐
│ Axiom Automation Dashboard          │
├─────────────────────────────────────┤
│                                     │
│ Active Sessions: 5/50               │
│ ████████░░░░░░░░░░░░░░░░░░░░░░     │
│                                     │
│ Success Rate: 96.3%                 │
│ ██████████████████░░░░░░░░░░░░░░   │
│                                     │
│ Avg Flow Completion: 98.7%          │
│ ████████████████████░░░░░░░░░░░░   │
│                                     │
│ Recent Sessions:                    │
│ ✅ test_login (2m 34s)             │
│ ✅ test_checkout (3m 12s)          │
│ ⚠️  test_profile (Partial)          │
│ ❌ test_search (Failed)             │
│                                     │
└─────────────────────────────────────┘
```

---

## 🔐 Seguridad

### Consideraciones

- Sessions expiran tras 24 horas de inactividad
- IDs de sesión son opacos (UUID-like)
- No se guardan credenciales del usuario
- Metadata encriptada en BD (futuro)

### Limpieza

```bash
# Limpiar sesiones expiradas
POST /api/automation/cleanup/expired?max_age_hours=24

# Verificar antes
GET /api/automation/sessions?status=abandoned
```

---

## 🎯 Próximas Características

- [ ] WebSocket para eventos en tiempo real
- [ ] Dashboard web con live metrics
- [ ] Integración con Slack/Teams para notificaciones
- [ ] Análisis de anomalías con ML
- [ ] Export a reportes HTML/PDF
- [ ] Comparación entre ejecuciones
- [ ] Integración con Jira para crear issues automáticos

---

**Resumen**: El sistema está diseñado para ser **transparente, escalable y no invasivo** con tus tests automatizados existentes. Agrégalo donde necesites validación de flujos. 🚀
