# 🏗️ ARQUITECTURA: FlowAnalyticsEngine Integration

## 1. FLUJO DE DATOS GENERAL

```
┌──────────────────────────────────────────────────────────────────┐
│                      ANDROID CLIENT                              │
│  (Enviando eventos de accesibilidad)                             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (backend.py)                        │
│                                                                  │
│  POST /collect (AccessibilityEvent)                            │
│    ├─ raw_nodes (Árbol de UI)                                 │
│    ├─ session_key (Sesión actual)                             │
│    ├─ tester_id, build_id                                     │
│    └─ event_type_name (Tipo de evento)                        │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│        analyze_and_train() - Procesa Evento                     │
│                                                                  │
│  1. Normalizar evento                                           │
│  2. Generar árbol UI                                            │
│  3. Comparar con builds anteriores                              │
│  4. Entrenar modelos (KMeans, HMM, RandomForest)                │
└────────┬────────────────────────────────────────────────────────┘
         │
    ┌────┴───────────────────────────────────────────────┐
    │                                                    │
    ▼                                                    ▼
┌─────────────────────────────┐       ┌────────────────────────────┐
│  FLUJO VALIDATION (HMM)     │       │ FLOW ANALYTICS ENGINE      │
│                             │       │ (NEW - 3 Endpoints)       │
│ • FlowValidator.py          │       │                            │
│ • Validación secuencias     │       │ • Análisis avanzado        │
│ • Detección de patrones     │       │ • Desviaciones             │
│ • % Validez: 90%            │       │ • Sugerencias recovery     │
│                             │       │ • Dashboard hotspots       │
│ Retorna: ✅/❌              │       │ • Historial anomalías      │
└─────────────────────────────┘       │                            │
                                      │ Retorna:                   │
                                      │ - Calidad score (0-100)    │
                                      │ - Anomaly rate %           │
                                      │ - Suggestions + Recovery   │
                                      │ - Severity (Low/Med/High)  │
                                      └────┬─────────────────────────┘
                                           │
    ┌──────────────────────────────────────┴─────────────────┐
    │                                                        │
    ▼                                                        ▼
┌──────────────────────────────┐        ┌──────────────────────────┐
│  screen_diffs TABLE          │        │  flow_anomalies TABLE    │
│ (Cambios de UI)              │        │ (Anomalías de Flujo)     │
│                              │        │                          │
│ • diff_hash                  │        │ • app_name               │
│ • removed/added/modified     │        │ • tester_id              │
│ • diff_priority (high/low)   │        │ • flow_sequence          │
│ • approved_before            │        │ • deviation_point        │
│ • similarity_to_approved     │        │ • deviation_reason       │
│ • text_overlap               │        │ • recovery_suggestion    │
│ • screen_status              │        │ • severity               │
│ • created_at                 │        │ • similarity_score       │
└──────────────────────────────┘        │ • timestamp              │
                                        └──────────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  3 NEW ENDPOINTS │
                                    └──────────────────┘
```

---

## 2. ENDPOINTS ARCHITECTURE

```
Backend.py (FastAPI)
│
├── Port: 8000
│
├── 🟢 POST /flow-analyze/{app_name}/{tester_id}
│   │
│   ├─ Request Body (optional):
│   │  ├─ session_key: str
│   │  └─ flow_sequence: List[str]
│   │
│   └─ Response (200 OK):
│      ├─ success: bool
│      ├─ app_name: str
│      ├─ tester_id: str
│      └─ report: ReportDict
│         ├─ total_flows: int
│         ├─ quality_score: float (0-100)
│         ├─ anomaly_rate: float (0-1)
│         └─ suggestions: List[Suggestion]
│            ├─ type: str ("recovery"|"suggestion"|"warning")
│            ├─ screen: str
│            ├─ message: str
│            └─ severity: str ("low"|"medium"|"high")
│
├── 🟢 GET /flow-dashboard/{app_name}
│   │
│   ├─ Query Params: (none required)
│   │
│   └─ Response (200 OK):
│      ├─ success: bool
│      ├─ app_name: str
│      └─ dashboard: DashboardDict
│         ├─ total_flows: int
│         ├─ unique_screens: int
│         ├─ interruption_hotspots: List[Hotspot]
│         │  ├─ screen: str
│         │  ├─ anomaly_count: int
│         │  ├─ failure_rate: float
│         │  └─ top_reason: str
│         ├─ anomalies_summary: Dict
│         │  ├─ total: int
│         │  └─ by_severity: Dict[str, int]
│         └─ recommendations: List[str]
│
└── 🟢 GET /flow-anomalies/{tester_id}
    │
    ├─ Query Params:
    │  ├─ limit: int = 50 (1-500)
    │  └─ severity: Optional[str] ("low"|"medium"|"high")
    │
    └─ Response (200 OK):
       ├─ success: bool
       ├─ tester_id: str
       ├─ anomalies: List[Anomaly]
       │  ├─ id: int
       │  ├─ app_name: str
       │  ├─ flow_sequence: List[str]
       │  ├─ deviation_point: str
       │  ├─ deviation_reason: str
       │  ├─ recovery_suggestion: str
       │  ├─ severity: str
       │  ├─ similarity_score: float
       │  └─ timestamp: str (ISO)
       ├─ total: int
       └─ status: "ok"
```

---

## 3. CLASS HIERARCHY

```
┌────────────────────────────────────────────┐
│        FlowAnalyticsEngine                 │
│  (FlowAnalyticsEngine.py - 500+ lines)    │
├────────────────────────────────────────────┤
│ Private Attributes:                       │
│  • app_name: str                          │
│  • db_name: str ("accessibility.db")      │
│  • flow_anomalies_table_created: bool     │
│                                           │
│ Public Methods:                           │
│  • analyze_deviation()                    │
│  • generate_tester_flow_report()          │
│  • get_flow_analytics_dashboard()         │
│  • log_flow_anomaly()                     │
│  • get_anomaly_history()                  │
└────────────────────────────────────────────┘
         │
         ├─ Uses: SiameseEncoder
         │        (Embedding de árboles UI)
         │
         ├─ Uses: models_pipeline
         │        (KMeans, RandomForest, HMM)
         │
         ├─ Uses: SQLite3
         │        (flow_anomalies table)
         │
         └─ Uses: FlowValidator
                  (Para validación de línea base)
```

---

## 4. DATA FLOW EXAMPLE

### Escenario: Un tester experimenta una anomalía en checkout

```
1. ANDROID SENDS EVENT
   ┌─────────────────────────────────────┐
   │ POST /collect                       │
   │ {                                   │
   │   "eventTypeName": "ViewScrolled",  │
   │   "packageName": "com.rappi",       │
   │   "headerText": "Checkout",         │
   │   "collectNodeTree": [              │
   │     {...payment button...},         │
   │     {...disabled state...},         │
   │     {...}                           │
   │   ],                                │
   │   "actualDevice": "Pixel_6",        │
   │   "version": "8.19.3"               │
   │ }                                   │
   └─────────────────────────────────────┘
              │
              ▼
2. BACKEND PROCESSES
   ┌──────────────────────────────────────┐
   │ analyze_and_train()                  │
   │ • Compara con build anterior (8.18)  │
   │ • Detecta cambio: Payment button     │
   │   disabled cuando debería estar      │
   │   enabled                            │
   │ • Genera diff hash                   │
   │ • Entrena modelos incrementales      │
   └──────────────────────────────────────┘
              │
              ▼
3. FLOW ANALYTICS ENGINE KICKS IN
   ┌──────────────────────────────────────┐
   │ FlowAnalyticsEngine                  │
   │ • Analiza secuencia:                 │
   │   ["home","cart","checkout"]         │
   │ • Detecta: checkout button disabled  │
   │   (desviación)                       │
   │ • Calcula: similarity_score = 0.42   │
   │   (baja similitud = anomalía alta)   │
   │ • Asigna: severity = "HIGH"          │
   │ • Sugiere: Recovery = "Go back to    │
   │   cart and retry payment"            │
   │ • Registra en flow_anomalies         │
   └──────────────────────────────────────┘
              │
              ▼
4. DATA AVAILABLE IN DB
   ┌────────────────────────────────────────┐
   │ flow_anomalies table                   │
   │ INSERT INTO flow_anomalies VALUES (    │
   │   id=1,                                │
   │   app_name="com.rappi",                │
   │   tester_id="Pixel_6",                 │
   │   flow_sequence=                       │
   │     "['home','cart','checkout']",      │
   │   deviation_point="checkout",          │
   │   deviation_reason=                    │
   │     "Payment button disabled",         │
   │   severity="high",                     │
   │   similarity_score=0.42,               │
   │   recovery_suggestion=                 │
   │     "Go back to cart and retry",       │
   │   timestamp=NOW()                      │
   │ )                                      │
   └────────────────────────────────────────┘
              │
              ▼
5. ENDPOINTS EXPOSE DATA
   ┌───────────────────────────────────────┐
   │ GET /flow-anomalies/Pixel_6           │
   │                                       │
   │ Response: [                           │
   │   {                                   │
   │     "id": 1,                          │
   │     "deviation_point": "checkout",    │
   │     "deviation_reason":               │
   │       "Payment button disabled",      │
   │     "recovery_suggestion":            │
   │       "Go back to cart and retry",    │
   │     "severity": "high",               │
   │     "similarity_score": 0.42,         │
   │     "timestamp": "2024-01-15..."      │
   │   }                                   │
   │ ]                                     │
   └───────────────────────────────────────┘
              │
              ▼
6. QA/TESTER SEES FEEDBACK
   ┌─────────────────────────────────────┐
   │ "Your checkout flow had an issue:    │
   │                                     │
   │ 🔴 HIGH SEVERITY                    │
   │ Payment button was disabled          │
   │ unexpectedly during checkout.        │
   │                                     │
   │ 💡 HOW TO RECOVER:                  │
   │ Go back to cart and retry payment    │
   │                                     │
   │ 📊 QUALITY SCORE: 42/100            │
   │ Anomaly detected in flow sequence    │
   └─────────────────────────────────────┘
```

---

## 5. SYSTEM DEPENDENCIES

```
┌─ FlowAnalyticsEngine.py ─┐
│                          │
├─ Imports:               │
│  • sqlite3              │
│  • json                 │
│  • datetime             │
│  • numpy                │
│  • logging              │
│  • typing               │
│                          │
└─ Called by: backend.py   │
   (at startup & per       │
    endpoint)              │
```

```
┌─ backend.py ──────────────────┐
│                               │
├─ imports FlowAnalyticsEngine │
│  at line ~50                 │
│                               │
├─ initializes at              │
│  @app.on_event("startup")    │
│                               │
├─ exposes 3 endpoints:        │
│  1. POST /flow-analyze/...   │
│  2. GET /flow-dashboard/...  │
│  3. GET /flow-anomalies/...  │
│                               │
└─ stores in global:           │
   flow_analytics_engine       │
```

---

## 6. ERROR HANDLING FLOW

```
User Request
    │
    ▼
┌──────────────────────────────┐
│ Is FlowAnalyticsEngine NULL? │
└──────────┬───────────────────┘
           │
      ┌────┴─────┐
      │           │
   NO │           │ YES
      │           │
      ▼           ▼
   Process   Return 503
   Request    (Service
   Normally   Unavailable)
      │
      ▼
   ┌────────────────────┐
   │ Database Query OK? │
   └──────┬─────────────┘
          │
     ┌────┴─────┐
     │           │
   NO│           │ YES
     │           │
     ▼           ▼
  Return      Process
  500         Results
  (Error)     │
              ▼
         ┌──────────────┐
         │ Return 200   │
         │ with data    │
         └──────────────┘
```

---

## 7. PERFORMANCE CONSIDERATIONS

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| POST /flow-analyze | O(n) | n = histórico de flows |
| GET /flow-dashboard | O(n*m) | m = testers, n = flows |
| GET /flow-anomalies | O(log n) | Búsqueda indexada |
| analyze_deviation() | O(1) | Cálculo simple |
| generate_report() | O(n) | n = flows por tester |

**Optimizaciones:**
- ✅ Índices en DB: `(tester_id, app_name)`
- ✅ Límite de resultados: `limit ≤ 500`
- ✅ Paginación disponible (opcional)

---

## 8. VERSIONING & COMPATIBILITY

```
FlowAnalyticsEngine v1.0
│
├─ Compatible with:
│  ├─ Python 3.8+
│  ├─ FastAPI 0.70+
│  ├─ SQLite3 (standard)
│  └─ NumPy 1.20+
│
└─ Breaking Changes: None
   (New feature, fully backward compatible)
```

---

**✅ Arquitectura Completa y Funcional**

