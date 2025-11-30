# 🔹 INTEGRACIÓN: FlowAnalyticsEngine en Backend.py

## ✅ Cambios Realizados

### 1. **Importación en backend.py** (Línea ~50)
```python
# 🔹 NUEVA IMPORTACIÓN: Motor de análisis de flujos
try:
    from FlowAnalyticsEngine import FlowAnalyticsEngine
    logger.info("✅ FlowAnalyticsEngine importado correctamente")
except ImportError as e:
    logger.warning(f"⚠️  No se pudo importar FlowAnalyticsEngine: {e}")
    FlowAnalyticsEngine = None
```

### 2. **Inicialización en Startup** (Línea ~4710)
```python
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
```

### 3. **Tres Nuevos Endpoints** (Línea ~4720)

#### a) **POST /flow-analyze/{app_name}/{tester_id}**
Analiza los flujos de un tester y genera reporte con:
- Calidad de flujo (0-100)
- Tasa de anomalías detectadas
- Sugerencias de mejora
- Recuperación desde anomalías

**Request Body (opcional):**
```json
{
    "session_key": "tester_123_minute_block_xyz",
    "flow_sequence": ["home", "profile", "settings"]
}
```

**Response (200 OK):**
```json
{
    "success": true,
    "app_name": "com.grability.rappi",
    "tester_id": "tester_001",
    "report": {
        "tester_id": "tester_001",
        "total_flows": 42,
        "quality_score": 85.5,
        "anomaly_rate": 0.12,
        "suggestions": [
            {
                "type": "recovery",
                "screen": "profile",
                "message": "Button inactive detected - go back to home and retry"
            }
        ]
    },
    "status": "ok"
}
```

#### b) **GET /flow-dashboard/{app_name}**
Dashboard global de análisis de flujos:
- Distribución de pantallas por frecuencia
- Puntos de interrupción (hotspots)
- Anomalías totales detectadas
- Recomendaciones generales

**Query Parameters:** Ninguno (solo app_name en path)

**Response (200 OK):**
```json
{
    "success": true,
    "app_name": "com.grability.rappi",
    "dashboard": {
        "total_flows": 500,
        "unique_screens": 25,
        "interruption_hotspots": [
            {
                "screen": "checkout",
                "anomaly_count": 45,
                "failure_rate": 0.18,
                "top_reason": "Payment method not accepting"
            }
        ],
        "anomalies_summary": {
            "total": 87,
            "by_severity": {
                "low": 45,
                "medium": 32,
                "high": 10
            }
        },
        "recommendations": [
            "Improve checkout flow - 18% of users experiencing issues",
            "Add error handling for payment validation"
        ]
    },
    "status": "ok"
}
```

#### c) **GET /flow-anomalies/{tester_id}**
Historial de anomalías detectadas para un tester:

**Query Parameters:**
- `limit` (int, default=50): Máximo de registros (1-500)
- `severity` (str, optional): Filtrar por severidad (low|medium|high)

**Response (200 OK):**
```json
{
    "success": true,
    "tester_id": "tester_001",
    "anomalies": [
        {
            "id": 1,
            "app_name": "com.grability.rappi",
            "flow_sequence": ["home", "profile", "settings"],
            "deviation_point": "settings",
            "deviation_reason": "Settings button disabled unexpectedly",
            "recovery_suggestion": "Return to home screen and reopen settings",
            "severity": "high",
            "similarity_score": 0.42,
            "timestamp": "2024-01-15T10:30:00"
        },
        ...
    ],
    "total": 15,
    "status": "ok"
}
```

---

## 🎯 Cómo Usar los Endpoints

### Ejemplo 1: Obtener reporte de flujos de un tester
```bash
curl -X POST "http://localhost:8000/flow-analyze/com.grability.rappi/tester_001" \
  -H "Content-Type: application/json" \
  -d '{
    "session_key": "tester_001_session_123",
    "flow_sequence": ["home", "profile"]
  }'
```

### Ejemplo 2: Obtener dashboard global
```bash
curl -X GET "http://localhost:8000/flow-dashboard/com.grability.rappi"
```

### Ejemplo 3: Obtener anomalías de un tester (filtradas por severidad)
```bash
curl -X GET "http://localhost:8000/flow-anomalies/tester_001?limit=20&severity=high"
```

---

## 🧪 Testing

Ejecutar el script de prueba:
```bash
python test_flow_analytics_endpoints.py
```

Este script valida:
1. ✅ POST /flow-analyze/{app_name}/{tester_id}
2. ✅ GET /flow-dashboard/{app_name}
3. ✅ GET /flow-anomalies/{tester_id}

---

## 📊 Características del FlowAnalyticsEngine

### Integrados en backend.py:
1. **Análisis de Desviaciones**: Detecta dónde los flujos se desvían del patrón esperado
2. **Reportes por Tester**: Calidad, anomalías, sugerencias personalizadas
3. **Dashboard Global**: Visión general de problemas en la app
4. **Historial de Anomalías**: Seguimiento temporal de problemas
5. **Recuperación**: Sugerencias de recuperación ante anomalías

### Base de datos:
- Nueva tabla: `flow_anomalies` (creada automáticamente)
- Columns:
  - `id` (PRIMARY KEY)
  - `app_name`, `tester_id`, `flow_sequence`
  - `deviation_point`, `deviation_reason`
  - `severity`, `similarity_score`
  - `recovery_suggestion`, `timestamp`

---

## ⚙️ Configuración

El motor se inicializa automáticamente en el evento `@app.on_event("startup")`.

Si necesitas reinicializar manualmente:
```python
from FlowAnalyticsEngine import FlowAnalyticsEngine
flow_analytics_engine = FlowAnalyticsEngine(app_name="com.grability.rappi")
```

---

## 🔗 Relación con Otros Sistemas

- **incremental_feedback_system**: Feedback de aprobaciones/rechazos
- **FlowValidator**: Validación básica de secuencias (HMM)
- **FlowAnalyticsEngine**: Análisis avanzado con desviaciones y anomalías

El FlowAnalyticsEngine **complementa** a FlowValidator, no lo reemplaza:
- FlowValidator: ¿El flujo es válido? (sí/no)
- FlowAnalyticsEngine: ¿Qué salió mal y cómo recuperarse?

---

## 📝 Notas de Implementación

1. **Thread-safe**: Usa locks para operaciones de BD
2. **Manejo de errores**: Devuelve 503 si FlowAnalyticsEngine no está inicializado
3. **Logging**: Todos los eventos registrados en el logger
4. **Async**: Compatible con async/await de FastAPI
5. **Backward compatible**: No afecta endpoints existentes

---

## ✅ Estado

- ✅ FlowAnalyticsEngine.py creado (500+ líneas)
- ✅ Importado en backend.py
- ✅ 3 nuevos endpoints agregados
- ✅ Script de prueba creado
- ⏳ Pruebas con servidor en ejecución (pendiente)

**Próximo paso:** Ejecutar servidor FastAPI y validar endpoints con test_flow_analytics_endpoints.py

