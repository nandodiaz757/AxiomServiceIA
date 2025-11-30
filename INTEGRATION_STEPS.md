# Instrucciones para Integrar Automation Routes en backend.py

## Paso 1: Agregar los Imports

En la sección superior de `backend.py`, después de los imports existentes, agrega:

```python
# ============================================
# IMPORTS PARA AUTOMATION INTEGRATION
# ============================================
from automation_endpoints import router as automation_router
from automation_endpoints import setup_automation_routes
from session_manager import init_session_manager, get_session_manager
```

## Paso 2: Modificar la Función Lifespan

Busca la función `lifespan()` (alrededor de línea 720) y modifica el bloque STARTUP:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================================
    # STARTUP
    # ==========================================
    logger.info("🚀 Iniciando Axiom Backend...")
    
    # Inicializar config manager (EXISTENTE)
    config = init_config()
    
    # ✨ NUEVO: Inicializar session manager para automatización
    session_mgr = init_session_manager()
    setup_automation_routes()
    logger.info("✨ Automation routes inicializadas")
    
    # ... resto del código STARTUP existente ...
    
    yield
    
    # ==========================================
    # SHUTDOWN
    # ==========================================
    logger.info("🛑 Apagando Axiom Backend...")
    # ... código SHUTDOWN existente ...
```

## Paso 3: Registrar el Router de Automatización

Busca donde se define `app = FastAPI(lifespan=lifespan)` (alrededor de línea 745) y agrega DESPUÉS:

```python
# ==============================
# APP PRINCIPAL
# ==============================

app = FastAPI(lifespan=lifespan)

# 🔹 INICIALIZAR CONFIG MANAGER AL STARTUP
config = init_config()

# ✨ NUEVO: Registrar router de automatización
app.include_router(automation_router)
```

## Paso 4: (OPCIONAL) Integrar con /collect

Si quieres que los eventos del `/collect` endpoint también se procesen en sesiones activas de automatización, busca la función `@app.post("/collect")` y agrega al inicio del try block:

```python
@app.post("/collect")
async def collect_event(event: AccessibilityEvent, background_tasks: BackgroundTasks):
    # ... código existente ...
    
    try:
        # ✨ NUEVO: Si hay una sesión de automatización activa para este app/build,
        # procesar evento también allí
        try:
            session_mgr = get_session_manager()
            for sid, session in session_mgr.sessions.items():
                if (session.app_name == (event.package_name or "default") and 
                    session.build_id == (event.build_id or "unknown") and
                    session.status == SessionStatus.RUNNING):
                    
                    # Procesar evento también en la sesión de automatización
                    await session_mgr.process_event(
                        session_id=sid,
                        screen_name=event.header_text or "",
                        header_text=event.header_text or "",
                        event_type="app_event",
                        additional_data={
                            "from_app": True,
                            "package_name": event.package_name
                        }
                    )
        except Exception as e:
            logger.debug(f"Info: No hay sesión activa de automatización: {e}")
        
        # ... resto del código existente ...
```

## Verificación

Después de agregar los cambios, verifica:

```bash
# 1. El servidor debe iniciar sin errores
python -m uvicorn backend:app --host 0.0.0.0 --port 8000

# 2. Los endpoints deben estar disponibles
curl http://localhost:8000/api/automation/stats

# Debe retornar:
# {
#   "total_sessions": 0,
#   "active_sessions": 0,
#   "completed_sessions": 0,
#   "failed_sessions": 0,
#   "total_events": 0,
#   "avg_flow_completion": 0
# }

# 3. Prueba crear una sesión
curl -X POST http://localhost:8000/api/automation/session/create \
  -H "Content-Type: application/json" \
  -d '{
    "test_name": "Test Integration",
    "tester_id": "test_bot",
    "build_id": "1.0.0",
    "app_name": "com.test.app",
    "expected_flow": ["screen1", "screen2"]
  }'

# Debe retornar algo como:
# {
#   "session_id": "A1B2C3D4",
#   "test_name": "Test Integration",
#   "status": "created",
#   ...
# }
```

## Importaciones Necesarias

Asegúrate de que en el top de `backend.py` tengas:

```python
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
```

## Troubleshooting

### Error: "No module named 'automation_endpoints'"
- Verifica que `automation_endpoints.py` existe en el mismo directorio que `backend.py`
- Verifica que no hay typos en el nombre del archivo

### Error: "No module named 'session_manager'"
- Verifica que `session_manager.py` existe en el mismo directorio

### Error: "SessionStatus is not defined"
- Agrega al import:
  ```python
  from session_manager import init_session_manager, get_session_manager, SessionStatus
  ```

### Los endpoints no responden
- Verifica que `setup_automation_routes()` se llamó en el startup
- Revisa los logs: `python -m uvicorn backend:app --log-level debug`

## Próximas Pruebas

Después de integración, prueba con:

```bash
# 1. Crea una sesión
curl -X POST http://localhost:8000/api/automation/session/create ...

# 2. Inicia la sesión
curl -X POST http://localhost:8000/api/automation/session/ABC123/start

# 3. Registra un evento
curl -X POST http://localhost:8000/api/automation/session/ABC123/event \
  -H "Content-Type: application/json" \
  -d '{"screen_name": "screen1"}'

# 4. Finaliza
curl -X POST http://localhost:8000/api/automation/session/ABC123/end \
  -H "Content-Type: application/json" \
  -d '{"success": true, "final_status": "completed"}'
```

## Documentación Referenciada

- `AUTOMATION_INTEGRATION_GUIDE.md` - Guía completa para testers
- `ARCHITECTURE.md` - Arquitectura del sistema
- `automation_endpoints.py` - Código de endpoints
- `session_manager.py` - Gestor de sesiones

---

¡Listo! Tu backend ahora soporta integración con tests automatizados. 🚀
