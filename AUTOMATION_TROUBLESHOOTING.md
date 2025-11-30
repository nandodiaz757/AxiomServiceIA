# 🛠️ Troubleshooting - Guía de Problemas y Soluciones

## 📋 Índice Rápido

- [Problemas de Conectividad](#problemas-de-conectividad)
- [Errores de Sesión](#errores-de-sesión)
- [Problemas de Eventos](#problemas-de-eventos)
- [Anomalías Falsas Positivas](#anomalías-falsas-positivas)
- [Performance y Timeouts](#performance-y-timeouts)
- [Problemas de Base de Datos](#problemas-de-base-de-datos)
- [Debugging](#debugging)

---

## Problemas de Conectividad

### ❌ Problema: "Connection refused" al conectar con Axiom

```
Error: Failed to connect to http://localhost:8000
```

**Causa:** El servicio no está ejecutándose en el puerto 8000.

**Soluciones:**

1. Verificar que el servicio está activo:
```powershell
# En PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/docs" -Method GET
```

2. Si no responde, iniciar el servicio:
```powershell
python -m debugpy --listen 5678 -m uvicorn backend:app --host 0.0.0.0 --port 8000
```

3. Verificar si el puerto 8000 está en uso:
```powershell
netstat -ano | findstr :8000
# Si aparece algo, matar el proceso:
taskkill /PID <PID> /F
```

4. Cambiar puerto si es necesario:
```powershell
python -m uvicorn backend:app --host 0.0.0.0 --port 8001
```

---

### ❌ Problema: "Network error" en cliente Python/Java

```python
# Python
requests.exceptions.ConnectionError: 
Max retries exceeded with url: /api/automation/session/create
```

**Soluciones:**

1. Verificar URL correcta en cliente:
```python
# ❌ INCORRECTO
session = AxiomTestSession(axiom_url="localhost:8000")

# ✅ CORRECTO
session = AxiomTestSession(axiom_url="http://localhost:8000")
```

2. Verificar firewall:
```powershell
# Permitir puerto en firewall
netsh advfirewall firewall add rule name="Allow Axiom" dir=in action=allow protocol=tcp localport=8000
```

3. Si está remoto, usar IP real:
```python
session = AxiomTestSession(axiom_url="http://192.168.1.100:8000")
```

---

## Errores de Sesión

### ❌ Problema: "Session not found"

```json
{
  "detail": "Session not found: qa_tester_01_1701345600"
}
```

**Causas posibles:**

1. **Session ID incorrecto**: Verificar que copió bien el ID
```bash
# ✅ Formato correcto
qa_tester_01_1701345600

# ❌ Formatos incorrectos
qa_tester_01  # Falta timestamp
qa_tester_01_170134560  # Timestamp truncado
```

2. **Session expirada**: Las sesiones se limpian después de 24 horas
```python
# Solución: Crear nueva sesión
session = AxiomTestSession(...)
session.create()
```

3. **BD corrupta**: Si la sesión se perdió
```bash
# Verificar en DB
sqlite3 axiom.db "SELECT COUNT(*) FROM test_sessions;"
```

**Solución:**

```python
# Siempre guardar el session_id
from axiom_test_client import AxiomTestSession

session = AxiomTestSession(...)
session.create()

# Guardar ID para recuperar después
SESSION_ID = session.session_id
print(f"Session ID guardado: {SESSION_ID}")

# Luego puedes recuperarlo
session2 = AxiomTestSession(session_id=SESSION_ID)
session2.get_status()
```

---

### ❌ Problema: "Cannot start session in state: COMPLETED"

```json
{
  "detail": "Cannot start session in state: COMPLETED"
}
```

**Causa:** Intentando iniciar una sesión que ya fue finalizada.

**Solución:**

```python
# ❌ INCORRECTO - Intentar reusar sesión terminada
session.create()
session.start()
session.end()

session.start()  # ❌ ERROR: Ya está COMPLETED

# ✅ CORRECTO - Crear nueva sesión
session1.create()
session1.start()
session1.end()

session2 = AxiomTestSession(...)  # Nueva instancia
session2.create()
session2.start()
```

---

### ❌ Problema: "Missing required field: expected_flow"

```json
{
  "detail": "Missing required field: expected_flow"
}
```

**Causa:** No proporcionar el flujo esperado al crear sesión.

**Solución:**

```python
# ❌ INCORRECTO
session = AxiomTestSession(
    tester_id="qa_01",
    build_id="v2.0.0"
)

# ✅ CORRECTO
session = AxiomTestSession(
    tester_id="qa_01",
    build_id="v2.0.0",
    app_name="com.example.app",
    expected_flow=["screen_a", "screen_b", "screen_c"]  # ← REQUERIDO
)
```

---

## Problemas de Eventos

### ❌ Problema: Evento registrado pero validación es MISSING

```json
{
  "validation_result": "MISSING",
  "message": "Expected screen was skipped in flow"
}
```

**Causa:** El evento recibido no está en la secuencia esperada.

**Ejemplo:**
```
Expected flow: ["login", "home", "profile"]
Received:      ["login", "profile"]  ← Falta "home"
```

**Solución:**

1. Verificar que eventos se registren en orden:
```python
# ✅ CORRECTO - Registrar en orden
session.record_event("login")
session.record_event("home")       # No saltarse pasos
session.record_event("profile")
```

2. Si debe saltarse un paso, actualizar expected_flow:
```python
# ✅ CORRECTO - Flujo flexible
expected_flow = [
    "login",
    "home",  # Opcional
    "profile"
]
```

---

### ❌ Problema: Evento registrado pero validación es UNEXPECTED

```json
{
  "validation_result": "UNEXPECTED",
  "anomaly_score": 0.65,
  "message": "Event not in expected flow"
}
```

**Causa:** Registraste un evento que no está en expected_flow.

**Ejemplo:**
```
Expected flow: ["login", "home", "profile"]
Received:      ["login", "ad_popup", "home"]  ← Ad popup no esperado
```

**Soluciones:**

1. Agregar el evento al flujo esperado:
```python
expected_flow = [
    "login",
    "home",
    "ad_popup",  # ← Agregar si es normal
    "profile"
]
```

2. O ignoral como conocido:
```python
# En el código de prueba
if event_name == "ad_popup":
    # Ignorar y continuar sin registrar
    pass
else:
    session.record_event(event_name)
```

3. Si es una anomalía real que quieres detectar, dejarla así:
```python
# Registrar como está - Axiom la marcará como UNEXPECTED
# Esto es útil para detectar regresiones
session.record_event("ad_popup")  # Axiom detecta que no era esperada
```

---

### ❌ Problema: event_name vacío o None

```python
# ❌ INCORRECTO
session.record_event("")
session.record_event(None)
```

**Solución:**

```python
# ✅ CORRECTO
event_name = screen_name.strip() if screen_name else "unknown_screen"
session.record_event(event_name)

# Validar antes
if not event_name or len(event_name) < 3:
    print("❌ Event name inválido")
else:
    session.record_event(event_name)
```

---

## Anomalías Falsas Positivas

### ❌ Problema: Anomalías detectadas en pantallas normales

**Síntoma:** `anomaly_score` muy alto (> 0.5) pero el test se ve normal.

**Causas comunes:**

1. **Elementos UI cambiaron ligeramente** (color, posición, tamaño)
   - Esto es **esperado** y normal
   - Si no afecta funcionalidad, puede ignorarse

2. **Elemento inesperado pero harmless** (ads, analytics, etc.)
   - Agregar a lista blanca de elementos conocidos

3. **Diferencias en dispositivo/versión SO**
   - Normalizar elementos antes de validar

**Solución:**

```python
# Filtrar anomalías conocidas/harmless
KNOWN_ANOMALIES = {
    "analytics_tracker",
    "ad_banner",
    "tracking_pixel",
    "debug_indicator"
}

def is_ignorable_anomaly(element_name):
    return element_name in KNOWN_ANOMALIES

# Usar en validación
if not is_ignorable_anomaly(event.element_name):
    session.record_event(event_name)
```

---

### ❌ Problema: El mismo test falla a veces, a veces pasa

**Causa:** Race conditions o timing issues.

**Solución:**

```python
import time

# ✅ AGREGAR WAITS ESTRATÉGICOS
session.record_event("screen_a")
time.sleep(0.5)  # Esperar a que UI se estabilice

session.record_event("screen_b")
time.sleep(0.5)

# O usar waits explícitos
def wait_for_screen(session, screen_name, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            session.record_event(screen_name)
            return True
        except:
            time.sleep(0.2)
    return False
```

---

## Performance y Timeouts

### ❌ Problema: "Request timeout" en sesiones largas

```python
Error: Timeout waiting for response from http://localhost:8000/api/automation/session/end
```

**Causa:** Session muy grande con muchos eventos/validaciones.

**Soluciones:**

1. Aumentar timeout del cliente:
```python
# Python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)

# Usar con timeout mayor
response = session.get(url, timeout=30)  # 30 segundos
```

2. Java - Aumentar timeout:
```java
OkHttpClient client = new OkHttpClient.Builder()
    .connectTimeout(30, TimeUnit.SECONDS)
    .readTimeout(60, TimeUnit.SECONDS)
    .build();
```

3. Dividir en múltiples sesiones más pequeñas:
```python
# En lugar de 1 sesión con 100 eventos
# Hacer 5 sesiones con 20 eventos cada una
for batch in range(0, 100, 20):
    session = AxiomTestSession(...)
    session.create()
    session.start()
    # Procesar 20 eventos
    session.end()
```

---

### ❌ Problema: API lenta / Respuestas lentas

**Síntoma:** Cada request toma >1 segundo.

**Diagnóstico:**

```bash
# Medir tiempos
curl -w "tiempo_total: %{time_total}s\n" -X GET http://localhost:8000/api/automation/stats

# Si > 2s, problema de BD o servidor
```

**Soluciones:**

1. Verificar BD no está corrupta:
```bash
sqlite3 axiom.db "PRAGMA integrity_check;"
```

2. Limpiar sesiones viejas:
```bash
curl -X POST http://localhost:8000/api/automation/cleanup/expired \
  -H "Content-Type: application/json" \
  -d '{"hours_old": 24}'
```

3. Crear índices en BD (si no existen):
```sql
CREATE INDEX IF NOT EXISTS idx_sessions_status 
  ON test_sessions(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_session 
  ON session_events(session_id, created_at);
```

---

## Problemas de Base de Datos

### ❌ Problema: "database is locked"

```
sqlite3.OperationalError: database is locked
```

**Causa:** Múltiples procesos accediendo simultáneamente.

**Solución:**

```python
# Python - Usar WAL mode (Write-Ahead Logging)
import sqlite3

conn = sqlite3.connect('axiom.db')
conn.execute('PRAGMA journal_mode=WAL')
conn.commit()
```

---

### ❌ Problema: BD crece mucho (> 1GB)

**Causa:** Sesiones no se limpian automáticamente.

**Solución:**

```bash
# Limpiar sesiones de más de 7 días
curl -X POST http://localhost:8000/api/automation/cleanup/expired \
  -H "Content-Type: application/json" \
  -d '{"hours_old": 168}'

# O vaciar todo (⚠️ CUIDADO)
sqlite3 axiom.db "DELETE FROM test_sessions WHERE 1=1;"
sqlite3 axiom.db "DELETE FROM session_events WHERE 1=1;"
sqlite3 axiom.db "VACUUM;"
```

---

### ❌ Problema: Datos inconsistentes en BD

**Síntoma:** Sesión dice 5 eventos pero solo hay 2 guardados.

**Solución:**

```bash
# Verificar integridad
sqlite3 axiom.db "PRAGMA integrity_check;"

# Ver estado de sesión específica
sqlite3 axiom.db "SELECT * FROM test_sessions WHERE session_id='qa_01_1701345600';"

# Verificar eventos asociados
sqlite3 axiom.db "SELECT COUNT(*) FROM session_events WHERE session_id='qa_01_1701345600';"

# Reconstruir si es necesario
sqlite3 axiom.db "REINDEX;"
```

---

## Debugging

### 📍 Activar logs detallados

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ahora verás todos los requests HTTP
```

### 📍 Capturar respuesta completa

```python
from axiom_test_client import AxiomTestSession
import json

session = AxiomTestSession(...)
session.create()

# Guardar respuesta
response = session.client.get(f"{session.axiom_url}/api/automation/session/{session.session_id}")
print(json.dumps(response.json(), indent=2))
```

### 📍 Verificar estado en tiempo real

```bash
# Monitorear sesión activa cada 2 segundos
while true; do
  clear
  echo "Status de sesión:"
  curl -s http://localhost:8000/api/automation/session/qa_01_1701345600 | jq .
  sleep 2
done
```

### 📍 Comparar sesiones

```bash
# Exportar dos sesiones a JSON
curl http://localhost:8000/api/automation/session/session1 > s1.json
curl http://localhost:8000/api/automation/session/session2 > s2.json

# Comparar
diff s1.json s2.json
```

---

## 🆘 Última Opción: Reset Completo

Si nada funciona, hacer reset:

```powershell
# 1. Parar el servidor
# Ctrl+C en la terminal

# 2. Eliminar DB
Remove-Item axiom.db

# 3. Reiniciar
python -m uvicorn backend:app --host 0.0.0.0 --port 8000

# 4. Verificar que funciona
curl http://localhost:8000/api/automation/stats
```

---

## 📞 Checklist para Reportar Bug

Si nada funciona, reporta con:

```markdown
## Bug Report

**Descripción:**
(Qué pasó)

**Pasos para reproducir:**
1. Crear sesión con...
2. Registrar evento...
3. Se produce error

**Comportamiento esperado:**
(Qué debería pasar)

**Logs:**
```
[pegue logs/errors]
```

**Entorno:**
- Python version: `python --version`
- OS: Windows/Linux/Mac
- Axiom URL: http://...
- Sesión ID: ...

**Archivos relacionados:**
- Session ID: ...
- DB file size: ... MB
- Número de eventos: ...
```

---

## 📊 Tabla Rápida de Errores

| Código HTTP | Significado | Solución |
|-------------|------------|----------|
| 200 | OK | ✅ Éxito |
| 201 | Created | ✅ Recurso creado |
| 400 | Bad Request | Verificar JSON/parámetros |
| 404 | Not Found | Session/endpoint no existe |
| 409 | Conflict | Estado inválido (ej: iniciar COMPLETED) |
| 500 | Server Error | Error en servidor, revisar logs |
| 504 | Gateway Timeout | Servidor no responde, timeout |

