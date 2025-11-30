# ✅ Axiom Automation Integration - Implementación Completada

## 📋 Resumen Ejecutivo

Has completado la integración entre **Axiom** y **sistemas de automatización de pruebas** (Selenium, Selenide, JUnit, TestNG, etc.).

Ahora tus automation testers pueden:

1. **Crear una sesión de prueba** en Axiom
2. **Ejecutar sus tests normalmente** con Selenium/Selenide
3. **Registrar eventos de navegación** conforme avanzan
4. **Axiom valida automáticamente** que el flujo sea correcto
5. **Obtener un reporte completo** con resultados de validación

### Ejemplo Rápido (Python + Selenium)

```python
from axiom_test_client import AxiomTestSession
from selenium import webdriver

# Crear sesión
session = AxiomTestSession(
    test_name="Login Flow",
    expected_flow=["login_screen", "home_screen", "cart_screen"]
)
session.create()
session.start()

# Ejecutar test normalmente
driver = webdriver.Chrome()
driver.get("https://app.example.com")

# Notificar a Axiom
session.record_event("login_screen", "Iniciar Sesión")
# ... hacer login ...

session.record_event("home_screen", "Inicio")
session.record_event("cart_screen", "Tu Carrito")

# Obtener reporte
result = session.end()
print(f"✅ Test {'EXITOSO' if result.success else 'FALLÓ'}")
```

---

## 🗂️ Archivos Creados

### Core System
| Archivo | Propósito | Líneas |
|---------|-----------|--------|
| `session_manager.py` | Gestor centralizado de sesiones | 650 |
| `automation_endpoints.py` | Endpoints FastAPI para testers | 450 |

### SDKs para Testers
| Archivo | Lenguaje | Propósito | Líneas |
|---------|----------|-----------|--------|
| `axiom_test_client.py` | Python | Cliente para usar desde Selenium | 350 |
| `examples/AxiomTestSession.java` | Java | Cliente para usar desde Selenide | 400 |
| `examples/TestResult.java` | Java | Clase de resultados | 80 |

### Ejemplos de Integración
| Archivo | Framework | Propósito | Líneas |
|---------|-----------|-----------|--------|
| `examples/selenium_example.py` | Selenium Python | Ejemplo completo | 180 |
| `examples/RappiFlowTest.java` | Selenide + TestNG | Ejemplo completo | 150 |

### Documentación
| Archivo | Propósito | Secciones |
|---------|-----------|-----------|
| `AUTOMATION_INTEGRATION_GUIDE.md` | Guía de integración | Uso, ejemplos, troubleshooting |
| `ARCHITECTURE.md` | Diseño del sistema | Componentes, flujos, BD |

**Total: 10 archivos nuevos, ~2,500 líneas de código**

---

## 🔄 Flujo General

```
1. PREPARACIÓN
   ├─ Tester inicializa AxiomTestSession
   ├─ Especifica expected_flow (pantallas en orden)
   └─ Session.create() → Session ID

2. INICIO
   └─ Session.start() → Status = RUNNING

3. DURANTE EL TEST (en paralelo)
   ├─ Tester ejecuta Selenium/Selenide normalmente
   ├─ Para cada cambio de pantalla:
   │  └─ Session.record_event(screen_name)
   └─ Axiom valida automáticamente

4. VALIDACIONES ADICIONALES
   └─ Session.add_validation(name, rule, passed)

5. FINALIZACIÓN
   ├─ Session.end(success)
   └─ Obtener reporte con métricas

6. ANÁLISIS
   ├─ Flujo completado %
   ├─ Eventos validados
   ├─ Errores detectados
   └─ Timeline de ejecución
```

---

## 📊 Validación en Tiempo Real

### Algoritmo
```
Para cada evento registrado:
1. ¿Coincide con la pantalla esperada en la posición actual?
   → ✅ MATCH (avanzar a siguiente pantalla)
2. ¿Está en el flujo esperado pero en orden incorrecto?
   → ⚠️ UNEXPECTED (error registrado)
3. ¿No está en el flujo esperado?
   → ❌ ANOMALY (error registrado)
```

### Ejemplo de Validación

```
Esperado:  ["login", "home", "cart", "checkout"]

Test 1:
  login     ✅ MATCH (1/4)
  home      ✅ MATCH (2/4)
  cart      ✅ MATCH (3/4)
  checkout  ✅ MATCH (4/4)
  → ✅ 100% completado

Test 2:
  login     ✅ MATCH (1/4)
  home      ✅ MATCH (2/4)
  settings  ❌ ANOMALY (no en flujo)
  cart      ✅ MATCH (3/4)
  checkout  ✅ MATCH (4/4)
  → ⚠️ 100% pero 1 anomalía detectada
```

---

## 🎯 Endpoints Disponibles

### Crear Sesión
```
POST /api/automation/session/create
Body: {
  test_name, tester_id, build_id, app_name, 
  expected_flow, metadata
}
Response: { session_id, status, ... }
```

### Registrar Evento
```
POST /api/automation/session/{session_id}/event
Body: { screen_name, header_text, event_type }
Response: { validation_result, message }
```

### Agregar Validación
```
POST /api/automation/session/{session_id}/validation
Body: { validation_name, rule, passed }
Response: { success, message }
```

### Finalizar Sesión
```
POST /api/automation/session/{session_id}/end
Body: { success, final_status }
Response: { reporte completo }
```

### Consultar
```
GET /api/automation/session/{session_id}      → Estado actual
GET /api/automation/sessions                   → Listar sesiones
GET /api/automation/stats                      → Estadísticas
```

---

## 🚀 Cómo Usar

### Opción 1: Python + Selenium

```bash
pip install selenium requests
```

```python
from axiom_test_client import AxiomTestSession
from selenium import webdriver

session = AxiomTestSession(
    base_url="http://localhost:8000",
    test_name="My Test",
    expected_flow=["screen1", "screen2"]
)
session.create()
session.start()

driver = webdriver.Chrome()
# ... tu código ...
session.record_event("screen1")
# ... más código ...

result = session.end()
session.print_report(result)
```

### Opción 2: Java + Selenide

```bash
# En pom.xml
<dependency>
    <groupId>com.codeborne</groupId>
    <artifactId>selenide</artifactId>
    <version>7.0.0</version>
</dependency>
```

```java
import com.axiom.integration.client.AxiomTestSession;

public class MyTest {
    @BeforeClass
    public void setUp() {
        axiom = new AxiomTestSession(
            "http://localhost:8000",
            "My Test",
            "bot_01",
            "1.0.0",
            "com.app",
            Arrays.asList("screen1", "screen2")
        );
        axiom.create();
        axiom.start();
    }

    @Test
    public void test() {
        // Tu código Selenide
        axiom.recordEvent("screen1", "Title");
    }
}
```

---

## 💾 Base de Datos

### Tablas Creadas Automáticamente

1. **test_sessions** - Sesiones de prueba
2. **session_events** - Eventos registrados
3. **session_validations** - Validaciones adicionales
4. **session_reports** - Reportes finales

Todas en: `axiom_test.db` (SQLite)

---

## 📈 Reporte de Sesión

Cada sesión genera un reporte con:

```json
{
  "session_id": "A1B2C3D4",
  "test_name": "Login Flow",
  "status": "completed",
  "success": true,
  "duration_seconds": 45.23,
  "events_received": 8,
  "events_validated": 8,
  "flow_completion_percentage": 100,
  "expected_flow": ["login", "home", "cart"],
  "actual_flow": ["login", "home", "cart"],
  "validation_errors": [],
  "errors_count": 0
}
```

---

## 🔧 Integración en Backend

Para agregar a tu `backend.py`:

```python
# 1. Importar
from automation_endpoints import router as automation_router
from session_manager import init_session_manager

# 2. En lifespan()
@asynccontextmanager
async def lifespan(app):
    # STARTUP
    session_mgr = init_session_manager()
    yield
    # SHUTDOWN

# 3. Registrar router
app.include_router(automation_router)
```

---

## 📚 Documentación Completa

- **AUTOMATION_INTEGRATION_GUIDE.md** - Cómo integrar (paso a paso)
- **ARCHITECTURE.md** - Diseño técnico (componentes, flujos)
- **examples/** - Ejemplos funcionales (Selenium, Selenide)

---

## ✅ Checklist de Implementación

- ✅ SessionManager creado
- ✅ Endpoints FastAPI implementados
- ✅ Cliente Python (axiom_test_client.py)
- ✅ Cliente Java (AxiomTestSession.java)
- ✅ Ejemplo Selenium
- ✅ Ejemplo Selenide + TestNG
- ✅ Validación en tiempo real
- ✅ Persistencia en BD
- ✅ Reportes automáticos
- ✅ Documentación completa

---

## 🎓 Próximos Pasos

### Fase 1 (Ahora)
1. Revisar `AUTOMATION_INTEGRATION_GUIDE.md`
2. Probar con el ejemplo `selenium_example.py`
3. Integrar en tu test suite

### Fase 2 (Futuro)
- [ ] WebSocket para eventos en tiempo real
- [ ] Dashboard web con live metrics
- [ ] Notificaciones Slack/Teams
- [ ] Export HTML/PDF
- [ ] Integración CI/CD (GitHub Actions, GitLab CI)

### Fase 3 (Optimización)
- [ ] Caché de sessiones
- [ ] Análisis predictivo con ML
- [ ] Comparación entre ejecuciones

---

## 🆘 Soporte Rápido

### ¿El servidor no responde?
```bash
python -m uvicorn backend:app --host 0.0.0.0 --port 8000
```

### ¿Session ID no se genera?
- Verificar que `session.create()` se llamó primero

### ¿Los eventos no se registran?
- Verificar que `session.start()` se llamó antes de `record_event()`
- Verificar que session_id es correcto

### ¿El flujo valida incorrecto?
- Verificar que `expected_flow` tiene los screen names exactos
- Los screen names son case-sensitive

---

## 📞 Contacto

Para preguntas o bugs, revisa:
- Logs del backend: `python -m uvicorn backend:app --log-level debug`
- BD: `sqlite3 axiom_test.db`

---

**¡Listo para usar! Tus tests automatizados ahora validan flujos en paralelo con Axiom.** 🚀

Documentación completa en `AUTOMATION_INTEGRATION_GUIDE.md`
Arquitectura técnica en `ARCHITECTURE.md`
