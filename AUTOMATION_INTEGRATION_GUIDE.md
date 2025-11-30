# 🔗 Guía de Integración: Automatización + Axiom

## Visión General

**Axiom Automation Integration** permite que tus tests automatizados (Selenium, Selenide, etc.) se ejecuten **mientras Axiom valida automáticamente los flujos de accesibilidad en paralelo**, sin necesidad de modificar tus tests existentes.

### Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────┐
│                    Automation Test Suite                         │
│  (Selenium/Selenide/JUnit/TestNG)                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ AxiomTestSession      │  ◄─── Cliente SDK
         │ .create()             │
         │ .start()              │
         │ .record_event()       │
         │ .end()                │
         └────────────┬──────────┘
                      │
                      ▼ HTTP REST API
         ┌───────────────────────────────────────────┐
         │    Axiom Backend Service                  │
         │                                           │
         │  📊 Session Manager                       │
         │  ✅ Flow Validator                        │
         │  📈 Real-time Analytics                   │
         │  🔔 Notification System                   │
         └───────────────────────────────────────────┘
                      │
                      ▼
         ┌───────────────────────┐
         │   App Under Test      │
         │  (Rappi/Web App)      │
         └───────────────────────┘
                      │
                      ▼ Accessibility Events
         ┌───────────────────────────────────────────┐
         │   Axiom Event Collector                   │
         │  (Device/Browser Integration)             │
         └───────────────────────────────────────────┘
```

---

## 🚀 Paso 1: Configurar el Servidor Axiom

### Verificar que está corriendo

```bash
# En tu terminal
python -m uvicorn backend:app --host 0.0.0.0 --port 8000

# Verificar en otra terminal
curl http://localhost:8000/api/config
```

### Endpoints disponibles para sesiones

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/automation/session/create` | POST | Crear nueva sesión |
| `/api/automation/session/{id}/start` | POST | Iniciar sesión |
| `/api/automation/session/{id}/event` | POST | Registrar evento de pantalla |
| `/api/automation/session/{id}/validation` | POST | Agregar validación |
| `/api/automation/session/{id}/end` | POST | Finalizar sesión |
| `/api/automation/session/{id}` | GET | Obtener estado de sesión |
| `/api/automation/sessions` | GET | Listar sesiones |

---

## 🐍 Integración Python + Selenium

### Instalación

```bash
pip install selenium requests

# Para Chrome
pip install chromedriver-binary
```

### Uso Básico

```python
from axiom_test_client import AxiomTestSession
from selenium import webdriver
from selenium.webdriver.common.by import By

# Crear sesión Axiom
session = AxiomTestSession(
    base_url="http://localhost:8000",
    test_name="Login Flow Test",
    tester_id="selenium_bot_01",
    build_id="1.0.0",
    app_name="com.example.app",
    expected_flow=["login_screen", "home_screen", "dashboard_screen"]
)

# Inicializar
session.create()
session.start()

# Tu código Selenium normal
driver = webdriver.Chrome()
driver.get("https://app.example.com/login")

# Registrar evento en Axiom
session.record_event(
    screen_name="login_screen",
    header_text="Iniciar Sesión"
)

# ... hacer login ...

# Registrar validación
session.add_validation(
    name="Login button clicked",
    rule={"visible": True, "clickable": True},
    passed=True
)

# Finalizar
result = session.end(success=True)
session.print_report(result)
```

### Context Manager (Recomendado)

```python
from axiom_test_client import AxiomTestSession, AxiomTestContext

# Auto-cleanup al salir
with AxiomTestContext(session) as axiom:
    # Tu test aquí
    pass
# Reporte automático al final
```

### Ejemplo Completo

Ver: `examples/selenium_example.py`

---

## ☕ Integración Java + Selenide + TestNG

### Dependencias (pom.xml)

```xml
<dependencies>
    <!-- Selenide -->
    <dependency>
        <groupId>com.codeborne</groupId>
        <artifactId>selenide</artifactId>
        <version>7.0.0</version>
    </dependency>

    <!-- TestNG -->
    <dependency>
        <groupId>org.testng</groupId>
        <artifactId>testng</artifactId>
        <version>7.8.0</version>
        <scope>test</scope>
    </dependency>

    <!-- OkHttp para HTTP requests -->
    <dependency>
        <groupId>com.squareup.okhttp3</groupId>
        <artifactId>okhttp</artifactId>
        <version>4.11.0</version>
    </dependency>

    <!-- Gson para JSON -->
    <dependency>
        <groupId>com.google.code.gson</groupId>
        <artifactId>gson</artifactId>
        <version>2.10.1</version>
    </dependency>

    <!-- SLF4J para logging -->
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-api</artifactId>
        <version>2.0.9</version>
    </dependency>
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-simple</artifactId>
        <version>2.0.9</version>
    </dependency>
</dependencies>
```

### Uso Básico

```java
import com.axiom.integration.client.AxiomTestSession;
import org.testng.annotations.*;

public class MyTest {
    
    private AxiomTestSession axiom;

    @BeforeClass
    public void setUp() {
        axiom = new AxiomTestSession(
            "http://localhost:8000",
            "My Login Test",
            "selenide_bot_01",
            "1.0.0",
            "com.example.app",
            Arrays.asList("login", "home", "dashboard"),
            Map.of("framework", "Selenide", "browser", "Chrome")
        );
        
        axiom.create();
        axiom.start();
    }

    @Test
    public void testLogin() {
        // Tu test Selenide aquí
        open("/login");
        
        axiom.recordEvent("login_screen", "Iniciar Sesión", "screen_change", null);
        
        $("#email").val("test@example.com");
        $("#password").val("pass123");
        
        axiom.addValidation("Fields filled", Map.of("email", true), true);
    }

    @AfterClass
    public void tearDown() {
        axiom.end(true);
    }
}
```

### Ejemplo Completo

Ver: `examples/RappiFlowTest.java`

---

## 🔧 Cómo Funciona la Validación

### 1️⃣ Flujo Esperado
```python
expected_flow = [
    "login_screen",
    "home_screen",
    "cart_screen",
    "checkout_screen"
]
```

### 2️⃣ Durante el Test
```
Tu test hace login          → Axiom registra: login_screen ✅
Tu test navega a home       → Axiom registra: home_screen ✅
Tu test abre carrito        → Axiom registra: cart_screen ✅
Tu test completa compra     → Axiom registra: checkout_screen ✅
```

### 3️⃣ Validaciones en Tiempo Real
- ✅ **MATCH**: Pantalla llegó en el momento correcto
- ⚠️ **UNEXPECTED**: Pantalla esperada pero en orden incorrecto
- ❌ **ANOMALY**: Pantalla no estaba en el flujo esperado

### 4️⃣ Reporte Final
```
═══════════════════════════════════════════════════════════════
📋 REPORTE DE AUTOMATIZACIÓN - Login and Cart Flow
═══════════════════════════════════════════════════════════════
🔑 Session ID: A1B2C3D4
⏱️  Duración: 45.23 segundos
📊 Eventos: 8 recibidos, 8 validados
📈 Flujo: 100.0% completado
✅ Resultado: EXITOSO

📍 Flujo esperado (4 pantallas):
  1. login_screen
  2. home_screen
  3. cart_screen
  4. checkout_screen

📍 Flujo realizado (4 pantallas):
  1. login_screen
  2. home_screen
  3. cart_screen
  4. checkout_screen

❌ Errores (0):
═══════════════════════════════════════════════════════════════
```

---

## 📊 API REST Detallada

### Crear Sesión

```bash
POST /api/automation/session/create

{
  "test_name": "Login Flow Test",
  "tester_id": "selenium_bot_01",
  "build_id": "1.0.0",
  "app_name": "com.example.app",
  "expected_flow": ["login_screen", "home_screen"],
  "metadata": {
    "browser": "Chrome",
    "environment": "staging",
    "device_type": "desktop"
  }
}

# Response
{
  "session_id": "A1B2C3D4",
  "status": "created",
  "timestamp": "2024-11-30T10:30:00Z"
}
```

### Registrar Evento

```bash
POST /api/automation/session/A1B2C3D4/event

{
  "screen_name": "login_screen",
  "header_text": "Iniciar Sesión",
  "event_type": "screen_change",
  "additional_data": {
    "url": "https://app.example.com/login",
    "user_logged_in": false
  }
}

# Response
{
  "success": true,
  "validation_result": "match",
  "message": "✅ Evento coincide: login_screen (posición 1/4)"
}
```

### Agregar Validación

```bash
POST /api/automation/session/A1B2C3D4/validation

{
  "validation_name": "Login fields visible",
  "rule": {
    "has_email_field": true,
    "has_password_field": true
  },
  "passed": true
}

# Response
{
  "success": true,
  "message": "✓ Validación registrada"
}
```

### Finalizar Sesión

```bash
POST /api/automation/session/A1B2C3D4/end

{
  "success": true,
  "final_status": "completed"
}

# Response
{
  "session_id": "A1B2C3D4",
  "test_name": "Login Flow Test",
  "status": "completed",
  "duration_seconds": 45.23,
  "events_received": 8,
  "events_validated": 8,
  "flow_completion_percentage": 100,
  "expected_flow": ["login_screen", "home_screen"],
  "actual_flow": ["login_screen", "home_screen"],
  "validation_errors": [],
  "success": true
}
```

---

## 🎯 Casos de Uso

### Caso 1: Validar Login + Flujo de Compra

```python
expected_flow = [
    "login_screen",
    "home_screen",
    "search_results",
    "product_detail",
    "cart_screen",
    "checkout_screen",
    "payment_screen",
    "order_confirmation"
]

session = AxiomTestSession(..., expected_flow=expected_flow)
session.create()
session.start()

# Ejecutar test automatizado
# Axiom valida automáticamente que cada pantalla llegue en orden
result = session.end()
```

### Caso 2: Detectar Anomalías

```python
# Si tu test hace algo inesperado
session.record_event("cart_screen")  # ✅ Esperado
session.record_event("home_screen")  # ⚠️ UNEXPECTED - iba a checkout

# Axiom detecta y reporta la desviación
```

### Caso 3: Validaciones de Accesibilidad

```python
# Además del flujo, valida elementos
session.add_validation(
    name="Accessibility - Button contrast",
    rule={"contrast_ratio": 4.5},
    passed=True
)

session.add_validation(
    name="Accessibility - Form labels",
    rule={"has_labels": True, "labels_associated": True},
    passed=True
)
```

### Caso 4: Monitoreo en Paralelo

```python
# Tu test ejecuta como siempre
for i in range(100):
    selenium_test()  # Tu lógica normal
    # Axiom monitorea en paralelo sin interferir
```

---

## 🐛 Troubleshooting

### El servidor Axiom no responde

```bash
# Verificar que está corriendo
curl -v http://localhost:8000/api/config

# Si no funciona, reiniciar
python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### Sesión no se crea

```python
# Verificar configuración
print(f"Base URL: {session.base_url}")
print(f"Test Name: {session.test_name}")

# Ver logs del servidor para más detalles
```

### Eventos no se registran

```python
# Asegúrate de haber llamado start()
session.create()
session.start()  # ← Esto es crítico
session.record_event(...)
```

### Validación siempre falla

```python
# Verifica que los screen_names coincidan exactamente
expected_flow = ["login_screen"]  # lowercase
session.record_event("login_screen")  # ← mismo caso

# ❌ Esto NO funciona:
session.record_event("Login_Screen")  # Diferente case
```

---

## 📈 Monitoreo y Reportes

### Ver todas las sesiones

```bash
GET /api/automation/sessions
```

### Obtener reporte de sesión

```bash
GET /api/automation/session/A1B2C3D4
```

### Filtrar por estado

```bash
GET /api/automation/sessions?status=completed
GET /api/automation/sessions?status=failed
GET /api/automation/sessions?tester_id=selenium_bot_01
```

---

## 🔒 Seguridad

### Credenciales

- **No** guardes URLs de base en el código
- Usa variables de entorno:

```python
import os

AXIOM_URL = os.getenv("AXIOM_BASE_URL", "http://localhost:8000")
session = AxiomTestSession(base_url=AXIOM_URL, ...)
```

### Timeout

Por defecto 30 segundos. Ajustar si necesitas:

```python
session = AxiomTestSession(
    ...,
    timeout=60  # Más tiempo para tests lentos
)
```

---

## 📚 Recursos Adicionales

- **Ejemplos**: `examples/` carpeta
- **Session Manager**: `session_manager.py`
- **Cliente Python**: `axiom_test_client.py`
- **Cliente Java**: `examples/AxiomTestSession.java`

---

## ✅ Checklist de Integración

- [ ] Axiom backend corriendo en puerto 8000
- [ ] Cliente SDK instalado (Python o Java)
- [ ] Test automatizado (Selenium/Selenide) listo
- [ ] `expected_flow` definido correctamente
- [ ] Sessión creada antes de iniciar test
- [ ] `record_event()` llamado en cada navegación
- [ ] Validaciones adicionales agregadas donde sea necesario
- [ ] Test finalizado con `.end()`
- [ ] Reporte revisado en logs o console

---

**¡Listo!** Tu test automatizado ahora valida flujos en paralelo con Axiom. 🚀
