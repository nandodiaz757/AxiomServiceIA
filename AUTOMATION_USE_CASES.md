# 🎯 Casos de Uso - Automatización Axiom

## Caso 1: Validar Flujo de Login en Aplicación Android

### Escenario
Un tester QA automatiza el flujo de login (campo usuario → campo password → botón login → home screen).
Axiom debe validar que:
- Los eventos ocurren en el orden esperado
- No hay elementos UI inesperados
- No se detectan anomalías durante el flujo

### Implementación (Python + Selenium)

```python
from axiom_test_client import AxiomTestSession
from selenium import webdriver
from selenium.webdriver.common.by import By

# 1️⃣ CREAR SESIÓN EN AXIOM
session = AxiomTestSession(
    tester_id="qa_team_01",
    build_id="v2.5.0",
    app_name="com.myapp.login",
    expected_flow=["login_screen", "password_screen", "home_screen"],
    axiom_url="http://localhost:8000"
)

# 2️⃣ CREAR Y INICIAR
session.create()
session.start()

try:
    # 3️⃣ EJECUTAR TEST
    driver = webdriver.Chrome()
    driver.get("app://login")
    
    # Pantalla 1: Login
    session.record_event("login_screen", "User entered email field")
    email_field = driver.find_element(By.ID, "email_input")
    email_field.send_keys("testuser@example.com")
    
    # Pantalla 2: Password
    session.record_event("password_screen", "User entered password")
    password_field = driver.find_element(By.ID, "password_input")
    password_field.send_keys("password123")
    
    # Validar que password field existe y es visible
    session.add_validation(
        expected_element="EditText",
        found=password_field is not None,
        error_message="Password field not found"
    )
    
    # Pantalla 3: Home
    login_btn = driver.find_element(By.ID, "login_button")
    login_btn.click()
    
    session.record_event("home_screen", "Login successful, home screen loaded")
    
    # 4️⃣ FINALIZAR Y OBTENER REPORTE
    result = session.end()
    
    # 5️⃣ VALIDAR RESULTADO
    if result.success:
        print(f"✅ Test exitoso - {result.flow_completion_percentage}% flujo completado")
        print(f"📊 Eventos validados: {result.events_validated}/{result.events_received}")
    else:
        print(f"❌ Test falló: {result.errors}")
        
finally:
    driver.quit()
```

### Reporte Esperado
```json
{
  "success": true,
  "session_id": "qa_team_01_1701345600",
  "test_name": "login_flow_test",
  "duration_seconds": 12.5,
  "events_received": 3,
  "events_validated": 3,
  "flow_completion_percentage": 100,
  "validation_results": [
    {
      "event": "login_screen",
      "status": "MATCH",
      "anomaly_score": 0.02
    },
    {
      "event": "password_screen",
      "status": "MATCH",
      "anomaly_score": 0.01
    },
    {
      "event": "home_screen",
      "status": "MATCH",
      "anomaly_score": 0.03
    }
  ],
  "errors": []
}
```

---

## Caso 2: Detectar Cambios Inesperados en Flujo de Compra

### Escenario
Durante un test de compra (product list → cart → checkout → confirmation),
Axiom detecta que se agregó un elemento NEW (banner publicitario) que no estaba antes.

### Implementación (Java + Selenide)

```java
import com.codeborne.selenide.*;
import axiom.client.AxiomTestSession;
import axiom.client.TestResult;

public class PurchaseFlowTest {
    private AxiomTestSession axiom;
    
    @BeforeClass
    public void setUp() throws Exception {
        axiom = new AxiomTestSession(
            "qa_automation_02",
            "v3.1.0",
            "com.myapp.shopping",
            new String[]{"product_list", "shopping_cart", "checkout", "confirmation"},
            "http://localhost:8000"
        );
        
        axiom.create();
        axiom.start();
        
        // Configurar Selenide
        Configuration.baseUrl = "app://shop";
        Configuration.timeout = 5000;
    }
    
    @Test
    public void testPurchaseFlow() {
        try {
            // Step 1: Product List
            axiom.recordEvent("product_list", "User browsing products");
            $("[data-testid='product-1']").click();
            sleep(1000);
            
            // Step 2: Shopping Cart
            axiom.recordEvent("shopping_cart", "User added item to cart");
            $("[data-testid='add-to-cart']").click();
            $("[data-testid='cart-icon']").click();
            sleep(1000);
            
            // Step 3: Checkout
            axiom.recordEvent("checkout", "User proceeding to checkout");
            $("[data-testid='checkout-btn']").click();
            sleep(1000);
            
            // Step 4: Confirmation
            axiom.recordEvent("confirmation", "Order confirmed");
            sleep(2000);
            
        } finally {
            TestResult result = axiom.end();
            
            if (!result.isSuccess()) {
                System.out.println("⚠️ Anomalías detectadas:");
                for (String error : result.getErrors()) {
                    System.out.println("  - " + error);
                }
            }
        }
    }
    
    @AfterClass
    public void tearDown() {
        WebDriverRunner.closeWebDriver();
    }
}
```

### Reporte con Anomalía Detectada
```json
{
  "success": false,
  "session_id": "qa_automation_02_1701345700",
  "test_name": "purchase_flow_test",
  "duration_seconds": 8.3,
  "events_received": 4,
  "events_validated": 4,
  "flow_completion_percentage": 100,
  "validation_results": [
    {"event": "product_list", "status": "MATCH", "anomaly_score": 0.05},
    {"event": "shopping_cart", "status": "MATCH", "anomaly_score": 0.08},
    {
      "event": "checkout",
      "status": "UNEXPECTED",
      "anomaly_score": 0.42,
      "details": "Banner publicitario detectado (NEW elemento)"
    },
    {"event": "confirmation", "status": "MATCH", "anomaly_score": 0.03}
  ],
  "errors": [
    "NEW element detected at checkout: AdView (banner_ad_1)",
    "Possible UI regression: advertisement added without notification"
  ]
}
```

---

## Caso 3: Monitoreo Continuo de Sesión

### Escenario
Un tester ejecuta múltiples sesiones de prueba a lo largo del día.
Axiom debe:
- Mantener todas las sesiones independientes
- Permitir consultar estado en tiempo real
- Generar reportes consolidados

### Implementación

```python
# test_runner.py - Ejecutor de múltiples sesiones
from axiom_test_client import AxiomTestSession, AxiomTestContext
import requests

def run_multiple_tests():
    """Ejecuta 5 tests consecutivos y monitorea progreso"""
    
    test_scenarios = [
        {
            "name": "Login Flow",
            "flow": ["splash", "login", "dashboard"],
            "duration": 10
        },
        {
            "name": "Search Flow",
            "flow": ["dashboard", "search_screen", "results", "details"],
            "duration": 15
        },
        {
            "name": "Payment Flow",
            "flow": ["cart", "shipping", "payment", "confirmation"],
            "duration": 20
        },
        {
            "name": "Settings Flow",
            "flow": ["dashboard", "settings", "profile", "preferences"],
            "duration": 8
        },
        {
            "name": "Logout Flow",
            "flow": ["dashboard", "menu", "logout", "splash"],
            "duration": 5
        }
    ]
    
    session_ids = []
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n🧪 Ejecutando {i+1}/5: {scenario['name']}")
        
        with AxiomTestContext(
            tester_id="continuous_monitor",
            build_id="v2.0.0",
            app_name="com.example.app",
            expected_flow=scenario['flow'],
            axiom_url="http://localhost:8000"
        ) as session:
            
            session_ids.append(session.session_id)
            
            # Simular test execution
            for screen in scenario['flow']:
                session.record_event(screen, f"Transitioning to {screen}")
                session.add_validation(
                    expected_element="ViewGroup",
                    found=True,
                    error_message=None
                )
            
            result = session.get_result()
            
            if result.success:
                print(f"  ✅ {scenario['name']}: {result.flow_completion_percentage}% completado")
            else:
                print(f"  ❌ {scenario['name']}: Fallos detectados")
                for err in result.errors:
                    print(f"     - {err}")
    
    # 📊 Consultar estadísticas globales
    print("\n📊 ESTADÍSTICAS GLOBALES")
    stats = requests.get("http://localhost:8000/api/automation/stats").json()
    print(f"  Total sesiones: {stats['total_sessions']}")
    print(f"  Sesiones exitosas: {stats['successful_sessions']}")
    print(f"  Tasa de éxito: {stats['success_rate']:.1%}")
    print(f"  Promedio de eventos: {stats['avg_events_per_session']:.1f}")
    
    # 📋 Listar todas las sesiones
    print("\n📋 SESIONES CREADAS")
    sessions = requests.get(
        "http://localhost:8000/api/automation/sessions",
        params={"limit": 10}
    ).json()
    
    for sess in sessions['sessions']:
        print(f"  - {sess['session_id']}: {sess['status']} ({sess['events_count']} eventos)")

if __name__ == "__main__":
    run_multiple_tests()
```

### Salida Esperada
```
🧪 Ejecutando 1/5: Login Flow
  ✅ Login Flow: 100% completado

🧪 Ejecutando 2/5: Search Flow
  ✅ Search Flow: 100% completado

🧪 Ejecutando 3/5: Payment Flow
  ⚠️ Payment Flow: 80% completado
     - UNEXPECTED element at payment: ProgressBar (loading_indicator)

🧪 Ejecutando 4/5: Settings Flow
  ✅ Settings Flow: 100% completado

🧪 Ejecutando 5/5: Logout Flow
  ✅ Logout Flow: 100% completado

📊 ESTADÍSTICAS GLOBALES
  Total sesiones: 5
  Sesiones exitosas: 4
  Tasa de éxito: 80.0%
  Promedio de eventos: 3.6

📋 SESIONES CREADAS
  - continuous_monitor_1701345600: COMPLETED (3 eventos)
  - continuous_monitor_1701345610: COMPLETED (4 eventos)
  - continuous_monitor_1701345625: COMPLETED (4 eventos)
  - continuous_monitor_1701345633: COMPLETED (3 eventos)
  - continuous_monitor_1701345638: COMPLETED (3 eventos)
```

---

## Caso 4: Validación de Elementos Específicos

### Escenario
El tester quiere asegurar que ciertos elementos críticos existan en cada pantalla.
Por ejemplo: verificar que todos los botones de pago tienen estado "enabled".

```python
from axiom_test_client import AxiomTestSession

session = AxiomTestSession(
    tester_id="payment_qa",
    build_id="v2.5.0",
    app_name="com.payment.app",
    expected_flow=["payment_methods", "card_selection", "confirmation"],
)

session.create()
session.start()

try:
    # Pantalla 1: Métodos de Pago
    session.record_event("payment_methods", "Showing payment method options")
    
    # Validar que todos los métodos están habilitados
    session.add_validation(
        expected_element="PaymentMethodButton",
        found=True,
        error_message="Payment method button disabled or not found"
    )
    
    # Pantalla 2: Selección de Tarjeta
    session.record_event("card_selection", "User selecting card")
    
    session.add_validation(
        expected_element="CardNumberInput",
        found=True,
        error_message="Card number input missing"
    )
    
    session.add_validation(
        expected_element="CVVInput",
        found=True,
        error_message="CVV input missing"
    )
    
    # Pantalla 3: Confirmación
    session.record_event("confirmation", "Payment confirmed")
    
    session.add_validation(
        expected_element="ConfirmationMessage",
        found=True,
        error_message="Confirmation message not displayed"
    )
    
finally:
    result = session.end()
    print(f"Validaciones completadas: {result.events_validated}/{result.events_received}")
```

---

## Caso 5: Regresión Visual con Detección de Anomalías

### Escenario
Después de una actualización de UI, Axiom detecta cambios estructurales que podrían indicar
una regresión visual (elementos movidos, eliminados o mal formateados).

```python
from axiom_test_client import AxiomTestSession
import json

session = AxiomTestSession(
    tester_id="ui_regression_test",
    build_id="v3.0.0",
    app_name="com.design.system",
    expected_flow=["home", "details", "gallery", "profile"],
)

session.create()
session.start()

try:
    screens_to_validate = {
        "home": {
            "required_elements": ["header", "menu", "content_area", "footer"],
            "required_attributes": ["visibility", "position", "size"]
        },
        "details": {
            "required_elements": ["back_button", "title", "description", "cta_button"],
            "required_attributes": ["enabled", "clickable", "visible"]
        },
        "gallery": {
            "required_elements": ["grid_layout", "image_items", "scroll_indicator"],
            "required_attributes": ["scrollable", "child_count"]
        },
        "profile": {
            "required_elements": ["avatar", "name", "email", "logout_button"],
            "required_attributes": ["text", "content_description"]
        }
    }
    
    for screen_name, validations in screens_to_validate.items():
        session.record_event(screen_name, f"Validating {screen_name} screen")
        
        for element in validations["required_elements"]:
            session.add_validation(
                expected_element=element,
                found=True,
                error_message=f"Critical element '{element}' missing from {screen_name}"
            )
        
        # Validación de atributos
        session.add_validation(
            expected_element=f"{screen_name}_attributes",
            found=True,
            error_message=f"Screen '{screen_name}' missing required attributes"
        )

finally:
    result = session.end()
    
    # Mostrar anomalías detectadas
    if result.errors:
        print("🚨 POSIBLES REGRESIONES VISUALES:")
        for error in result.errors:
            print(f"  ❌ {error}")
    else:
        print("✅ No se detectaron regresiones visuales")
```

---

## Caso 6: Test de Carga (Múltiples Usuarios Simultáneos)

### Escenario
Simular múltiples testers ejecutando tests en paralelo para validar
que el servicio Axiom puede manejar carga concurrente.

```python
import asyncio
import aiohttp
from axiom_test_client import AxiomTestSession

async def concurrent_test_runner():
    """Ejecuta 10 tests de forma concurrente"""
    
    tasks = []
    
    for i in range(10):
        task = run_single_test(
            tester_id=f"load_test_user_{i}",
            build_id="v2.0.0",
            iteration=i
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Consolidar resultados
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n📊 RESULTADOS DE CARGA")
    print(f"  Total tests: {len(results)}")
    print(f"  Exitosos: {successful}")
    print(f"  Fallidos: {failed}")
    print(f"  Tasa de éxito: {successful/len(results):.1%}")

async def run_single_test(tester_id, build_id, iteration):
    """Ejecuta un test individual"""
    
    session = AxiomTestSession(
        tester_id=tester_id,
        build_id=build_id,
        app_name="com.loadtest.app",
        expected_flow=["screen_a", "screen_b", "screen_c"],
    )
    
    try:
        session.create()
        session.start()
        
        # Simular test execution
        for screen in ["screen_a", "screen_b", "screen_c"]:
            session.record_event(screen, f"Test iteration {iteration}")
            session.add_validation(
                expected_element="ViewGroup",
                found=True
            )
        
        result = session.end()
        return {"success": result.success, "tester_id": tester_id}
        
    except Exception as e:
        return {"success": False, "tester_id": tester_id, "error": str(e)}

if __name__ == "__main__":
    asyncio.run(concurrent_test_runner())
```

---

## Troubleshooting por Caso de Uso

| Caso | Problema | Solución |
|------|----------|----------|
| **Login Flow** | Sesión no inicia | Verificar que `/api/automation/session/create` responde 200 |
| **Compra** | Anomalía falso positivo en checkout | Agregar elementos conocidos a lista blanca |
| **Monitoreo Continuo** | Sesiones se quedan en RUNNING | Llamar a `/api/automation/session/{id}/end` explícitamente |
| **Validación de Elementos** | Elemento no encontrado | Verificar que `expected_element` coincide con `className` real |
| **Regresión Visual** | Demasiadas anomalías | Aumentar `threshold` o revisar que baseline sea correcto |
| **Carga Concurrente** | Timeout | Aumentar timeout de cliente o batch size |

