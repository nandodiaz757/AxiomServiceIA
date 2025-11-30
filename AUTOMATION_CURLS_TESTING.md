# 🧪 cURLs de Prueba - Endpoint Automation API

## 📌 Configuración Inicial

```bash
# Variables globales (reemplaza con tus valores)
AXIOM_URL="http://localhost:8000"
TESTER_ID="qa_tester_01"
BUILD_ID="v2.5.0"
APP_NAME="com.example.app"
SESSION_ID=""  # Se genera después de crear sesión
```

---

## 1️⃣ CREAR SESIÓN

### Request
```bash
curl -X POST "${AXIOM_URL}/api/automation/session/create" \
  -H "Content-Type: application/json" \
  -d '{
    "tester_id": "'"${TESTER_ID}"'",
    "build_id": "'"${BUILD_ID}"'",
    "app_name": "'"${APP_NAME}"'",
    "expected_flow": ["login_screen", "home_screen", "profile_screen"],
    "test_name": "Complete User Flow Test",
    "description": "Testing login -> home -> profile navigation"
  }'
```

### Response Exitosa (200)
```json
{
  "session_id": "qa_tester_01_1701345600",
  "status": "CREATED",
  "tester_id": "qa_tester_01",
  "build_id": "v2.5.0",
  "app_name": "com.example.app",
  "expected_flow": ["login_screen", "home_screen", "profile_screen"],
  "created_at": "2025-11-30T10:30:00Z",
  "message": "Session created successfully"
}
```

### Response con Error (400)
```json
{
  "detail": "Missing required field: expected_flow"
}
```

---

## 2️⃣ INICIAR SESIÓN

### Request
```bash
SESSION_ID="qa_tester_01_1701345600"

curl -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/start" \
  -H "Content-Type: application/json" \
  -d '{
    "additional_info": {
      "device_model": "Pixel 6",
      "os_version": "12",
      "app_version": "2.5.0"
    }
  }'
```

### Response Exitosa (200)
```json
{
  "session_id": "qa_tester_01_1701345600",
  "status": "RUNNING",
  "started_at": "2025-11-30T10:30:15Z",
  "message": "Session started successfully",
  "device_info": {
    "device_model": "Pixel 6",
    "os_version": "12",
    "app_version": "2.5.0"
  }
}
```

### Response con Error - Sesión no existe (404)
```json
{
  "detail": "Session not found: qa_tester_01_1701345600"
}
```

### Response con Error - Estado inválido (409)
```json
{
  "detail": "Cannot start session in state: COMPLETED"
}
```

---

## 3️⃣ REGISTRAR EVENTO

### Request (Evento que coincide con flujo esperado)
```bash
SESSION_ID="qa_tester_01_1701345600"

curl -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/event" \
  -H "Content-Type: application/json" \
  -d '{
    "event_name": "login_screen",
    "event_type": "screen_change",
    "timestamp": 1701345630.5,
    "additional_data": {
      "elements_visible": 5,
      "interactive_elements": 3,
      "screen_hash": "abc123def456"
    }
  }'
```

### Response Exitosa (200) - MATCH
```json
{
  "event_id": "evt_001",
  "session_id": "qa_tester_01_1701345600",
  "event_name": "login_screen",
  "validation_result": "MATCH",
  "anomaly_score": 0.02,
  "position_in_flow": 0,
  "message": "Event matched expected flow",
  "details": {
    "expected": "login_screen",
    "received": "login_screen",
    "confidence": 0.98
  },
  "timestamp": "2025-11-30T10:30:30Z"
}
```

### Request (Evento que NO esperábamos)
```bash
curl -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/event" \
  -H "Content-Type: application/json" \
  -d '{
    "event_name": "unexpected_popup",
    "event_type": "screen_change",
    "timestamp": 1701345645.2,
    "additional_data": {
      "popup_title": "Advertisement",
      "is_closeable": false
    }
  }'
```

### Response (200) - UNEXPECTED
```json
{
  "event_id": "evt_002",
  "session_id": "qa_tester_01_1701345600",
  "event_name": "unexpected_popup",
  "validation_result": "UNEXPECTED",
  "anomaly_score": 0.65,
  "message": "Event not in expected flow",
  "details": {
    "expected": "home_screen",
    "received": "unexpected_popup",
    "suggestion": "Check if advertisement is new or test UI has changed"
  },
  "timestamp": "2025-11-30T10:30:45Z"
}
```

### Request (Falta evento esperado)
```bash
# Registrar evento pero saltando uno del flujo esperado
curl -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/event" \
  -H "Content-Type: application/json" \
  -d '{
    "event_name": "profile_screen",
    "event_type": "screen_change",
    "timestamp": 1701345660.1
  }'
```

### Response (200) - MISSING
```json
{
  "event_id": "evt_003",
  "session_id": "qa_tester_01_1701345600",
  "event_name": "profile_screen",
  "validation_result": "MISSING",
  "anomaly_score": 0.48,
  "message": "Expected screen was skipped in flow",
  "details": {
    "expected_sequence": ["login_screen", "home_screen", "profile_screen"],
    "received_sequence": ["login_screen", "profile_screen"],
    "missing_screens": ["home_screen"]
  },
  "timestamp": "2025-11-30T10:31:00Z"
}
```

---

## 4️⃣ AGREGAR VALIDACIÓN

### Request (Validación exitosa)
```bash
SESSION_ID="qa_tester_01_1701345600"

curl -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/validation" \
  -H "Content-Type: application/json" \
  -d '{
    "validation_name": "Login button enabled",
    "expected_result": true,
    "actual_result": true,
    "element_type": "Button",
    "element_id": "login_btn",
    "assertion_type": "element_exists"
  }'
```

### Response (201)
```json
{
  "validation_id": "val_001",
  "session_id": "qa_tester_01_1701345600",
  "validation_name": "Login button enabled",
  "status": "PASSED",
  "expected": true,
  "actual": true,
  "created_at": "2025-11-30T10:30:30Z"
}
```

### Request (Validación fallida)
```bash
curl -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/validation" \
  -H "Content-Type: application/json" \
  -d '{
    "validation_name": "Payment button visible",
    "expected_result": true,
    "actual_result": false,
    "element_type": "Button",
    "element_id": "payment_btn",
    "assertion_type": "element_visible",
    "error_message": "Payment button not visible on checkout screen"
  }'
```

### Response (201) - FAILED
```json
{
  "validation_id": "val_002",
  "session_id": "qa_tester_01_1701345600",
  "validation_name": "Payment button visible",
  "status": "FAILED",
  "expected": true,
  "actual": false,
  "error_message": "Payment button not visible on checkout screen",
  "created_at": "2025-11-30T10:30:45Z"
}
```

---

## 5️⃣ CONSULTAR ESTADO DE SESIÓN

### Request
```bash
SESSION_ID="qa_tester_01_1701345600"

curl -X GET "${AXIOM_URL}/api/automation/session/${SESSION_ID}" \
  -H "Content-Type: application/json"
```

### Response (200)
```json
{
  "session_id": "qa_tester_01_1701345600",
  "tester_id": "qa_tester_01",
  "build_id": "v2.5.0",
  "app_name": "com.example.app",
  "status": "RUNNING",
  "created_at": "2025-11-30T10:30:00Z",
  "started_at": "2025-11-30T10:30:15Z",
  "expected_flow": ["login_screen", "home_screen", "profile_screen"],
  "actual_flow": ["login_screen", "home_screen"],
  "events_received": 2,
  "events_validated": 2,
  "validations_passed": 1,
  "validations_failed": 0,
  "anomaly_score": 0.15
}
```

---

## 6️⃣ FINALIZAR SESIÓN

### Request
```bash
SESSION_ID="qa_tester_01_1701345600"

curl -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/end" \
  -H "Content-Type: application/json" \
  -d '{
    "test_result": "PASSED",
    "notes": "All validations passed successfully"
  }'
```

### Response (200)
```json
{
  "session_id": "qa_tester_01_1701345600",
  "status": "COMPLETED",
  "test_result": "PASSED",
  "summary": {
    "duration_seconds": 65,
    "events_received": 3,
    "events_validated": 3,
    "flow_completion_percentage": 100,
    "validations_summary": {
      "passed": 5,
      "failed": 0,
      "total": 5
    },
    "anomaly_score": 0.12
  },
  "report": {
    "test_name": "Complete User Flow Test",
    "tester_id": "qa_tester_01",
    "build_id": "v2.5.0",
    "started_at": "2025-11-30T10:30:15Z",
    "ended_at": "2025-11-30T10:31:20Z",
    "events": [
      {
        "event_name": "login_screen",
        "status": "MATCH",
        "anomaly_score": 0.02
      },
      {
        "event_name": "home_screen",
        "status": "MATCH",
        "anomaly_score": 0.10
      },
      {
        "event_name": "profile_screen",
        "status": "MATCH",
        "anomaly_score": 0.12
      }
    ],
    "validations": [
      {
        "validation_name": "Login button enabled",
        "status": "PASSED"
      },
      {
        "validation_name": "Home page loaded",
        "status": "PASSED"
      }
    ],
    "errors": []
  },
  "completed_at": "2025-11-30T10:31:20Z"
}
```

---

## 7️⃣ LISTAR SESIONES

### Request (Sin filtros)
```bash
curl -X GET "${AXIOM_URL}/api/automation/sessions?limit=10" \
  -H "Content-Type: application/json"
```

### Response (200)
```json
{
  "total": 3,
  "sessions": [
    {
      "session_id": "qa_tester_01_1701345600",
      "tester_id": "qa_tester_01",
      "build_id": "v2.5.0",
      "status": "COMPLETED",
      "events_count": 3,
      "validations_count": 5,
      "created_at": "2025-11-30T10:30:00Z"
    },
    {
      "session_id": "qa_tester_02_1701345700",
      "tester_id": "qa_tester_02",
      "build_id": "v2.5.0",
      "status": "RUNNING",
      "events_count": 2,
      "validations_count": 3,
      "created_at": "2025-11-30T10:35:00Z"
    }
  ]
}
```

### Request (Con filtros)
```bash
# Filtrar por estado COMPLETED y tester específico
curl -X GET "${AXIOM_URL}/api/automation/sessions?status=COMPLETED&tester_id=qa_tester_01&limit=5" \
  -H "Content-Type: application/json"
```

---

## 8️⃣ OBTENER ESTADÍSTICAS GLOBALES

### Request
```bash
curl -X GET "${AXIOM_URL}/api/automation/stats" \
  -H "Content-Type: application/json"
```

### Response (200)
```json
{
  "total_sessions": 42,
  "successful_sessions": 38,
  "failed_sessions": 4,
  "running_sessions": 3,
  "success_rate": 0.90476,
  "total_events": 156,
  "avg_events_per_session": 3.71,
  "total_validations": 287,
  "avg_validations_per_session": 6.83,
  "validation_success_rate": 0.955,
  "avg_session_duration_seconds": 48.5,
  "testers_count": 7,
  "builds_count": 5,
  "apps_count": 3,
  "last_updated": "2025-11-30T10:40:00Z"
}
```

---

## 9️⃣ LIMPIAR SESIONES EXPIRADAS

### Request (Limpiar sesiones de más de 24 horas)
```bash
curl -X POST "${AXIOM_URL}/api/automation/cleanup/expired" \
  -H "Content-Type: application/json" \
  -d '{
    "hours_old": 24
  }'
```

### Response (200)
```json
{
  "deleted_sessions": 5,
  "deleted_events": 18,
  "deleted_validations": 32,
  "message": "Cleanup completed successfully"
}
```

---

## 🔟 SCRIPT COMPLETO DE PRUEBA

### `test_automation_api.sh` - Ejecutar flujo completo

```bash
#!/bin/bash

set -e

AXIOM_URL="http://localhost:8000"
TESTER_ID="qa_automation_test_$(date +%s)"
BUILD_ID="v2.5.0"
APP_NAME="com.test.app"

echo "🚀 INICIANDO PRUEBAS DE AUTOMATION API"
echo "================================================"

# 1. Crear sesión
echo -e "\n1️⃣ CREAR SESIÓN"
SESSION_RESPONSE=$(curl -s -X POST "${AXIOM_URL}/api/automation/session/create" \
  -H "Content-Type: application/json" \
  -d '{
    "tester_id": "'"${TESTER_ID}"'",
    "build_id": "'"${BUILD_ID}"'",
    "app_name": "'"${APP_NAME}"'",
    "expected_flow": ["screen_a", "screen_b", "screen_c"],
    "test_name": "Full Integration Test"
  }')

SESSION_ID=$(echo $SESSION_RESPONSE | grep -o '"session_id":"[^"]*' | cut -d'"' -f4)
echo "✅ Sesión creada: $SESSION_ID"

# 2. Iniciar sesión
echo -e "\n2️⃣ INICIAR SESIÓN"
curl -s -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/start" \
  -H "Content-Type: application/json" \
  -d '{"additional_info": {"device": "Pixel 6"}}' | jq .
echo "✅ Sesión iniciada"

# 3. Registrar eventos
echo -e "\n3️⃣ REGISTRAR EVENTOS"
for SCREEN in "screen_a" "screen_b" "screen_c"; do
  echo "  📍 Registrando: $SCREEN"
  curl -s -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/event" \
    -H "Content-Type: application/json" \
    -d '{
      "event_name": "'"${SCREEN}"'",
      "event_type": "screen_change",
      "timestamp": '$(date +%s.%3N)'
    }' | jq '.validation_result'
done
echo "✅ Eventos registrados"

# 4. Agregar validaciones
echo -e "\n4️⃣ AGREGAR VALIDACIONES"
curl -s -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/validation" \
  -H "Content-Type: application/json" \
  -d '{
    "validation_name": "Test validation",
    "expected_result": true,
    "actual_result": true,
    "assertion_type": "element_exists"
  }' | jq '.status'
echo "✅ Validación agregada"

# 5. Consultar estado
echo -e "\n5️⃣ CONSULTAR ESTADO DE SESIÓN"
curl -s -X GET "${AXIOM_URL}/api/automation/session/${SESSION_ID}" \
  -H "Content-Type: application/json" | jq '{status, events_received, events_validated}'
echo "✅ Estado consultado"

# 6. Finalizar sesión
echo -e "\n6️⃣ FINALIZAR SESIÓN"
FINAL_RESPONSE=$(curl -s -X POST "${AXIOM_URL}/api/automation/session/${SESSION_ID}/end" \
  -H "Content-Type: application/json" \
  -d '{"test_result": "PASSED"}')

echo $FINAL_RESPONSE | jq '.summary'
echo "✅ Sesión finalizada"

# 7. Obtener estadísticas
echo -e "\n7️⃣ OBTENER ESTADÍSTICAS"
curl -s -X GET "${AXIOM_URL}/api/automation/stats" \
  -H "Content-Type: application/json" | jq '{total_sessions, success_rate, total_events}'

echo -e "\n================================================"
echo "✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE"
echo "================================================"
```

### Ejecución
```bash
bash test_automation_api.sh
```

---

## 🔧 TROUBLESHOOTING cURLs

### Verificar que el servicio está activo
```bash
curl -I http://localhost:8000/docs
# Debe retornar 200 OK
```

### Ver detalles de error
```bash
curl -v http://localhost:8000/api/automation/session/invalid_id
# Muestra headers y respuesta completa
```

### Probar con autenticación (si la implementas)
```bash
curl -X GET "http://localhost:8000/api/automation/stats" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json"
```

### Exportar respuesta a archivo
```bash
curl -X GET "http://localhost:8000/api/automation/sessions" \
  -H "Content-Type: application/json" > sessions_response.json

# Ver contenido formateado
cat sessions_response.json | jq .
```

---

## 📊 Casos de Prueba Recomendados

1. ✅ **Happy Path**: Crear → Iniciar → 3 eventos → Validaciones → Finalizar
2. ✅ **Error: Sesión no existe**: GET con session_id inválido → 404
3. ✅ **Error: Estado inválido**: Iniciar sesión ya COMPLETED → 409
4. ✅ **Anomalía detectada**: Evento que no está en expected_flow → UNEXPECTED
5. ✅ **Validación fallida**: expected_result != actual_result
6. ✅ **Carga concurrente**: 5 sesiones simultáneas
7. ✅ **Limpieza**: Verificar que sesiones viejas se eliminan

