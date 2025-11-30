# 🎯 Sistema de Configuración Server-side - AxiomServiceIA

> **Sistema centralizado para gestionar webhooks de Slack, Teams y Jira sin cambiar código**

---

## 📊 Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                      │
│                      (backend.py)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ConfigManager (config_manager.py)                  │   │
│  │  ├── Carga config.yaml                              │   │
│  │  ├── Resuelve variables de entorno ${VAR}           │   │
│  │  ├── Proporciona acceso thread-safe                 │   │
│  │  └── Enmasca valores sensibles                      │   │
│  └─────────────────────────────────────────────────────┘   │
│           ▲                    ▲                  ▲          │
│           │                    │                  │          │
│  ┌────────┴─────┐  ┌──────────┴────────┐  ┌──────┴──────┐ │
│  │send_slack_   │  │send_teams_alert()  │  │send_jira_  │ │
│  │alert()       │  │+ Retry automático  │  │issue()     │ │
│  │+ Retry x2    │  │+ Timeout configurable    │+ Retry x2  │ │
│  └──────┬────────┘  └──────────┬────────┘  └──────┬──────┘ │
│         │                      │                   │        │
└─────────┼──────────────────────┼───────────────────┼────────┘
          │                      │                   │
          ▼                      ▼                   ▼
    ┌──────────┐            ┌──────────┐        ┌──────────┐
    │  Slack   │            │  Teams   │        │   Jira   │
    └──────────┘            └──────────┘        └──────────┘
```

---

## 📁 Archivos Creados

### Core Components

```
config_manager.py         ← Gestor centralizado
config.yaml              ← Archivo de configuración (crear copiando example)
config.yaml.example      ← Plantilla con comentarios
```

### Scripts Utilities

```
setup.py                 ← Setup interactivo (fácil configuración)
test_config.py          ← Tests automatizados
```

### Configuración

```
.env.example            ← Template de variables de entorno
```

### Documentación

```
CONFIG_SYSTEM.md        ← Docs completas del sistema
IMPLEMENTATION_SUMMARY.md ← Resumen de implementación
QUICKSTART_CONFIG.md    ← Esta guía rápida
```

---

## ⚡ Quick Start (5 minutos)

### 1️⃣ Setup Automático

```bash
# Ejecutar script interactivo
python setup.py

# O manual:
cp config.yaml.example config.yaml
cp .env.example .env
```

### 2️⃣ Configurar Webhooks

**Opción A: Variables de entorno**
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export TEAMS_WEBHOOK_URL="https://outlook.webhook.office.com/webhookb2/YOUR/WEBHOOK/URL"
```

**Opción B: En config.yaml**
```yaml
notifications:
  slack:
    webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### 3️⃣ Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4️⃣ Verificar Configuración

```bash
# Iniciar servidor
python -m uvicorn backend:app --reload

# En otra terminal, verificar
curl http://localhost:8000/api/config/health

# Probar notificación
curl -X POST http://localhost:8000/api/config/test-slack
```

### 5️⃣ Ejecutar Tests

```bash
python test_config.py
```

---

## 🔌 Endpoints API

### Obtener Configuración

```bash
# Configuración completa (sensibles enmascarados)
curl http://localhost:8000/api/config

# Solo notificaciones
curl http://localhost:8000/api/config/notifications

# Solo CI/CD
curl http://localhost:8000/api/config/ci

# Solo ML
curl http://localhost:8000/api/config/ml
```

### Probar Notificaciones

```bash
# Test Slack
curl -X POST http://localhost:8000/api/config/test-slack

# Test Teams
curl -X POST http://localhost:8000/api/config/test-teams
```

### Health Check

```bash
# Verificar que todo está ok
curl http://localhost:8000/api/config/health
```

### Recargar Configuración

```bash
# Sin reiniciar servidor
curl -X POST http://localhost:8000/api/config/reload
```

---

## 💻 Uso en Código

### Acceder a Configuración

```python
from config_manager import get_config

config = get_config()

# Obtener valores
slack_enabled = config.is_notification_enabled("slack")
webhook_url = config.get_webhook_url("slack")
threshold = config.get("ci.similarity_threshold", 0.7)
```

### Enviar Notificaciones

```python
from backend import send_slack_alert, send_teams_alert

payload = {
    "tester_id": "qa-001",
    "build_id": "v1.2.3",
    "severity": 0.8,
    "diff_count": 5
}

# Uso automático: lee de config.yaml
send_slack_alert(title="Build Failed", payload=payload)
send_teams_alert(title="Build Failed", payload=payload)

# O con webhook custom
send_slack_alert(webhook_url="https://...", title="...", payload=payload)
```

---

## 📋 Configuración Típica

### Mínimo (Solo Slack)

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    timeout: 5
    retry_count: 2
```

### Recomendado (Slack + Teams)

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    timeout: 5
    retry_count: 2
  
  teams:
    enabled: true
    webhook_url: "${TEAMS_WEBHOOK_URL}"
    timeout: 5
    retry_count: 2
  
  jira:
    enabled: false
```

### Completo (Todos los servicios)

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    retry_count: 2
  
  teams:
    enabled: true
    webhook_url: "${TEAMS_WEBHOOK_URL}"
    retry_count: 2
  
  jira:
    enabled: true
    base_url: "${JIRA_BASE_URL}"
    api_token: "${JIRA_API_TOKEN}"
    project_key: "QA"
    retry_count: 2
```

---

## 🎯 Casos de Uso

### Caso 1: CI Failure Notification

```python
# Cuando falla un CI test
is_failure = True

if is_failure:
    payload = {
        "tester_id": "ci-agent",
        "build_id": "v1.2.3-build-456",
        "severity": 0.95,
        "diff_count": 15
    }
    
    # Automáticamente envía a Slack, Teams y Jira según config.yaml
    send_slack_alert(title="🔴 CI Build Failed", payload=payload)
    send_teams_alert(title="🔴 CI Build Failed", payload=payload)
    
    issue_key = send_jira_issue(
        summary="Build v1.2.3 failed - 15 UI diffs detected",
        description="Multiple UI changes detected in build"
    )
```

### Caso 2: Model Training Complete

```python
# Cuando termina entrenamiento del modelo
model_accuracy = 0.92

send_slack_alert(
    title="✅ Model Training Complete",
    payload={
        "tester_id": "ml-trainer",
        "build_id": "model-v2.1",
        "severity": 0.0,  # Info, no error
        "diff_count": 0
    }
)
```

### Caso 3: Configuration Hot-Reload

```python
# Cambiar configuración sin reiniciar
# 1. Editar config.yaml
# 2. Hacer POST a /api/config/reload
# 3. Sistema cargará nueva configuración automáticamente

curl -X POST http://localhost:8000/api/config/reload
```

---

## 🔐 Seguridad

### ✅ Hacer

```bash
# Usar variables de entorno
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
python -m uvicorn backend:app
```

```yaml
# config.yaml
notifications:
  slack:
    webhook_url: "${SLACK_WEBHOOK_URL}"
```

### ❌ NO Hacer

```yaml
# ❌ NUNCA guardar URLs directas en config.yaml
notifications:
  slack:
    webhook_url: "https://hooks.slack.com/services/xxx"
```

```bash
# ❌ NUNCA exponer valores sensibles en logs
# Los valores sensibles se enmascaran automáticamente ✅
```

---

## 🧪 Testing

### Verificar Configuración

```bash
python test_config.py
```

Resultado esperado:
```
✅ PASS | GET /api/config
✅ PASS | GET /api/config/notifications
✅ PASS | GET /api/config/ci
✅ PASS | GET /api/config/ml
✅ PASS | GET /api/config/health
✅ PASS | POST /api/config/test-slack
✅ PASS | POST /api/config/test-teams
✅ PASS | POST /api/config/reload

📊 RESULTS: 8/8 tests passed (100.0%)
```

---

## 🚀 Deployment

### Local Development

```bash
# Con reload automático
python -m uvicorn backend:app --reload
```

### Production

```bash
# Con variables de entorno
export SLACK_WEBHOOK_URL="your-webhook-url"
export TEAMS_WEBHOOK_URL="your-webhook-url"

# Iniciar con múltiples workers
python -m uvicorn backend:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Docker

```dockerfile
FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Usar variables de entorno en tiempo de ejecución
CMD ["python", "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker run \
  -e SLACK_WEBHOOK_URL="https://..." \
  -e TEAMS_WEBHOOK_URL="https://..." \
  -p 8000:8000 \
  axiom-service
```

---

## 📈 Monitoreo

### Health Check Regular

```bash
# Implementar health check en CI/CD
curl -f http://localhost:8000/api/config/health || exit 1
```

### Verificar Servicios

```bash
curl http://localhost:8000/api/config/notifications | jq .
```

### Logs

```bash
# Monitorear logs en tiempo real
tail -f logs/axiom.log | grep "slack\|teams\|jira"
```

---

## ❓ FAQs

**P: ¿Cómo cambio la configuración sin reiniciar?**
R: Edita `config.yaml` y haz POST a `/api/config/reload`

**P: ¿Cómo uso valores diferentes por ambiente?**
R: Usa variables de entorno. Cada ambiente exporta diferentes valores.

**P: ¿Qué pasa si el webhook falla?**
R: Reintentos automáticos (configurable). Log completo en axiom.log

**P: ¿Cómo verifico que funciona?**
R: POST a `/api/config/test-slack` o ejecuta `python test_config.py`

**P: ¿Puedo deshabilitar un servicio?**
R: Sí. En `config.yaml`: `slack: enabled: false`

---

## 📚 Documentación Completa

- **CONFIG_SYSTEM.md** - Guía completa + todas las APIs
- **IMPLEMENTATION_SUMMARY.md** - Resumen técnico detallado
- **config.yaml.example** - Plantilla con comentarios completos
- **.env.example** - Variables de entorno

---

## 🎓 Ejemplos Adicionales

### Obtener Todas las Configuraciones

```bash
curl http://localhost:8000/api/config | python -m json.tool
```

### Usar ConfigManager en Scripts

```python
#!/usr/bin/env python3
from config_manager import get_config, init_config

# Opción 1: Usar singleton global
config = get_config()

# Opción 2: Inicializar con ruta custom
config = init_config("/path/to/config.yaml")

# Acceder valores
print(config.get("notifications.slack.webhook_url"))
print(config.to_dict())  # Todo con sensibles enmascarados
```

---

## 🤝 Contribuir

Mejoras sugeridas:
- [ ] Validación de webhooks al cargar config
- [ ] Persistencia de config a través de API
- [ ] Sincronización con Key Vault
- [ ] UI Dashboard de configuración

---

## 📞 Soporte

Ver **CONFIG_SYSTEM.md** → Troubleshooting para problemas comunes.

---

**Última actualización**: Nov 30, 2025  
**Versión**: 1.0 (Stable)
