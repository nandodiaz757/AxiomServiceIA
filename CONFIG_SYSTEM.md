# 🔧 Sistema de Configuración - AxiomServiceIA

## Descripción General

AxiomServiceIA ahora incluye un **ConfigManager centralizado** que gestiona:
- 📨 Webhooks de notificaciones (Slack, Teams, Jira)
- 🤖 Configuración de modelos ML
- 🔄 Parámetros de CI/CD
- 💾 Configuración de base de datos
- 📝 Configuración de logging

## Estructura del Archivo `config.yaml`

### 1. Notificaciones

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"  # Variable de entorno
    timeout: 5
    retry_count: 2
    
  teams:
    enabled: true
    webhook_url: "${TEAMS_WEBHOOK_URL}"
    timeout: 5
    retry_count: 2
    
  jira:
    enabled: true
    base_url: "${JIRA_BASE_URL}"
    api_token: "${JIRA_API_TOKEN}"
    project_key: "QA"
    issue_type: "Task"
```

**Características:**
- ✅ Variables de entorno automáticas: `${VAR_NAME}`
- ✅ Retry automático configurable
- ✅ Timeouts personalizables
- ✅ Habilitar/deshabilitar servicios sin cambiar código

### 2. Configuración de CI/CD

```yaml
ci:
  similarity_threshold: 0.7      # Umbral para cambios "significativos"
  auto_report_failures: true     # Reportar fallos automáticamente
  max_results: 20               # Máximo de resultados en check-diff
```

### 3. Configuración de Modelos ML

```yaml
ml:
  train_general_on_collect: true
  min_samples_for_training: 3
  batch_size: 500
  use_general_as_base: true
```

## Variables de Entorno

Las variables de entorno se resuelven automáticamente en tiempo de carga:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export TEAMS_WEBHOOK_URL="https://outlook.webhook.office.com/webhookb2/YOUR/WEBHOOK/URL"
export JIRA_BASE_URL="https://your-domain.atlassian.net"
export JIRA_API_TOKEN="your-api-token"
```

## API de Configuración

### 📊 Obtener Configuración Actual

```bash
curl http://localhost:8000/api/config
```

Respuesta (valores sensibles enmascarados):
```json
{
  "status": "ok",
  "config": {
    "notifications": {
      "slack": {
        "enabled": true,
        "webhook_url": "https://***url***"
      }
    }
  },
  "file_path": "/path/to/config.yaml"
}
```

### 📨 Obtener Solo Notificaciones

```bash
curl http://localhost:8000/api/config/notifications
```

Respuesta:
```json
{
  "status": "ok",
  "slack": {
    "enabled": true,
    "has_webhook": true
  },
  "teams": {
    "enabled": false,
    "has_webhook": false
  },
  "jira": {
    "enabled": true,
    "has_credentials": true
  }
}
```

### 🧪 Probar Notificación en Slack

```bash
curl -X POST http://localhost:8000/api/config/test-slack
```

Resultado:
```json
{
  "status": "ok",
  "message": "Test message sent to Slack"
}
```

### 🧪 Probar Notificación en Teams

```bash
curl -X POST http://localhost:8000/api/config/test-teams
```

### 🔄 Recargar Configuración (sin reiniciar)

```bash
curl -X POST http://localhost:8000/api/config/reload
```

### 🏥 Health Check de Configuración

```bash
curl http://localhost:8000/api/config/health
```

Respuesta:
```json
{
  "status": "ok",
  "overall": "healthy",
  "checks": {
    "slack": {
      "enabled": true,
      "configured": true,
      "ready": true
    },
    "teams": {
      "enabled": false,
      "configured": false,
      "ready": false
    },
    "jira": {
      "enabled": true,
      "configured": true,
      "ready": true
    },
    "database": {
      "path": "./axiom.db",
      "exists": true
    }
  }
}
```

### 📋 Obtener Configuración de CI

```bash
curl http://localhost:8000/api/config/ci
```

### 🤖 Obtener Configuración de ML

```bash
curl http://localhost:8000/api/config/ml
```

## Uso en Código Python

### Acceso Simple

```python
from config_manager import get_config

config = get_config()

# Obtener valor con notación de punto
slack_url = config.get("notifications.slack.webhook_url")
threshold = config.get("ci.similarity_threshold", 0.7)  # con default

# Verificar si un servicio está habilitado
if config.is_notification_enabled("slack"):
    print("Slack está habilitado")

# Obtener webhook URL
webhook_url = config.get_webhook_url("slack")
```

### Acceder a Secciones Completas

```python
ml_config = config.get_section("ml")
print(ml_config["batch_size"])
```

### Recargar Configuración en Tiempo Real

```python
config.reload()  # Sin necesidad de reiniciar el servidor
```

### Obtener Diccionario Completo (para debugging)

```python
full_config = config.to_dict()  # Valores sensibles enmascarados
```

## Funciones de Notificación Actualizadas

### Slack

```python
from backend import send_slack_alert

# Opción 1: Usar configuración automática
send_slack_alert(
    title="CI Failure",
    payload={
        "tester_id": "tester-001",
        "build_id": "v1.0.0",
        "severity": 0.8,
        "diff_count": 5
    }
)

# Opción 2: Webhook personalizado
send_slack_alert(
    webhook_url="https://hooks.slack.com/...",
    title="Custom Alert",
    payload={...}
)
```

### Teams

```python
from backend import send_teams_alert

# Usar configuración automática
send_teams_alert(
    title="CI Failure",
    payload={
        "tester_id": "tester-001",
        "build_id": "v1.0.0",
        "severity": 0.8,
        "diff_count": 5
    }
)
```

### Jira

```python
from backend import send_jira_issue

# Usar configuración automática
issue_key = send_jira_issue(
    summary="UI differences detected",
    description="Found 5 UI changes in build v1.0.0"
)

# Retorna: "QA-123" o None si falló
```

## Características Principales

### ✅ Resolución de Variables de Entorno

```yaml
# config.yaml
notifications:
  slack:
    webhook_url: "${SLACK_WEBHOOK_URL}"  # ← Se resuelve automáticamente
```

```bash
# Terminal
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
python -m uvicorn backend:app
# La URL se cargará correctamente
```

### ✅ Retry Automático

Todas las funciones de notificación incluyen reintentos configurables:

```yaml
notifications:
  slack:
    retry_count: 2    # Reintentar 2 veces si falla
    timeout: 5        # Timeout de 5 segundos
```

### ✅ Enmascaramiento de Valores Sensibles

Los valores sensibles se enmascaran en logs y respuestas API:

```json
{
  "webhook_url": "https://***url***"  // Enmascarado en respuestas
}
```

### ✅ Health Check Integrado

Verifica automáticamente que todos los servicios estén correctamente configurados.

## Desarrollo y Testing

### 1. Setup Local

```bash
# Crear archivo config.yaml en la raíz del proyecto
cp config.yaml.example config.yaml

# Configurar variables de entorno
export SLACK_WEBHOOK_URL="your-webhook-url"
export TEAMS_WEBHOOK_URL="your-webhook-url"

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
python -m uvicorn backend:app --reload
```

### 2. Probar Configuración

```bash
# Check de salud
curl http://localhost:8000/api/config/health

# Probar Slack
curl -X POST http://localhost:8000/api/config/test-slack

# Probar Teams
curl -X POST http://localhost:8000/api/config/test-teams
```

### 3. Script de Ejemplo

```python
from config_manager import get_config
from backend import send_slack_alert, send_teams_alert, send_jira_issue

# Inicializar config
config = get_config()

# Notificar a todos los servicios habilitados
payload = {
    "tester_id": "qa-001",
    "build_id": "v1.2.3",
    "severity": 0.8,
    "diff_count": 10
}

if config.is_notification_enabled("slack"):
    send_slack_alert(title="Build Failed", payload=payload)

if config.is_notification_enabled("teams"):
    send_teams_alert(title="Build Failed", payload=payload)

if config.is_notification_enabled("jira"):
    issue = send_jira_issue(
        summary="Build v1.2.3 failed",
        description="Multiple UI differences detected"
    )
    print(f"Jira issue created: {issue}")
```

## Troubleshooting

### Variable de Entorno No Se Resuelve

```yaml
# ❌ Incorrecto
webhook_url: ${SLACK_WEBHOOK_URL}

# ✅ Correcto
webhook_url: "${SLACK_WEBHOOK_URL}"
```

### Config.yaml No Se Encuentra

```python
from config_manager import init_config

# Usar ruta personalizada
config = init_config("/path/to/config.yaml")
```

### Recargar Configuración Sin Reiniciar

```bash
# Hacer POST al endpoint reload
curl -X POST http://localhost:8000/api/config/reload

# O desde Python
from config_manager import get_config
config = get_config()
config.reload()
```

## Notas de Seguridad

⚠️ **IMPORTANTE:**
- Nunca comitear valores reales de webhooks en el repositorio
- Siempre usar variables de entorno para secretos
- El archivo `config.yaml` en producción debe estar gitignored
- Las respuestas API enmascaran valores sensibles automáticamente

## Roadmap

- [ ] Validación de schema para config.yaml
- [ ] Endpoints para modificar configuración en tiempo real
- [ ] Persistencia de cambios de config a través de API
- [ ] Auditoría de cambios de configuración
- [ ] Configuración por tenant/workspace
