# ✅ Sistema de Configuración - Implementación Completa

## 📋 Resumen de Cambios

### 1. **Nuevos Archivos Creados**

```
✅ config_manager.py          - Gestor centralizado de configuración
✅ config.yaml               - Archivo de configuración principal
✅ config.yaml.example       - Plantilla de configuración (ejemplo)
✅ .env.example             - Plantilla de variables de entorno
✅ CONFIG_SYSTEM.md         - Documentación completa del sistema
✅ setup.py                 - Script interactivo de setup
✅ test_config.py           - Suite de tests para validar configuración
```

### 2. **Archivos Modificados**

```
✅ backend.py
   - Importado ConfigManager
   - Actualizado send_slack_alert() con retry automático
   - Actualizado send_teams_alert() con retry automático
   - Actualizado send_jira_issue() con retry automático
   - Agregados 10 nuevos endpoints de configuración
   - Inicialización de config al startup

✅ requirements.txt
   - Agregado: pyyaml==6.0.1
   - Agregado: requests==2.31.0
```

---

## 🔌 Nuevos Endpoints API

### Configuración General

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/config` | Obtener configuración completa (valores sensibles enmascarados) |
| POST | `/api/config/reload` | Recargar configuración sin reiniciar servidor |

### Notificaciones

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/config/notifications` | Obtener estado de servicios de notificación |
| POST | `/api/config/test-slack` | Enviar mensaje de prueba a Slack |
| POST | `/api/config/test-teams` | Enviar mensaje de prueba a Teams |

### ML y CI

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/config/ci` | Obtener configuración de CI/CD |
| GET | `/api/config/ml` | Obtener configuración de modelos ML |

### Health & Diagnostics

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/config/health` | Health check de toda la configuración |

---

## 🎯 Características Principales

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
# ✅ La URL se cargará correctamente
```

### ✅ Retry Automático

```yaml
notifications:
  slack:
    retry_count: 2    # Reintentar 2 veces si falla
    timeout: 5        # Timeout de 5 segundos
```

Las funciones de notificación ahora:
- Reintentan automáticamente si fallan
- Respetan timeouts configurables
- Registran intentos en logs

### ✅ Enmascaramiento de Valores Sensibles

```json
{
  "webhook_url": "https://***url***"  // Enmascarado en respuestas API
}
```

Los valores sensibles NO se exponen en:
- Respuestas de API
- Logs
- Debugging

### ✅ Health Check Integrado

```bash
curl http://localhost:8000/api/config/health
```

Verifica automáticamente:
- Servicios habilitados vs configurados
- Base de datos accesible
- Estado general del sistema

---

## 🚀 Quick Start

### 1. Setup Inicial (Automático)

```bash
# Opción A: Interactivo
python setup.py

# Opción B: Manual
cp config.yaml.example config.yaml
cp .env.example .env
# Editar config.yaml y .env con tus valores
```

### 2. Configurar Variables de Entorno

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export TEAMS_WEBHOOK_URL="https://outlook.webhook.office.com/webhookb2/YOUR/WEBHOOK/URL"
export JIRA_BASE_URL="https://your-domain.atlassian.net"
export JIRA_API_TOKEN="your-api-token"
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Configuración

```bash
# Health check
curl http://localhost:8000/api/config/health

# Ver estado de notificaciones
curl http://localhost:8000/api/config/notifications

# Probar Slack
curl -X POST http://localhost:8000/api/config/test-slack

# Probar Teams
curl -X POST http://localhost:8000/api/config/test-teams
```

### 5. Ejecutar Tests

```bash
python test_config.py
```

---

## 📊 Ejemplo de Uso en Código

### Acceso a Configuración

```python
from config_manager import get_config

config = get_config()

# Obtener valores específicos
slack_url = config.get("notifications.slack.webhook_url")
threshold = config.get("ci.similarity_threshold", 0.7)

# Verificar si servicio está habilitado
if config.is_notification_enabled("slack"):
    print("✅ Slack is enabled")

# Obtener webhook URL
webhook_url = config.get_webhook_url("slack")
```

### Enviar Notificaciones

```python
from backend import send_slack_alert, send_teams_alert, send_jira_issue

# Las funciones usan automáticamente la configuración
payload = {
    "tester_id": "qa-001",
    "build_id": "v1.2.3",
    "severity": 0.8,
    "diff_count": 10
}

# Slack
send_slack_alert(title="Build Failed", payload=payload)

# Teams
send_teams_alert(title="Build Failed", payload=payload)

# Jira
issue = send_jira_issue(
    summary="Build failed",
    description="Multiple UI differences detected"
)
```

---

## 🔧 Configuración Avanzada

### Habilitar/Deshabilitar Servicios

```yaml
# config.yaml
notifications:
  slack:
    enabled: false      # ← Desactivar Slack
  teams:
    enabled: true       # ← Activar Teams
  jira:
    enabled: true
```

### Ajustar Parámetros

```yaml
# config.yaml
notifications:
  slack:
    timeout: 10         # Aumentar timeout
    retry_count: 3      # Más reintentos

ci:
  similarity_threshold: 0.8   # Umbral más alto
  max_results: 50             # Más resultados
```

---

## 🧪 Testing

### Script Interactivo

```bash
python test_config.py
```

Prueba:
- ✅ Carga de configuración
- ✅ Endpoints de configuración
- ✅ Funciones de notificación
- ✅ Health checks

### Pruebas Individuales

```bash
# Obtener configuración
curl http://localhost:8000/api/config

# Probar notificación
curl -X POST http://localhost:8000/api/config/test-slack

# Recargar sin reiniciar
curl -X POST http://localhost:8000/api/config/reload

# Health check
curl http://localhost:8000/api/config/health
```

---

## 📝 Ejemplos de Configuración

### Mínimo (solo Slack)

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
  teams:
    enabled: false
  jira:
    enabled: false
```

### Completo (Slack + Teams + Jira)

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    retry_count: 3
  teams:
    enabled: true
    webhook_url: "${TEAMS_WEBHOOK_URL}"
    retry_count: 3
  jira:
    enabled: true
    base_url: "${JIRA_BASE_URL}"
    api_token: "${JIRA_API_TOKEN}"
    project_key: "QA"
    retry_count: 2
```

### Desarrollo (Local)

```yaml
notifications:
  slack:
    enabled: false      # Desactivo para desarrollo
  teams:
    enabled: false
  jira:
    enabled: false

logging:
  level: "DEBUG"        # Más verbose
  file: "./logs/debug.log"
```

---

## 🔐 Seguridad

### ⚠️ Nunca hacer en Producción

```yaml
# ❌ NUNCA
notifications:
  slack:
    webhook_url: "https://hooks.slack.com/services/xxx"  # URL real en config
```

### ✅ Siempre hacer

```yaml
# ✅ BIEN
notifications:
  slack:
    webhook_url: "${SLACK_WEBHOOK_URL}"  # Variable de entorno
```

```bash
# ✅ BIEN
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/xxx"
python -m uvicorn backend:app
```

---

## 📚 Archivos de Documentación

- **CONFIG_SYSTEM.md** - Documentación completa
- **config.yaml.example** - Plantilla con comentarios
- **.env.example** - Template de variables de entorno
- **setup.py** - Script de configuración interactivo
- **test_config.py** - Suite de tests automatizados

---

## 🎓 Flujo de Configuración

```
┌─────────────────────────────────────────────┐
│     1. Startup (backend.py)                 │
│        - init_config()                      │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│     2. ConfigManager Carga config.yaml      │
│        - Lee archivo YAML                   │
│        - Resuelve ${VAR_NAME}               │
│        - Valida estructura                  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│     3. Servicios Usan Configuración         │
│        - send_slack_alert()                 │
│        - send_teams_alert()                 │
│        - send_jira_issue()                  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│     4. Endpoints API Exponen Config         │
│        - /api/config                        │
│        - /api/config/notifications          │
│        - /api/config/health                 │
│        - /api/config/reload                 │
└─────────────────────────────────────────────┘
```

---

## ✨ Ventajas

✅ **Centralizado**: Una sola fuente de verdad
✅ **Flexible**: Soporta variables de entorno
✅ **Seguro**: Enmascaramiento automático de secretos
✅ **Resiliente**: Retry automático en notificaciones
✅ **Observable**: Health checks integrados
✅ **Hot-reload**: Recargar sin reiniciar servidor
✅ **Fácil Setup**: Script interactivo de configuración
✅ **Bien Documentado**: Docs completas + ejemplos

---

## 🔄 Próximas Mejoras (Roadmap)

- [ ] Validación de schema para config.yaml
- [ ] Endpoints para modificar configuración en tiempo real
- [ ] Persistencia de cambios de config a través de API
- [ ] Auditoría de cambios de configuración
- [ ] Configuración por tenant/workspace
- [ ] Sincronización con Azure Key Vault / AWS Secrets Manager

---

## 📞 Soporte

Problemas comunes y soluciones en **CONFIG_SYSTEM.md** → Troubleshooting

