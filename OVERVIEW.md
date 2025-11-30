# 🎯 VISIÓN GENERAL - Sistema de Configuración

## 📊 Arquitectura de Alto Nivel

```
┌──────────────────────────────────────────────────────────────────────┐
│                      AxiomServiceIA Backend                          │
│                      (FastAPI Application)                           │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌──────────────────────┐    ┌───────────────────────┐
        │   ConfigManager      │    │   Notification Funcs  │
        │  (config_manager.py) │    │  (backend.py)        │
        │                      │    │                       │
        │ • Load config.yaml   │    │ • send_slack_alert   │
        │ • Resolve ${ENV}     │◄───┤ • send_teams_alert   │
        │ • Thread-safe access │    │ • send_jira_issue    │
        │ • Mask secrets       │    │                       │
        └──────────────────────┘    └───────────────────────┘
                    │                         │
                    ├─────────────┬───────────┘
                    │             │
                    ▼             ▼
            ┌─────────────────────────────┐
            │  API Endpoints (10 nuevos)  │
            │                             │
            │ GET /api/config             │
            │ GET /api/config/...         │
            │ POST /api/config/test-*     │
            │ POST /api/config/reload     │
            └─────────────────────────────┘
                    │
         ┌──────────┼──────────┬──────────┐
         ▼          ▼          ▼          ▼
      ┌─────┐  ┌──────┐  ┌─────┐  ┌──────┐
      │ CLI │  │Health│  │Slack│  │Teams │
      │Test │  │Check │  │Hooks│  │Hooks │
      └─────┘  └──────┘  └─────┘  └──────┘
```

---

## 🔄 Flujo de Configuración (Startup)

```
1. Backend Start
   └─ init_config()
      └─ ConfigManager()
         ├─ Lee config.yaml
         │  ├─ Si no existe → fallback defaults
         │  └─ Si existe → parsear YAML
         │
         ├─ Resuelve variables de entorno
         │  └─ ${SLACK_WEBHOOK_URL} → valor real
         │
         ├─ Valida estructura
         │  └─ Check required fields
         │
         └─ Singleton global
            └─ get_config() siempre retorna misma instancia
```

---

## 📨 Flujo de Envío de Notificación

```
send_slack_alert(title, payload)
  │
  ├─ webhook_url = config.get_webhook_url("slack")
  │
  ├─ is_enabled = config.is_notification_enabled("slack")
  │
  ├─ if not enabled:
  │  └─ return False  (early exit)
  │
  ├─ for attempt in range(retry_count):
  │  │
  │  ├─ try:
  │  │  ├─ Prepare message
  │  │  ├─ POST to webhook
  │  │  ├─ Handle response
  │  │  ├─ if success: return True
  │  │  │
  │  │  └─ if fail and retry_available:
  │  │     └─ sleep(1)
  │  │
  │  └─ except:
  │     ├─ log error
  │     ├─ if retry_available: continue
  │     └─ else: return False
  │
  └─ return False (all retries exhausted)
```

---

## 🏗️ Estructura de Archivos

```
AxiomServiceIA/
│
├─ 📝 Configuración
│  ├─ config.yaml              (🔐 Gitignored - Crear localmente)
│  ├─ config.yaml.example      (✅ Template con comentarios)
│  └─ .env.example             (✅ Variables de entorno)
│
├─ 💾 Core
│  ├─ config_manager.py        (✅ Gestor centralizado)
│  ├─ backend.py               (✅ + 10 endpoints, 3 funciones mejoradas)
│  └─ requirements.txt          (✅ + pyyaml, requests)
│
├─ 🧪 Tools
│  ├─ setup.py                 (✅ Setup interactivo)
│  └─ test_config.py           (✅ Suite de tests)
│
└─ 📚 Documentación
   ├─ INDEX.md                 (✅ Este índice)
   ├─ QUICKSTART_CONFIG.md     (✅ 5 min start)
   ├─ CONFIG_SYSTEM.md         (✅ Referencia completa)
   ├─ IMPLEMENTATION_SUMMARY.md (✅ Resumen técnico)
   ├─ IMPLEMENTATION_STATUS.md  (✅ Status de implementación)
   └─ FEATURE_COMPLETE.md      (✅ Checklist)
```

---

## 🎯 Flujo de Usuario

### Primera Vez (Setup)

```
1. python setup.py
   ├─ Crea config.yaml desde example
   ├─ Crea .env desde example
   ├─ Pregunta por webhooks
   └─ Guarda en .env

2. Editar .env con credenciales reales

3. python test_config.py
   ├─ Testa todos los endpoints
   ├─ Verifica configuración
   └─ OK si 8/8 tests passed

4. python -m uvicorn backend:app --reload
   └─ Backend corriendo y listo
```

---

### Uso Regular

```
1. Código envía notificación
   └─ send_slack_alert(title, payload)

2. ConfigManager obtiene config
   ├─ Lee config.yaml
   ├─ Valida que esté enabled
   └─ Obtiene webhook_url

3. Envía con retry automático
   ├─ Intento 1: success → return True
   ├─ Intento 1: fail → sleep, intento 2
   └─ Intento 2: success → return True

4. Si falla todo
   └─ Log error, return False
```

---

### Cambiar Configuración

```
1. Editar config.yaml
   └─ Cambiar enabled, webhook_url, etc.

2. Sin reiniciar:
   └─ curl -X POST http://localhost:8000/api/config/reload

3. Nueva configuración activa inmediatamente
   └─ No requiere reiniciar servidor
```

---

## 📊 Matriz de Decisión

```
¿Necesitas...?

├─ Empezar rápido (5 min)
│  └─ Lee: QUICKSTART_CONFIG.md
│     Haz: python setup.py
│
├─ Entender la arquitectura
│  └─ Lee: IMPLEMENTATION_SUMMARY.md
│     Mira: Diagramas en este archivo
│
├─ Referencia API completa
│  └─ Lee: CONFIG_SYSTEM.md
│     Busca: Endpoint que necesites
│
├─ Probar notificaciones
│  └─ Ejecuta: python test_config.py
│     O: curl -X POST http://localhost:8000/api/config/test-slack
│
├─ Debugging/Troubleshooting
│  └─ Lee: CONFIG_SYSTEM.md → Troubleshooting
│     Ejecuta: curl http://localhost:8000/api/config/health
│
├─ Integrar en tu código
│  └─ Lee: IMPLEMENTATION_SUMMARY.md → Uso en Código
│     Copia: Patrón que necesites
│
├─ Deployar en producción
│  └─ Lee: CONFIG_SYSTEM.md → Deployment
│     Sigue: Pasos para tu plataforma
│
└─ Ver qué se completó
   └─ Lee: FEATURE_COMPLETE.md
      Revisa: Checklist ✅
```

---

## 🔐 Matriz de Seguridad

```
Valor Sensible          ¿En config.yaml?    ¿En Logs?    ¿En API?
─────────────────────────────────────────────────────────────────
Slack Webhook           ${VAR}              Masked      Masked
Teams Webhook           ${VAR}              Masked      Masked
Jira API Token          ${VAR}              Masked      Masked
Webhook URLs            ${VAR}              Masked      Masked

✅ = Nunca expuesto
❌ = Puede estar expuesto si mal configurado
```

---

## 📈 Mejoras Realizadas

### Antes
```
❌ Webhooks hardcoded en código
❌ Cambios requieren reiniciar
❌ Manejo de errores inconsistente
❌ Sin retry automático
❌ Variables de entorno dispersas
```

### Ahora
```
✅ Webhooks en config.yaml
✅ Hot-reload sin restart
✅ Error handling robusto
✅ Retry automático + timeouts
✅ Variables centralizadas
✅ 10 endpoints API nuevos
✅ Health checks integrados
✅ Tests automatizados
✅ Documentación completa
```

---

## 🎓 Conceptos Clave

### 1. ConfigManager (Singleton)
```
Una única instancia global que:
├─ Carga configuración una sola vez
├─ Proporciona acceso thread-safe
├─ Resuelve variables de entorno
└─ Enmasca valores sensibles
```

### 2. Hot-Reload
```
Cambiar config sin reiniciar:
├─ Editar config.yaml
├─ POST /api/config/reload
└─ Nueva config activa inmediatamente
```

### 3. Retry Automático
```
Si webhook falla:
├─ Intento 1: fail
├─ Wait 1 segundo
├─ Intento 2: success ✅
└─ O fail después de N intentos
```

### 4. Environment Awareness
```
Soporta múltiples ambientes:
├─ Development: local config.yaml
├─ Staging: variables de entorno
└─ Production: secretos en Key Vault
```

---

## 🚀 Performance & Scalability

```
Configuración
├─ Carga inicial: ~10ms (una sola vez en startup)
├─ Acceso a config: ~1μs (en memoria, singleton)
├─ Health check: ~50ms (validaciones simples)
└─ Envío de notificación: ~500-2000ms (depende del webhook)

Escalabilidad
├─ Soporta múltiples servicios
├─ Retry automático no bloquea
├─ Thread-safe para acceso concurrente
└─ Sin overhead significativo
```

---

## 📞 Paths de Soporte

```
Problema                        Solución
─────────────────────────────────────────────────────────────
Config no se carga              Check: config.yaml existe?
Variable no se resuelve         Format: "${VAR_NAME}" en yaml
Webhook no funciona             Test: /api/config/test-slack
Health check falla              Leer: Troubleshooting en docs
Tests fallan                    Ejecutar: python test_config.py
Quiero entender el sistema      Leer: IMPLEMENTATION_SUMMARY.md
```

---

## ✅ Checklist de Verificación

```
Después de implementar, verificar:

□ config.yaml existe (creado desde example)
□ .env tiene tus credenciales
□ python test_config.py pasa 8/8
□ curl http://localhost:8000/api/config/health → OK
□ curl -X POST http://localhost:8000/api/config/test-slack → OK
□ Logs muestran mensajes correctamente
□ Cambios en config.yaml se reflejan después de reload
□ Valores sensibles NO aparecen en logs/API
□ Backend inicia sin errores
```

---

## 🎯 Próximos Pasos

1. ✅ **Setup**: `python setup.py`
2. ✅ **Test**: `python test_config.py`
3. ✅ **Integrar**: Usar en tu código
4. ✅ **Documentar**: Tu caso de uso
5. ✅ **Deploy**: A producción

---

**Versión**: 1.0 Stable  
**Última actualización**: Nov 30, 2025  
**Status**: ✅ Ready for Production

Para empezar → **QUICKSTART_CONFIG.md**
