# =====================================================
# Config Manager - Gestión centralizada de configuración
# =====================================================
import os
import yaml
import logging
from typing import Any, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Gestor centralizado de configuración que:
    - Carga config.yaml
    - Resuelve variables de entorno (${VAR_NAME})
    - Proporciona acceso seguro a valores con defaults
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Carga la configuración desde el archivo YAML."""
        if not self.config_path.exists():
            logger.warning(
                f"⚠️ Config file not found at {self.config_path}. "
                "Using default configuration."
            )
            self._config = self._default_config()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
            
            # Resolver variables de entorno
            self._config = self._resolve_env_vars(raw_config)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
        
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}. Using defaults.")
            self._config = self._default_config()

    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        Resuelve recursivamente variables de entorno en formato ${VAR_NAME}.
        Ejemplo: "${SLACK_WEBHOOK_URL}" → valor de la variable de entorno.
        """
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                value = os.environ.get(var_name)
                if value is None:
                    logger.warning(
                        f"⚠️ Environment variable '{var_name}' not found. "
                        f"Using placeholder: {obj}"
                    )
                return value or obj
            return obj
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        return obj

    def _default_config(self) -> Dict[str, Any]:
        """Retorna configuración por defecto si no existe config.yaml."""
        return {
            "notifications": {
                "slack": {
                    "enabled": False,
                    "webhook_url": None,
                    "timeout": 5,
                    "retry_count": 2,
                },
                "teams": {
                    "enabled": False,
                    "webhook_url": None,
                    "timeout": 5,
                    "retry_count": 2,
                },
                "jira": {
                    "enabled": False,
                    "base_url": None,
                    "api_token": None,
                    "project_key": "QA",
                    "issue_type": "Task",
                    "timeout": 10,
                    "retry_count": 2,
                },
            },
            "ci": {
                "similarity_threshold": 0.7,
                "auto_report_failures": True,
                "max_results": 20,
            },
            "ml": {
                "train_general_on_collect": True,
                "min_samples_for_training": 3,
                "batch_size": 500,
                "use_general_as_base": True,
            },
            "database": {
                "path": "./axiom.db",
                "cleanup_old_records_days": 90,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "./logs/axiom.log",
                "max_file_size": 10485760,
                "backup_count": 5,
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False,
                "workers": 4,
            },
            "features": {
                "flow_validation_enabled": True,
                "feedback_system_enabled": True,
                "diff_deduplication_enabled": True,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto.
        Ejemplo: config.get('notifications.slack.webhook_url')
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Obtiene una sección completa de la configuración."""
        return self._config.get(section, {})

    def is_notification_enabled(self, service: str) -> bool:
        """Verifica si un servicio de notificación está habilitado."""
        return self.get(f"notifications.{service}.enabled", False)

    def get_webhook_url(self, service: str) -> Optional[str]:
        """
        Obtiene la URL del webhook para un servicio.
        Soporta: slack, teams, jira
        """
        if service == "slack":
            return self.get("notifications.slack.webhook_url")
        elif service == "teams":
            return self.get("notifications.teams.webhook_url")
        elif service == "jira":
            return self.get("notifications.jira.base_url")
        return None

    def reload(self):
        """Recarga la configuración (útil para hot-reload)."""
        logger.info("🔄 Reloading configuration...")
        self._load_config()

    def to_dict(self) -> Dict[str, Any]:
        """Retorna toda la configuración como diccionario (para debugging)."""
        # Crear copia sin exponer valores sensibles
        safe_config = self._mask_sensitive_values(self._config.copy())
        return safe_config

    def _mask_sensitive_values(self, obj: Any) -> Any:
        """Enmascara valores sensibles para logging."""
        if isinstance(obj, str):
            if any(keyword in obj.lower() for keyword in ["webhook", "token", "password", "api"]):
                if len(obj) > 10:
                    return obj[:5] + "***" + obj[-5:]
            return obj
        elif isinstance(obj, dict):
            return {k: self._mask_sensitive_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._mask_sensitive_values(item) for item in obj]
        return obj


# =====================================================
# Instancia global singleton
# =====================================================
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Retorna la instancia singleton del ConfigManager."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def init_config(config_path: str = "config.yaml") -> ConfigManager:
    """Inicializa el ConfigManager con una ruta personalizada."""
    global _config_instance
    _config_instance = ConfigManager(config_path)
    return _config_instance
