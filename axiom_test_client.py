"""
AxiomTestClient - Cliente SDK para integración con automatización de pruebas
Permite a testers iniciar sesiones, registrar validaciones y obtener reportes en tiempo real.

Uso:
    # Python + Selenium
    from axiom_test_client import AxiomTestSession
    
    session = AxiomTestSession(
        base_url="http://localhost:8000",
        test_name="Test Login Flow",
        tester_id="automation_bot_01",
        build_id="8.19.20251107",
        app_name="com.grability.rappi",
        expected_flow=["login_screen", "home_screen", "cart_screen"]
    )
    
    # En tu test de Selenium
    session.start()
    
    # ... ejecutar test ...
    
    # Notificar cambios de pantalla
    driver.get(url)
    session.record_event(screen_name="login_screen", header_text="Iniciar Sesión")
    
    # Validaciones adicionales
    session.add_validation(
        name="Login fields visible",
        rule={"has_email_field": True, "has_password_field": True},
        passed=True
    )
    
    # Finalizar y obtener reporte
    report = session.end()
    print(report)
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Resultado de la ejecución del test"""
    success: bool
    session_id: str
    test_name: str
    duration_seconds: float
    events_received: int
    events_validated: int
    flow_completion_percentage: float
    errors: List[Dict]
    expected_flow: List[str]
    actual_flow: List[str]
    timestamp: str


class AxiomTestSession:
    """
    Cliente para gestionar sesiones de prueba automatizada.
    
    Proporciona métodos para:
    - Crear y gestionar sesiones
    - Registrar eventos de pantalla
    - Validar flujos en tiempo real
    - Obtener reportes automáticos
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        test_name: str = "Untitled Test",
        tester_id: str = "automation_tester",
        build_id: str = "1.0.0",
        app_name: str = "com.example.app",
        expected_flow: List[str] = None,
        metadata: Optional[Dict] = None,
        timeout: int = 30
    ):
        """
        Inicializa el cliente de sesión de prueba.
        
        Args:
            base_url: URL base del servicio Axiom
            test_name: Nombre del test
            tester_id: ID único del tester/bot de automatización
            build_id: ID del build bajo prueba
            app_name: Nombre de la aplicación
            expected_flow: Lista de pantallas esperadas en orden
            metadata: Datos adicionales (URL, dispositivo, ambiente, etc.)
            timeout: Timeout para requests HTTP (segundos)
        """
        self.base_url = base_url.rstrip('/')
        self.test_name = test_name
        self.tester_id = tester_id
        self.build_id = build_id
        self.app_name = app_name
        self.expected_flow = expected_flow or []
        self.metadata = metadata or {}
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.is_active = False

    def create(self) -> bool:
        """
        Crea una nueva sesión de prueba en el servidor.
        
        Returns:
            True si la creación fue exitosa
        """
        try:
            endpoint = f"{self.base_url}/api/automation/session/create"
            payload = {
                "test_name": self.test_name,
                "tester_id": self.tester_id,
                "build_id": self.build_id,
                "app_name": self.app_name,
                "expected_flow": self.expected_flow,
                "metadata": self.metadata
            }

            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            self.session_id = data.get("session_id")

            if self.session_id:
                logger.info(f"✅ Sesión creada: {self.session_id} - {self.test_name}")
                return True
            else:
                logger.error(f"❌ No se recibió session_id en respuesta: {data}")
                return False

        except Exception as e:
            logger.error(f"❌ Error creando sesión: {e}")
            return False

    def start(self) -> bool:
        """
        Inicia la sesión (la marca como RUNNING en el servidor).
        
        Returns:
            True si la sesión fue iniciada
        """
        if not self.session_id:
            logger.error("⚠️ Primero debes crear la sesión con create()")
            return False

        try:
            endpoint = f"{self.base_url}/api/automation/session/{self.session_id}/start"
            response = requests.post(endpoint, timeout=self.timeout)
            response.raise_for_status()

            self.is_active = True
            self.start_time = time.time()
            logger.info(f"▶️ Sesión iniciada: {self.session_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error iniciando sesión: {e}")
            return False

    def record_event(
        self,
        screen_name: str,
        header_text: str = "",
        event_type: str = "screen_change",
        additional_data: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Registra un evento (cambio de pantalla) en la sesión.
        
        Args:
            screen_name: Nombre/ID de la pantalla
            header_text: Texto del header/título
            event_type: Tipo de evento (default: screen_change)
            additional_data: Datos adicionales del evento
        
        Returns:
            (success, message)
        """
        if not self.session_id:
            return False, "Sesión no creada"

        if not self.is_active:
            return False, "Sesión no está activa"

        try:
            endpoint = f"{self.base_url}/api/automation/session/{self.session_id}/event"
            payload = {
                "screen_name": screen_name,
                "header_text": header_text,
                "event_type": event_type,
                "additional_data": additional_data or {}
            }

            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            validation_result = data.get("validation_result")
            message = data.get("message", "")

            logger.debug(f"📊 Evento registrado: {screen_name} - {validation_result}")
            return True, message

        except Exception as e:
            logger.error(f"❌ Error registrando evento: {e}")
            return False, str(e)

    def add_validation(
        self,
        name: str,
        rule: Dict,
        passed: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Registra una validación adicional (assertion) en la sesión.
        
        Args:
            name: Nombre de la validación
            rule: Diccionario con reglas/criterios
            passed: Si la validación pasó
            error_message: Mensaje de error si falló
        
        Returns:
            True si se registró exitosamente
        """
        if not self.session_id:
            logger.error("⚠️ Sesión no creada")
            return False

        try:
            endpoint = f"{self.base_url}/api/automation/session/{self.session_id}/validation"
            payload = {
                "validation_name": name,
                "rule": rule,
                "passed": passed,
                "error_message": error_message
            }

            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"✓ Validación registrada: {name} - {'✅ PASS' if passed else '❌ FAIL'}")
            return True

        except Exception as e:
            logger.error(f"❌ Error registrando validación: {e}")
            return False

    def end(self, success: bool = True) -> Optional[TestResult]:
        """
        Finaliza la sesión y obtiene el reporte.
        
        Args:
            success: Si el test finalizó exitosamente
        
        Returns:
            TestResult con reporte de la sesión
        """
        if not self.session_id:
            logger.error("⚠️ Sesión no creada")
            return None

        try:
            endpoint = f"{self.base_url}/api/automation/session/{self.session_id}/end"
            payload = {
                "success": success,
                "final_status": "completed" if success else "failed"
            }

            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            self.is_active = False

            duration = time.time() - (self.start_time or time.time())

            result = TestResult(
                success=data.get("success", False),
                session_id=self.session_id,
                test_name=data.get("test_name", self.test_name),
                duration_seconds=duration,
                events_received=data.get("events_received", 0),
                events_validated=data.get("events_validated", 0),
                flow_completion_percentage=data.get("flow_completion_percentage", 0),
                errors=data.get("validation_errors", []),
                expected_flow=data.get("expected_flow", []),
                actual_flow=data.get("actual_flow", []),
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"🏁 Sesión finalizada: {self.session_id}")
            logger.info(f"   Estado: {'✅ COMPLETADA' if result.success else '❌ FALLÓ'}")
            logger.info(f"   Duración: {result.duration_seconds:.2f}s")
            logger.info(f"   Flujo: {result.events_validated}/{len(self.expected_flow)} pantallas")
            logger.info(f"   Errores: {len(result.errors)}")

            return result

        except Exception as e:
            logger.error(f"❌ Error finalizando sesión: {e}")
            return None

    def get_status(self) -> Optional[Dict]:
        """Obtiene el estado actual de la sesión"""
        if not self.session_id:
            return None

        try:
            endpoint = f"{self.base_url}/api/automation/session/{self.session_id}"
            response = requests.get(endpoint, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado: {e}")
            return None

    def print_report(self, result: TestResult):
        """Imprime un reporte formateado del test"""
        if not result:
            print("❌ No hay resultado para mostrar")
            return

        print("\n" + "="*70)
        print(f"📋 REPORTE DE AUTOMATIZACIÓN - {result.test_name}")
        print("="*70)
        print(f"🔑 Session ID: {result.session_id}")
        print(f"⏱️  Duración: {result.duration_seconds:.2f} segundos")
        print(f"📊 Eventos: {result.events_received} recibidos, {result.events_validated} validados")
        print(f"📈 Flujo: {result.flow_completion_percentage:.1f}% completado")
        print(f"✅ Resultado: {'EXITOSO' if result.success else 'FALLÓ'}")

        print(f"\n📍 Flujo esperado ({len(result.expected_flow)} pantallas):")
        for i, screen in enumerate(result.expected_flow, 1):
            print(f"  {i}. {screen}")

        print(f"\n📍 Flujo realizado ({len(result.actual_flow)} pantallas):")
        for i, screen in enumerate(result.actual_flow, 1):
            print(f"  {i}. {screen}")

        if result.errors:
            print(f"\n❌ Errores ({len(result.errors)}):")
            for error in result.errors:
                print(f"  • {error.get('type')}: {error.get('received', 'N/A')}")
                if error.get('expected'):
                    print(f"    Esperado: {error.get('expected')}")

        print("\n" + "="*70 + "\n")


class AxiomTestContext:
    """Context manager para usar AxiomTestSession"""

    def __init__(self, session: AxiomTestSession):
        self.session = session

    def __enter__(self):
        self.session.create()
        self.session.start()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        result = self.session.end(success=success)
        self.session.print_report(result)
