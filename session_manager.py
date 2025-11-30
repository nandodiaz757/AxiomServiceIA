"""
Session Manager for Automation Testing Integration
Gestiona sesiones de pruebas automatizadas, monitorea eventos y valida flujos en tiempo real.
"""

import sqlite3
import json
import time
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_NAME = "axiom_test.db"


class SessionStatus(str, Enum):
    """Estados de una sesión de prueba"""
    CREATED = "created"           # Sesión creada, esperando eventos
    RUNNING = "running"           # Recibiendo eventos
    VALIDATING = "validating"     # En proceso de validación
    PASSED = "passed"             # Validación exitosa
    FAILED = "failed"             # Validación falló
    COMPLETED = "completed"       # Finalizada exitosamente
    ERROR = "error"               # Error durante ejecución
    ABANDONED = "abandoned"       # Timeout sin eventos


class EventValidationResult(str, Enum):
    """Resultado de validación de evento"""
    MATCH = "match"               # Evento coincide con lo esperado
    UNEXPECTED = "unexpected"     # Evento no esperado en este punto
    MISSING = "missing"           # Evento esperado no llegó
    ANOMALY = "anomaly"           # Evento anómalo


@dataclass
class TestSession:
    """Representa una sesión de prueba automatizada"""
    session_id: str
    test_name: str
    tester_id: str
    build_id: str
    app_name: str
    expected_flow: List[str]       # Secuencia esperada de pantallas
    created_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    status: SessionStatus = SessionStatus.CREATED
    events_received: int = 0
    events_validated: int = 0
    flow_position: int = 0         # Posición actual en el flujo esperado
    last_event_at: Optional[float] = None
    validation_errors: List[Dict] = None
    screen_sequence: List[str] = None  # Pantallas reales recibidas
    metadata: Dict = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.screen_sequence is None:
            self.screen_sequence = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        """Convierte a diccionario para serialización"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = datetime.fromtimestamp(self.created_at).isoformat()
        if self.started_at:
            data['started_at'] = datetime.fromtimestamp(self.started_at).isoformat()
        if self.ended_at:
            data['ended_at'] = datetime.fromtimestamp(self.ended_at).isoformat()
        if self.last_event_at:
            data['last_event_at'] = datetime.fromtimestamp(self.last_event_at).isoformat()
        return data


@dataclass
class ValidationEvent:
    """Evento de validación recibido durante la sesión"""
    event_id: str
    session_id: str
    screen_name: str
    header_text: str
    timestamp: float
    event_type: str
    class_name: str
    text: str
    content_description: str
    result: EventValidationResult
    expected: Optional[str] = None
    actual: Optional[str] = None
    anomaly_score: float = 0.0
    is_baseline: bool = False


class SessionManager:
    """
    Gestor centralizado de sesiones de prueba automatizada.
    Coordina eventos, valida flujos y genera reportes.
    """

    def __init__(self):
        self.sessions: Dict[str, TestSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.event_handlers: Dict[str, List[callable]] = {}  # session_id -> [callbacks]
        self._init_db()

    def _init_db(self):
        """Inicializa tablas de BD para persistencia de sesiones"""
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            
            # Tabla de sesiones
            c.execute("""
                CREATE TABLE IF NOT EXISTS test_sessions (
                    session_id TEXT PRIMARY KEY,
                    test_name TEXT NOT NULL,
                    tester_id TEXT NOT NULL,
                    build_id TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    expected_flow TEXT NOT NULL,  -- JSON list
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    status TEXT DEFAULT 'created',
                    events_received INTEGER DEFAULT 0,
                    events_validated INTEGER DEFAULT 0,
                    flow_position INTEGER DEFAULT 0,
                    last_event_at TIMESTAMP,
                    screen_sequence TEXT,  -- JSON list
                    validation_errors TEXT,  -- JSON list
                    metadata TEXT,  -- JSON dict
                    UNIQUE(session_id)
                )
            """)

            # Tabla de eventos por sesión
            c.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    screen_name TEXT,
                    header_text TEXT,
                    event_type TEXT,
                    class_name TEXT,
                    text TEXT,
                    content_description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    validation_result TEXT,
                    expected TEXT,
                    actual TEXT,
                    anomaly_score REAL DEFAULT 0.0,
                    is_baseline INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES test_sessions(session_id),
                    INDEX idx_session_events (session_id, timestamp)
                )
            """)

            # Tabla de validaciones agregadas
            c.execute("""
                CREATE TABLE IF NOT EXISTS session_validations (
                    validation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    validation_name TEXT NOT NULL,
                    rule TEXT,  -- JSON con reglas
                    passed INTEGER DEFAULT 0,  -- 0/1
                    error_message TEXT,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES test_sessions(session_id),
                    INDEX idx_session_validations (session_id)
                )
            """)

            # Tabla de reportes de sesión
            c.execute("""
                CREATE TABLE IF NOT EXISTS session_reports (
                    report_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    summary TEXT,  -- JSON con resumen
                    total_events INTEGER,
                    matched_events INTEGER,
                    unexpected_events INTEGER,
                    anomalies_detected INTEGER,
                    flow_completion_percentage REAL,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES test_sessions(session_id)
                )
            """)

            conn.commit()

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Obtiene o crea un lock para una sesión"""
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]

    def create_session(
        self,
        test_name: str,
        tester_id: str,
        build_id: str,
        app_name: str,
        expected_flow: List[str],
        metadata: Optional[Dict] = None
    ) -> TestSession:
        """
        Crea una nueva sesión de prueba.
        
        Args:
            test_name: Nombre descriptivo del test
            tester_id: ID del tester/automatizador
            build_id: ID del build bajo prueba
            app_name: Nombre de la aplicación
            expected_flow: Lista de pantallas esperadas en orden
            metadata: Datos adicionales (URL, dispositivo, ambiente, etc.)
        
        Returns:
            TestSession creada
        """
        session_id = str(uuid.uuid4())[:8].upper()
        
        session = TestSession(
            session_id=session_id,
            test_name=test_name,
            tester_id=tester_id,
            build_id=build_id,
            app_name=app_name,
            expected_flow=expected_flow,
            created_at=time.time(),
            metadata=metadata or {}
        )

        self.sessions[session_id] = session

        # Persistir en BD
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO test_sessions (
                    session_id, test_name, tester_id, build_id, app_name,
                    expected_flow, metadata, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, test_name, tester_id, build_id, app_name,
                json.dumps(expected_flow), json.dumps(metadata or {}),
                SessionStatus.CREATED.value
            ))
            conn.commit()

        logger.info(f"✅ Sesión creada: {session_id} - {test_name}")
        return session

    def start_session(self, session_id: str) -> bool:
        """Inicia una sesión (marca como RUNNING)"""
        if session_id not in self.sessions:
            logger.warning(f"⚠️ Sesión no encontrada: {session_id}")
            return False

        session = self.sessions[session_id]
        session.status = SessionStatus.RUNNING
        session.started_at = time.time()

        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE test_sessions
                SET status = ?, started_at = datetime('now')
                WHERE session_id = ?
            """, (SessionStatus.RUNNING.value, session_id))
            conn.commit()

        logger.info(f"▶️ Sesión iniciada: {session_id}")
        return True

    async def process_event(
        self,
        session_id: str,
        screen_name: str,
        header_text: str,
        event_type: str = "screen_change",
        additional_data: Optional[Dict] = None
    ) -> Tuple[bool, EventValidationResult, str]:
        """
        Procesa un evento recibido durante la sesión.
        Valida si coincide con el flujo esperado.
        
        Args:
            session_id: ID de la sesión
            screen_name: Nombre de la pantalla
            header_text: Texto del header/título
            event_type: Tipo de evento
            additional_data: Datos adicionales del evento
        
        Returns:
            (success, validation_result, message)
        """
        async with self._get_lock(session_id):
            if session_id not in self.sessions:
                return False, EventValidationResult.ANOMALY, f"Sesión no encontrada: {session_id}"

            session = self.sessions[session_id]

            # Validar estado
            if session.status not in [SessionStatus.RUNNING, SessionStatus.VALIDATING]:
                return False, EventValidationResult.ANOMALY, f"Sesión no está en ejecución: {session.status.value}"

            # Actualizar timestampe
            session.last_event_at = time.time()
            session.events_received += 1

            # Normalizar screen_name para comparación
            normalized_screen = (screen_name or "").strip().lower()
            session.screen_sequence.append(normalized_screen)

            # Validar contra flujo esperado
            expected_at_position = None
            if session.flow_position < len(session.expected_flow):
                expected_at_position = session.expected_flow[session.flow_position].lower()

            validation_result = EventValidationResult.ANOMALY
            message = ""

            if normalized_screen == expected_at_position:
                validation_result = EventValidationResult.MATCH
                session.flow_position += 1
                session.events_validated += 1
                message = f"✅ Evento coincide: {screen_name} (posición {session.flow_position}/{len(session.expected_flow)})"

            elif normalized_screen in [s.lower() for s in session.expected_flow]:
                # Screen esperado pero en orden incorrecto
                validation_result = EventValidationResult.UNEXPECTED
                message = f"⚠️ Pantalla inesperada aquí: {screen_name} (esperado: {expected_at_position})"
                session.validation_errors.append({
                    "type": "unexpected_screen",
                    "received": screen_name,
                    "expected": expected_at_position,
                    "timestamp": session.last_event_at
                })
            else:
                # Screen no esperado en absoluto
                validation_result = EventValidationResult.ANOMALY
                message = f"❌ Pantalla no en flujo esperado: {screen_name}"
                session.validation_errors.append({
                    "type": "anomaly_screen",
                    "received": screen_name,
                    "timestamp": session.last_event_at
                })

            # Persistir evento en BD
            event_id = str(uuid.uuid4())
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO session_events (
                        event_id, session_id, screen_name, header_text,
                        event_type, validation_result, expected, actual
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id, session_id, screen_name, header_text,
                    event_type, validation_result.value,
                    expected_at_position, normalized_screen
                ))
                conn.commit()

            # Ejecutar callbacks registrados
            if session_id in self.event_handlers:
                for callback in self.event_handlers[session_id]:
                    try:
                        await callback(session, validation_result)
                    except Exception as e:
                        logger.error(f"Error en callback: {e}")

            logger.info(f"📊 [{session_id}] {message}")

            return True, validation_result, message

    def register_event_handler(self, session_id: str, callback: callable):
        """Registra un callback para eventos de la sesión"""
        if session_id not in self.event_handlers:
            self.event_handlers[session_id] = []
        self.event_handlers[session_id].append(callback)

    def add_validation(
        self,
        session_id: str,
        validation_name: str,
        rule: Dict,
        passed: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """Agrega una validación adicional a la sesión"""
        if session_id not in self.sessions:
            return False

        validation_id = str(uuid.uuid4())

        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO session_validations (
                    validation_id, session_id, validation_name,
                    rule, passed, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                validation_id, session_id, validation_name,
                json.dumps(rule), 1 if passed else 0, error_message
            ))
            conn.commit()

        return True

    def end_session(self, session_id: str, final_status: SessionStatus = SessionStatus.COMPLETED) -> Optional[Dict]:
        """
        Finaliza una sesión y genera reporte.
        
        Args:
            session_id: ID de la sesión
            final_status: Estado final (COMPLETED, FAILED, ERROR, etc.)
        
        Returns:
            Reporte de la sesión
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        session.status = final_status
        session.ended_at = time.time()

        # Calcular métricas
        duration = session.ended_at - (session.started_at or session.created_at)
        completion_percentage = (session.flow_position / len(session.expected_flow) * 100) if session.expected_flow else 0

        report = {
            "session_id": session_id,
            "test_name": session.test_name,
            "status": final_status.value,
            "duration_seconds": duration,
            "events_received": session.events_received,
            "events_validated": session.events_validated,
            "flow_completion_percentage": completion_percentage,
            "expected_flow": session.expected_flow,
            "actual_flow": session.screen_sequence,
            "validation_errors": session.validation_errors,
            "errors_count": len(session.validation_errors),
            "success": final_status == SessionStatus.COMPLETED and len(session.validation_errors) == 0
        }

        # Persistir en BD
        report_id = str(uuid.uuid4())
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE test_sessions
                SET status = ?, ended_at = datetime('now'),
                    screen_sequence = ?, validation_errors = ?
                WHERE session_id = ?
            """, (
                final_status.value,
                json.dumps(session.screen_sequence),
                json.dumps(session.validation_errors),
                session_id
            ))

            c.execute("""
                INSERT INTO session_reports (
                    report_id, session_id, summary,
                    total_events, matched_events, unexpected_events,
                    flow_completion_percentage
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id, session_id,
                json.dumps(report),
                session.events_received,
                session.events_validated,
                len(session.validation_errors),
                completion_percentage
            ))
            conn.commit()

        logger.info(f"🏁 Sesión finalizada: {session_id} - Status: {final_status.value}")
        logger.info(f"   📈 Flujo: {session.flow_position}/{len(session.expected_flow)} pantallas")
        logger.info(f"   ✅ Validaciones exitosas: {session.events_validated}/{session.events_received} eventos")
        logger.info(f"   ❌ Errores: {len(session.validation_errors)}")

        return report

    def get_session(self, session_id: str) -> Optional[TestSession]:
        """Obtiene una sesión por ID"""
        return self.sessions.get(session_id)

    def get_session_report(self, session_id: str) -> Optional[Dict]:
        """Obtiene el reporte de una sesión"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return session.to_dict()

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        tester_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Lista sesiones con filtros opcionales"""
        sessions = list(self.sessions.values())

        if status:
            sessions = [s for s in sessions if s.status == status]
        if tester_id:
            sessions = [s for s in sessions if s.tester_id == tester_id]

        return [s.to_dict() for s in sorted(
            sessions,
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]]

    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """Limpia sesiones antiguas sin cerrar"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        expired = [
            sid for sid, session in self.sessions.items()
            if session.status == SessionStatus.CREATED and session.created_at < cutoff_time
        ]

        for sid in expired:
            self.end_session(sid, SessionStatus.ABANDONED)
            del self.sessions[sid]

        logger.info(f"🧹 Limpias {len(expired)} sesiones expiradas")
        return len(expired)

    def validate_session_flow(self, session_id: str) -> Tuple[bool, str]:
        """
        Valida si una sesión completó el flujo esperado.
        
        Returns:
            (success, message)
        """
        if session_id not in self.sessions:
            return False, "Sesión no encontrada"

        session = self.sessions[session_id]

        if session.flow_position < len(session.expected_flow):
            missing = session.expected_flow[session.flow_position:]
            return False, f"Flujo incompleto. Faltaron: {', '.join(missing)}"

        if len(session.validation_errors) > 0:
            return False, f"Se detectaron {len(session.validation_errors)} errores de validación"

        return True, "Flujo completado exitosamente ✅"


# Singleton global
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Obtiene instancia global del SessionManager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def init_session_manager() -> SessionManager:
    """Inicializa el SessionManager"""
    global _session_manager
    _session_manager = SessionManager()
    return _session_manager
