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
import psycopg2
from psycopg2.extras import RealDictCursor
from db import get_conn

logger = logging.getLogger(__name__)

# =====================================
# ENUMS y DATACLASSES
# =====================================

class SessionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    COMPLETED = "completed"
    ERROR = "error"
    ABANDONED = "abandoned"

class EventValidationResult(str, Enum):
    MATCH = "match"
    UNEXPECTED = "unexpected"
    MISSING = "missing"
    ANOMALY = "anomaly"

@dataclass
class TestSession:
    session_id: str
    test_name: str
    tester_id: str
    build_id: str
    app_name: str
    expected_flow: List[str]
    created_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    status: SessionStatus = SessionStatus.CREATED
    events_received: int = 0
    events_validated: int = 0
    flow_position: int = 0
    last_event_at: Optional[float] = None
    validation_errors: List[Dict] = None
    screen_sequence: List[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.screen_sequence is None:
            self.screen_sequence = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
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

# =====================================
# SESSION MANAGER (PostgreSQL)
# =====================================

class SessionManager:
    def __init__(self, conn_params: dict = get_conn()):
        self.sessions: Dict[str, TestSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.event_handlers: Dict[str, List[callable]] = {}
        self.conn_params = conn_params
        self._init_db()

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def _init_db(self):
        with self._get_connection() as conn:
            with conn.cursor() as c:
                # Sesiones
                c.execute("""
                    CREATE TABLE IF NOT EXISTS test_sessions (
                        session_id TEXT PRIMARY KEY,
                        test_name TEXT NOT NULL,
                        tester_id TEXT NOT NULL,
                        build_id TEXT NOT NULL,
                        app_name TEXT NOT NULL,
                        expected_flow JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        started_at TIMESTAMP WITH TIME ZONE,
                        ended_at TIMESTAMP WITH TIME ZONE,
                        status TEXT DEFAULT 'created',
                        events_received INTEGER DEFAULT 0,
                        events_validated INTEGER DEFAULT 0,
                        flow_position INTEGER DEFAULT 0,
                        last_event_at TIMESTAMP WITH TIME ZONE,
                        screen_sequence JSONB,
                        validation_errors JSONB,
                        metadata JSONB
                    )
                """)
                # Eventos
                c.execute("""
                    CREATE TABLE IF NOT EXISTS session_events (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES test_sessions(session_id),
                        screen_name TEXT,
                        header_text TEXT,
                        event_type TEXT,
                        class_name TEXT,
                        text TEXT,
                        content_description TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        validation_result TEXT,
                        expected TEXT,
                        actual TEXT,
                        anomaly_score DOUBLE PRECISION DEFAULT 0.0,
                        is_baseline BOOLEAN DEFAULT FALSE
                    )
                """)
                c.execute("CREATE INDEX IF NOT EXISTS idx_session_events ON session_events(session_id, timestamp)")
                # Validaciones
                c.execute("""
                    CREATE TABLE IF NOT EXISTS session_validations (
                        validation_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES test_sessions(session_id),
                        validation_name TEXT NOT NULL,
                        rule JSONB,
                        passed BOOLEAN DEFAULT FALSE,
                        error_message TEXT,
                        evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                c.execute("CREATE INDEX IF NOT EXISTS idx_session_validations ON session_validations(session_id)")
                # Reportes
                c.execute("""
                    CREATE TABLE IF NOT EXISTS session_reports (
                        report_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES test_sessions(session_id),
                        summary JSONB,
                        total_events INTEGER,
                        matched_events INTEGER,
                        unexpected_events INTEGER,
                        anomalies_detected INTEGER,
                        flow_completion_percentage DOUBLE PRECISION,
                        generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
            conn.commit()

    def _get_lock(self, session_id: str) -> asyncio.Lock:
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

        with self._get_connection() as conn:
            with conn.cursor() as c:
                c.execute("""
                    INSERT INTO test_sessions (
                        session_id, test_name, tester_id, build_id, app_name,
                        expected_flow, metadata, status
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                """, (
                    session_id, test_name, tester_id, build_id, app_name,
                    json.dumps(expected_flow),
                    json.dumps(metadata or {}),
                    SessionStatus.CREATED.value
                ))
                conn.commit()

        logger.info(f"✅ Sesión creada: {session_id} - {test_name}")
        return session

    def start_session(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        session = self.sessions[session_id]
        session.status = SessionStatus.RUNNING
        session.started_at = time.time()
        with self._get_connection() as conn:
            with conn.cursor() as c:
                c.execute("""
                    UPDATE test_sessions
                    SET status=%s, started_at=NOW()
                    WHERE session_id=%s
                """, (SessionStatus.RUNNING.value, session_id))
                conn.commit()
        return True

    # =====================================
    # El resto de funciones (process_event, add_validation, end_session, list_sessions)
    # se migran igual: reemplazar sqlite3 por psycopg2, ? por %s, JSONB y NOW()
    # =====================================

# =====================================
# Singleton
# =====================================
_session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

def init_session_manager() -> SessionManager:
    global _session_manager
    _session_manager = SessionManager()
    return _session_manager