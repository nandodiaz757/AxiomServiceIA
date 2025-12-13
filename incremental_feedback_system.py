"""
Sistema de Retroalimentación Incremental para Diffs
====================================================
Mantiene el aprendizaje entre versiones evitando repetir diffs aprobados.

Flujo:
1. Diff detectado en v1 → mostrado al tester
2. Tester aprueba el diff → se guarda como "approved"
3. Diff similar aparece en v2 → modelo reconoce que fue aprobado antes
4. Acción: desapriorizar, NO mostrar, o mostrar con baja prioridad

Beneficios:
- Menos falsos positivos repetidos
- Modelo aprende evolución esperada
- Mejor experiencia de tester
- Predicciones más precisas
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from db import get_conn_cm

# ============================================
# Incremental Feedback System con pool de conexiones
# ============================================

class IncrementalFeedbackSystem:
    """Gestiona retroalimentación incremental para modelos de predicción."""
    
    def __init__(self, db_name: str = "accessibility.db"):
        self.db_name = db_name
        self._init_feedback_tables()
    
    def _init_feedback_tables(self):
        """Crear tablas para retroalimentación incremental."""
        #conn = get_conn()
        try:
            with get_conn_cm() as conn:
                with conn.cursor() as c:
                    #c = conn.cursor()
                    # Tabla de feedback de diffs
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS diff_feedback (
                            id SERIAL PRIMARY KEY,
                            diff_hash TEXT NOT NULL,
                            diff_signature TEXT NOT NULL,
                            app_name TEXT NOT NULL,
                            tester_id TEXT NOT NULL,
                            build_version_first TEXT,
                            build_version_approved TEXT,
                            feedback_type TEXT CHECK(feedback_type IN ('approved', 'rejected', 'pending')) DEFAULT 'pending',
                            feedback_count INTEGER DEFAULT 0,
                            confidence_score REAL DEFAULT 0.0,
                            approved_at TIMESTAMP,
                            rejected_at TIMESTAMP,
                            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(diff_hash, tester_id, app_name)
                        );
                    """)
                    
                    # Tabla de patrones aprobados
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS approved_diff_patterns (
                            id SERIAL PRIMARY KEY,
                            app_name TEXT NOT NULL,
                            pattern_hash TEXT NOT NULL,
                            pattern_signature TEXT NOT NULL,
                            screen_name TEXT NOT NULL,
                            approval_count INTEGER DEFAULT 1,
                            rejection_count INTEGER DEFAULT 0,
                            confidence REAL DEFAULT 0.5,
                            last_approved TIMESTAMP,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(pattern_hash, app_name)
                        );
                    """)
                    
                    # Tabla de métricas de aprendizaje
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS learning_metrics (
                            id SERIAL PRIMARY KEY,
                            app_name TEXT NOT NULL,
                            tester_id TEXT NOT NULL,
                            metric_name TEXT NOT NULL,
                            metric_value REAL,
                            build_version TEXT,
                            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(app_name, tester_id, metric_name, build_version)
                        );
                    """)
                    
                    # Tabla de historial de decisiones
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS model_decision_log (
                            id SERIAL PRIMARY KEY,
                            app_name TEXT NOT NULL,
                            diff_hash TEXT NOT NULL,
                            decision TEXT CHECK(decision IN ('show', 'hide', 'low_priority', 'high_priority')) DEFAULT 'show',
                            reason TEXT,
                            similarity_to_approved REAL,
                            model_confidence REAL,
                            user_feedback TEXT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    conn.commit()
        # finally:
        #     release_conn(conn)
        except Exception as e:
            print(f"❌ Error generate_tester_flow_report: {e}")
    # -------------------------------
    # Registro de feedback
    # -------------------------------
    
    def record_diff_feedback(
        self,
        diff_hash: str,
        diff_signature: str,
        app_name: str,
        tester_id: str,
        feedback_type: str,  # 'approved' o 'rejected'
        build_version: str,
        screen_name: str
    ) -> bool:
        try:
            #conn = get_conn()
            try:
                timestamp = datetime.utcnow().isoformat()
                with get_conn_cm() as conn:
                    with conn.cursor() as c:
                      
                        # Registrar feedback del diff
                        c.execute("""
                            INSERT INTO diff_feedback 
                            (diff_hash, diff_signature, app_name, tester_id, 
                            build_version_first, feedback_type, feedback_count, last_seen)
                            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                            ON CONFLICT(diff_hash, tester_id, app_name) DO UPDATE SET
                                feedback_count = feedback_count + 1,
                                feedback_type = excluded.feedback_type,
                                last_seen = excluded.last_seen
                        """, (
                            diff_hash,
                            diff_signature,
                            app_name,
                            tester_id,
                            build_version,
                            feedback_type,
                            timestamp
                        ))

                        # Actualizar patrones aprobados
                        if feedback_type == 'approved':
                            pattern_hash = self._create_pattern_hash(diff_signature)
                            c.execute("""
                                INSERT INTO approved_diff_patterns
                                (app_name, pattern_hash, pattern_signature, screen_name, approval_count, confidence)
                                VALUES (?, ?, ?, ?, 1, 0.7)
                                ON CONFLICT(pattern_hash, app_name) DO UPDATE SET
                                    approval_count = approval_count + 1,
                                    confidence = MIN(0.99, confidence + 0.05),
                                    last_approved = CURRENT_TIMESTAMP
                            """, (app_name, pattern_hash, diff_signature, screen_name))
                        
                        # Registrar métricas
                        approval_rate = self._calculate_approval_rate(app_name, tester_id)
                        c.execute("""
                            INSERT INTO learning_metrics
                            (app_name, tester_id, metric_name, metric_value, build_version)
                            VALUES (?, ?, 'approval_rate', ?, ?)
                            ON CONFLICT(app_name, tester_id, metric_name, build_version)
                            DO UPDATE SET metric_value = excluded.metric_value,
                                        recorded_at = CURRENT_TIMESTAMP
                        """, (app_name, tester_id, approval_rate, build_version))
                        
                        conn.commit()
                        return True
            finally:
                release_conn(conn)
        except Exception as e:
            print(f"❌ Error registrando feedback: {e}")
            return False
    
    # -------------------------------
    # Verificación de similitud con aprobados
    # -------------------------------
    
    def is_diff_similar_to_approved(
        self,
        diff_signature: str,
        app_name: str,
        similarity_threshold: float = 0.75
    ) -> Tuple[bool, float, Optional[str]]:
        try:
            with get_conn_cm() as conn:
                with conn.cursor() as c:
                    c.execute("""
                        SELECT pattern_hash, pattern_signature, approval_count, confidence
                        FROM approved_diff_patterns
                        WHERE app_name = %s AND approval_count >= 1
                        ORDER BY confidence DESC, last_approved DESC
                        LIMIT 10
                    """, (app_name,))
                    approved_patterns = c.fetchall()

            if not approved_patterns:
                return False, 0.0, None

            best_similarity = 0.0
            best_pattern = None

            for pattern_hash, pattern_sig, approval_count, confidence in approved_patterns:
                similarity = self._calculate_signature_similarity(diff_signature, pattern_sig)
                adjusted_similarity = similarity * confidence
                if adjusted_similarity > best_similarity:
                    best_similarity = adjusted_similarity
                    best_pattern = pattern_hash

            is_similar = best_similarity >= similarity_threshold
            return is_similar, best_similarity, best_pattern

        except Exception as e:
            print(f"⚠️ Error verificando similitud: {e}")
            return False, 0.0, None
    
    # -------------------------------
    # Obtener prioridad del diff
    # -------------------------------
    
    def get_diff_priority(
        self,
        diff_hash: str,
        diff_signature: str,
        app_name: str,
        tester_id: str
    ) -> Dict[str, any]:
        is_similar, sim_score, pattern = self.is_diff_similar_to_approved(diff_signature, app_name)
        
        # conn = get_conn()
        
        with get_conn_cm() as conn:
            with conn.cursor() as c:
                # c = conn.cursor()
                c.execute("""
                    SELECT feedback_type, feedback_count FROM diff_feedback
                    WHERE diff_hash = %s AND app_name = %s AND tester_id = %s
                """, (diff_hash, app_name, tester_id))
                feedback_record = c.fetchone()
        # finally:
        #     release_conn(conn)
        
        if is_similar and sim_score > 0.85:
            priority = 'low'
            reason = f"Similar a diff aprobado antes (sim={sim_score:.2f})"
            confidence = sim_score
        elif feedback_record and feedback_record[0] == 'approved':
            priority = 'low'
            reason = f"Diff ya fue aprobado {feedback_record[1]} veces"
            confidence = 0.95
        elif feedback_record and feedback_record[0] == 'rejected':
            priority = 'low'
            reason = "Diff fue rechazado previamente"
            confidence = 0.90
        else:
            priority = 'high'
            reason = "Nuevo diff no registrado en historial"
            confidence = 0.5
        
        return {
            'priority': priority,
            'confidence': confidence,
            'reason': reason,
            'similar_to_approved': is_similar,
            'similarity_score': sim_score
        }
    
    # -------------------------------
    # Helpers privados
    # -------------------------------
    
    def _create_pattern_hash(self, diff_signature: str) -> str:
        try:
            parts = diff_signature.split('|')[:3]
            pattern = '|'.join(parts)
            return hashlib.sha256(pattern.encode()).hexdigest()[:16]
        except:
            return hashlib.sha256(diff_signature.encode()).hexdigest()[:16]
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        try:
            parts1 = set(sig1.split('|'))
            parts2 = set(sig2.split('|'))
            if not parts1 or not parts2:
                return 0.0
            return len(parts1 & parts2) / len(parts1 | parts2)
        except:
            return 0.0
    
    def _calculate_approval_rate(self, app_name: str, tester_id: str) -> float:
        try:
            # conn = get_conn()
            
            with get_conn_cm() as conn:
                with conn.cursor() as c:
                # c = conn.cursor()
                    c.execute("""
                        SELECT 
                            COUNT(CASE WHEN feedback_type = 'approved' THEN 1 END) as approved,
                            COUNT(*) as total
                        FROM diff_feedback
                        WHERE app_name = ? AND tester_id = ?
                        AND created_at > datetime('now', '-30 days')
                    """, (app_name, tester_id))
                    result = c.fetchone()
            # finally:
            #     release_conn(conn)
            
            if result and result[1] > 0:
                return result[0] / result[1]
            return 0.5
        except:
            return 0.5
    
    def get_learning_insights(self, app_name: str, tester_id: str) -> Dict:
        try:
            # conn = get_conn()
        
            with get_conn_cm() as conn:
                with conn.cursor() as c:
                # c = conn.cursor()
                    c.execute("""
                        SELECT 
                            COUNT(CASE WHEN feedback_type = 'approved' THEN 1 END) as approved,
                            COUNT(CASE WHEN feedback_type = 'rejected' THEN 1 END) as rejected,
                            COUNT(*) as total
                        FROM diff_feedback
                        WHERE app_name = ? AND tester_id = ?
                        AND created_at > datetime('now', '-7 days')
                    """, (app_name, tester_id))
                    result = c.fetchone()
            # finally:
            #     release_conn(conn)
            
            if result:
                approved, rejected, total = result
                return {
                    'approval_rate_7d': approved / total if total > 0 else 0,
                    'approved_count': approved,
                    'rejected_count': rejected,
                    'total_feedbacks': total,
                    'improvement_trend': 'positive' if approved >= rejected else 'negative'
                }
            return {'error': 'No data available'}
        except Exception as e:
            return {'error': str(e)}    
        

# # ============= FUNCIONES DE INTEGRACIÓN PARA backend.py =============

def check_approved_diff_pattern(
    diff_signature: str,
    app_name: str,
    tester_id: str,
    feedback_system: IncrementalFeedbackSystem
) -> Dict[str, any]:
    """
    Verifica si un diff es similar a uno ya aprobado.
    Retorna info para decidir si mostrarlo o no.
    """
    is_similar, sim_score, pattern = feedback_system.is_diff_similar_to_approved(
        diff_signature, app_name
    )
    
    priority_info = feedback_system.get_diff_priority(
        "", diff_signature, app_name, tester_id
    )
    
    return {
        'should_show': priority_info['priority'] == 'high',
        'priority': priority_info['priority'],
        'is_similar_to_approved': is_similar,
        'similarity_score': sim_score,
        'confidence': priority_info['confidence'],
        'reason': priority_info['reason']
    }


def record_diff_decision(
    diff_hash: str,
    diff_signature: str,
    app_name: str,
    tester_id: str,
    build_version: str,
    decision: str,  # 'show', 'hide', 'low_priority'
    user_approved: bool,  # True si user aprobó, False si rechazó
    feedback_system: IncrementalFeedbackSystem
):
    """Registra la decisión y feedback del usuario sobre un diff."""
    
    feedback_type = 'approved' if user_approved else 'rejected'
    
    feedback_system.record_diff_feedback(
        diff_hash=diff_hash,
        diff_signature=diff_signature,
        app_name=app_name,
        tester_id=tester_id,
        feedback_type=feedback_type,
        build_version=build_version,
        screen_name="unknown"
    )        