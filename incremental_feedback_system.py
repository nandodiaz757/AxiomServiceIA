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


class IncrementalFeedbackSystem:
    """Gestiona retroalimentación incremental para modelos de predicción."""
    
    def __init__(self, db_name: str = "feedback.db"):
        self.db_name = db_name
        self._init_feedback_tables()
    
    def _init_feedback_tables(self):
        """Crear tablas para retroalimentación incremental."""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        # Tabla de feedback de diffs (aprobado/rechazado)
        c.execute("""
            CREATE TABLE IF NOT EXISTS diff_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diff_hash TEXT NOT NULL UNIQUE,
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
        
        # Tabla de aprendizaje: diffs similares que ya fueron aprobados
        c.execute("""
            CREATE TABLE IF NOT EXISTS approved_diff_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name TEXT NOT NULL,
                pattern_hash TEXT NOT NULL,
                pattern_signature TEXT NOT NULL,
                screen_name TEXT NOT NULL,
                approval_count INTEGER DEFAULT 1,
                rejection_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.5,  -- Score de cuán confiable es este patrón
                last_approved TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern_hash, app_name)
            );
        """)
        
        # Tabla de métricas de aprendizaje
        c.execute("""
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name TEXT NOT NULL,
                tester_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,  -- 'false_positives', 'precision', 'recall'
                metric_value REAL,
                build_version TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(app_name, tester_id, metric_name, build_version)
            );
        """)
        
        
        # Tabla de historial de decisiones del modelo
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        conn.close()
    
    def record_diff_feedback(
        self,
        diff_hash: str,
        diff_signature: str,
        app_name: str,
        tester_id: str,
        feedback_type: str,  # 'approved', 'rejected'
        build_version: str,
        screen_name: str
    ) -> bool:
        """
        Registra feedback sobre un diff (aprobado/rechazado).
        
        Args:
            diff_hash: Hash del diff
            diff_signature: Firma detallada del diff
            app_name: Nombre de la app
            tester_id: ID del tester
            feedback_type: 'approved' o 'rejected'
            build_version: Versión del build
            screen_name: Nombre de la pantalla
        
        Returns:
            True si se registró correctamente
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            timestamp = datetime.utcnow().isoformat()
            
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

            # c.execute("""
            #     INSERT INTO diff_feedback 
            #     (diff_hash, diff_signature, app_name, tester_id, build_version_first, feedback_type, feedback_count, last_seen, created_at)
            #     VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            #     ON CONFLICT(diff_hash, tester_id, app_name) DO UPDATE SET
            #         feedback_count = feedback_count + 1,
            #         feedback_type = ?,
            #         last_seen = ?
            # """, (
            #     diff_hash, diff_signature, app_name, tester_id, build_version,
            #     feedback_type, timestamp, timestamp,
            #     feedback_type, timestamp
            # ))
            
            # Actualizar tabla de patrones aprobados
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
            
            # Registrar métrica de aprendizaje
            approval_rate = self._calculate_approval_rate(app_name, tester_id)

            c.execute("""
                INSERT INTO learning_metrics
                (app_name, tester_id, metric_name, metric_value, build_version)
                VALUES (?, ?, 'approval_rate', ?, ?)
                ON CONFLICT(app_name, tester_id, metric_name, build_version)
                DO UPDATE SET metric_value = excluded.metric_value,
                            recorded_at = CURRENT_TIMESTAMP
            """, (app_name, tester_id, approval_rate, build_version))
            # c.execute("""
            #     INSERT INTO learning_metrics
            #     (app_name, tester_id, metric_name, metric_value, build_version)
            #     VALUES (?, ?, 'approval_rate', ?, ?)
            # """, (app_name, tester_id, approval_rate, build_version))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error registrando feedback: {e}")
            return False
    
    def is_diff_similar_to_approved(
        self,
        diff_signature: str,
        app_name: str,
        similarity_threshold: float = 0.75
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Verifica si un diff es similar a uno ya aprobado.
        
        Args:
            diff_signature: Firma del diff actual
            app_name: Nombre de la app
            similarity_threshold: Threshold de similitud (0-1)
        
        Returns:
            (es_similar, score_similitud, patrón_encontrado)
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Buscar patrones aprobados
            c.execute("""
                SELECT pattern_hash, pattern_signature, approval_count, confidence
                FROM approved_diff_patterns
                WHERE app_name = ? AND approval_count >= 1
                ORDER BY confidence DESC, last_approved DESC
                LIMIT 10
            """, (app_name,))
            
            approved_patterns = c.fetchall()
            conn.close()
            
            if not approved_patterns:
                return False, 0.0, None
            
            # Calcular similitud con cada patrón aprobado
            best_similarity = 0.0
            best_pattern = None
            
            for pattern_hash, pattern_sig, approval_count, confidence in approved_patterns:
                # Similitud semántica simple (puede ser mejorada con embeddings)
                similarity = self._calculate_signature_similarity(
                    diff_signature,
                    pattern_sig
                )
                
                # Ajustar por confianza del patrón
                adjusted_similarity = similarity * confidence
                
                if adjusted_similarity > best_similarity:
                    best_similarity = adjusted_similarity
                    best_pattern = pattern_hash
            
            is_similar = best_similarity >= similarity_threshold
            return is_similar, best_similarity, best_pattern
            
        except Exception as e:
            print(f"⚠️ Error verificando similitud: {e}")
            return False, 0.0, None
    
    def get_diff_priority(
        self,
        diff_hash: str,
        diff_signature: str,
        app_name: str,
        tester_id: str
    ) -> Dict[str, any]:
        """
        Calcula la prioridad de un diff basado en feedback histórico.
        
        Returns:
            {
                'priority': 'high' | 'medium' | 'low',
                'confidence': float 0-1,
                'reason': str,
                'similar_to_approved': bool,
                'similarity_score': float
            }
        """
        # Verificar similitud a aprobados
        is_similar, sim_score, pattern = self.is_diff_similar_to_approved(
            diff_signature, app_name
        )
        
        # Verificar historial del diff
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("""
            SELECT feedback_type, feedback_count FROM diff_feedback
            WHERE diff_hash = ? AND app_name = ? AND tester_id = ?
        """, (diff_hash, app_name, tester_id))
        
        feedback_record = c.fetchone()
        conn.close()
        
        # Lógica de prioridad
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
    
    def _create_pattern_hash(self, diff_signature: str) -> str:
        """Crear hash del patrón (más abstracto que diff_hash)."""
        # Extraer componentes principales del diff para crear patrón
        try:
            parts = diff_signature.split('|')[:3]  # Primeras 3 partes
            pattern = '|'.join(parts)
            return hashlib.sha256(pattern.encode()).hexdigest()[:16]
        except:
            return hashlib.sha256(diff_signature.encode()).hexdigest()[:16]
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calcular similitud entre dos firmas de diff."""
        try:
            # Similitud basada en componentes compartidos
            parts1 = set(sig1.split('|'))
            parts2 = set(sig2.split('|'))
            
            if not parts1 or not parts2:
                return 0.0
            
            intersection = len(parts1 & parts2)
            union = len(parts1 | parts2)
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_approval_rate(self, app_name: str, tester_id: str) -> float:
        """Calcular tasa de aprobación del tester."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            c.execute("""
                SELECT 
                    COUNT(CASE WHEN feedback_type = 'approved' THEN 1 END) as approved,
                    COUNT(*) as total
                FROM diff_feedback
                WHERE app_name = ? AND tester_id = ?
                AND created_at > datetime('now', '-30 days')
            """, (app_name, tester_id))
            
            result = c.fetchone()
            conn.close()
            
            if result and result[1] > 0:
                return result[0] / result[1]
            return 0.5  # Default si no hay datos
            
        except:
            return 0.5
    
    def get_learning_insights(self, app_name: str, tester_id: str) -> Dict:
        """Obtener insights de aprendizaje del tester."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Últimos 7 días de aprobaciones
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
            conn.close()
            
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


# ============= FUNCIONES DE INTEGRACIÓN PARA backend.py =============

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
