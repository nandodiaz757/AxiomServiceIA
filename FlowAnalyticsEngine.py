"""
FlowAnalyticsEngine.py
Sistema avanzado de análisis de flujos con retroalimentación al tester.
Complementa HMM con análisis detallado y accionable.
"""

import sqlite3
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import os

logger = logging.getLogger(__name__)
DB_NAME = "accessibility.db"
FLOW_MODEL_DIR = "models/flows"

class FlowAnalyticsEngine:
    """
    Motor de análisis de flujos con:
    - Detección de deviaciones
    - Generación de reportes por tester
    - Historial de anomalías
    - Sugerencias de flujos correctos
    """
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.db = DB_NAME
        
    # ========================================
    # 1. ANÁLISIS DE DESVIACIONES
    # ========================================
    
    def analyze_deviation(self, session_key: str, expected_flow: List[str]) -> Dict:
        """
        Compara un flujo observado con el esperado.
        Retorna detalles de dónde se desvió y por qué.
        
        Args:
            session_key: Identificador de sesión
            expected_flow: Secuencia de pantallas esperadas
            
        Returns:
            {
                "session_key": str,
                "observed_flow": List[str],
                "expected_flow": List[str],
                "is_deviated": bool,
                "deviation_point": int,  # índice donde se desvió
                "deviation_details": {
                    "expected": str,
                    "actual": str,
                    "step_number": int
                },
                "similarity_score": float,  # 0.0-1.0
                "recovery_path": List[str],  # cómo volver al flujo normal
                "suggestions": List[str]
            }
        """
        observed_flow = self._get_session_flow(session_key)
        
        if not observed_flow:
            return {
                "session_key": session_key,
                "error": "No flow data found",
                "is_deviated": None
            }
        
        # Comparar flujos
        deviation_point = None
        for i, (expected, actual) in enumerate(zip(expected_flow, observed_flow)):
            if expected != actual:
                deviation_point = i
                break
        
        is_deviated = deviation_point is not None
        
        # Calcular similitud (Levenshtein-like)
        similarity = self._calculate_flow_similarity(expected_flow, observed_flow)
        
        # Sugerir path de recuperación
        recovery_path = []
        if is_deviated and deviation_point is not None:
            actual_at_deviation = observed_flow[deviation_point]
            expected_at_deviation = expected_flow[deviation_point]
            recovery_path = self._find_recovery_path(
                actual_at_deviation, 
                expected_at_deviation,
                expected_flow[deviation_point:]
            )
        
        return {
            "session_key": session_key,
            "observed_flow": observed_flow,
            "expected_flow": expected_flow,
            "is_deviated": is_deviated,
            "deviation_point": deviation_point,
            "deviation_details": {
                "expected": expected_flow[deviation_point] if deviation_point else None,
                "actual": observed_flow[deviation_point] if deviation_point and deviation_point < len(observed_flow) else None,
                "step_number": deviation_point + 1 if deviation_point is not None else None
            } if deviation_point is not None else None,
            "similarity_score": similarity,
            "recovery_path": recovery_path,
            "suggestions": self._generate_suggestions(is_deviated, similarity, observed_flow)
        }
    
    # ========================================
    # 2. REPORTES POR TESTER
    # ========================================
    
    def generate_tester_flow_report(self, tester_id: str, days: int = 7) -> Dict:
        """
        Genera reporte completo de flujos para un tester.
        Muestra: flujos usados, anomalías detectadas, mejoras.
        """
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            # Obtener todas las sesiones del tester en el período
            c.execute("""
                SELECT DISTINCT session_key, MIN(created_at)
                FROM accessibility_data
                WHERE tester_id = ? AND app_name = ? 
                  AND created_at > datetime('now', '-' || ? || ' days')
                GROUP BY session_key
                ORDER BY MIN(created_at) DESC
            """, (tester_id, self.app_name, days))
            
            sessions = c.fetchall()
        
        if not sessions:
            return {
                "tester_id": tester_id,
                "app_name": self.app_name,
                "period_days": days,
                "total_sessions": 0,
                "message": "No sessions found"
            }
        
        # Analizar cada sesión
        session_analyses = []
        anomaly_count = 0
        
        for session_key, _ in sessions:
            expected = self._get_most_common_flow()
            analysis = self.analyze_deviation(session_key, expected)
            session_analyses.append(analysis)
            
            if analysis.get("is_deviated"):
                anomaly_count += 1
        
        # Estadísticas
        avg_similarity = sum(
            s.get("similarity_score", 1.0) for s in session_analyses
        ) / len(session_analyses) if session_analyses else 1.0
        
        # Flujos más frecuentes del tester
        frequent_flows = self._get_tester_frequent_flows(tester_id, days, limit=5)
        
        return {
            "tester_id": tester_id,
            "app_name": self.app_name,
            "period_days": days,
            "total_sessions": len(sessions),
            "anomalous_sessions": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(sessions), 3) if sessions else 0,
            "avg_flow_similarity": round(avg_similarity, 3),
            "quality_score": round(avg_similarity * 100, 1),  # 0-100
            "session_analyses": session_analyses[:10],  # últimas 10
            "frequent_flows": frequent_flows,
            "recommendations": self._generate_recommendations(
                anomaly_count, len(sessions), frequent_flows
            ),
            "generated_at": datetime.now().isoformat()
        }
    
    # ========================================
    # 3. DASHBOARD DE FLUJOS
    # ========================================
    
    def get_flow_analytics_dashboard(self, include_testers: Optional[List[str]] = None) -> Dict:
        """
        Dashboard global de flujos con anomalías, tendencias, etc.
        """
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            # Total de sesiones
            query = "SELECT COUNT(DISTINCT session_key) FROM accessibility_data WHERE app_name = ?"
            params = [self.app_name]
            
            if include_testers:
                query += " AND tester_id IN ({})".format(
                    ",".join(["?" for _ in include_testers])
                )
                params.extend(include_testers)
            
            c.execute(query, params)
            total_sessions = c.fetchone()[0]
            
            # Distribuición de flujos
            c.execute("""
                SELECT header_text, COUNT(*) as count
                FROM accessibility_data
                WHERE app_name = ?
                GROUP BY header_text
                ORDER BY count DESC
                LIMIT 10
            """, (self.app_name,))
            
            screen_distribution = c.fetchall()
        
        # Calcular anomalías por pantalla (donde se interrumpen flujos)
        interruption_points = self._calculate_interruption_points()
        
        return {
            "app_name": self.app_name,
            "total_sessions": total_sessions,
            "screen_distribution": {
                screen: count for screen, count in screen_distribution
            },
            "interruption_hotspots": interruption_points,
            "most_common_flow": self._get_most_common_flow(),
            "flow_variations": self._get_flow_variations(),
            "generated_at": datetime.now().isoformat()
        }
    
    # ========================================
    # 4. HISTORIAL DE ANOMALÍAS
    # ========================================
    
    def log_flow_anomaly(self, tester_id: str, session_key: str, 
                        deviation_details: Dict, severity: str = "medium"):
        """
        Registra una anomalía detectada en la BD.
        Permite tracking histórico de problemas.
        """
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            # Crear tabla si no existe
            c.execute("""
                CREATE TABLE IF NOT EXISTS flow_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name TEXT NOT NULL,
                    tester_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    deviation_point INTEGER,
                    expected_screen TEXT,
                    actual_screen TEXT,
                    severity TEXT DEFAULT 'medium',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved INTEGER DEFAULT 0
                )
            """)
            
            c.execute("""
                INSERT INTO flow_anomalies 
                (app_name, tester_id, session_key, deviation_point, 
                 expected_screen, actual_screen, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                self.app_name,
                tester_id,
                session_key,
                deviation_details.get("step_number"),
                deviation_details.get("expected"),
                deviation_details.get("actual"),
                severity
            ))
            
            conn.commit()
            logger.info(f"📍 Anomalía registrada: {tester_id} en {session_key}")
    
    def get_anomaly_history(self, tester_id: str = None, days: int = 30) -> List[Dict]:
        """
        Obtiene historial de anomalías detectadas.
        """
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            query = """
                SELECT id, tester_id, session_key, deviation_point,
                       expected_screen, actual_screen, severity, created_at
                FROM flow_anomalies
                WHERE app_name = ? AND created_at > datetime('now', '-' || ? || ' days')
            """
            params = [self.app_name, days]
            
            if tester_id:
                query += " AND tester_id = ?"
                params.append(tester_id)
            
            query += " ORDER BY created_at DESC"
            
            c.execute(query, params)
            rows = c.fetchall()
            
            return [
                {
                    "id": r[0],
                    "tester_id": r[1],
                    "session_key": r[2],
                    "deviation_point": r[3],
                    "expected_screen": r[4],
                    "actual_screen": r[5],
                    "severity": r[6],
                    "created_at": r[7]
                }
                for r in rows
            ]
    
    # ========================================
    # 5. HELPERS PRIVADOS
    # ========================================
    
    def _get_session_flow(self, session_key: str) -> List[str]:
        """Obtiene secuencia de pantallas de una sesión."""
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT header_text
                FROM accessibility_data
                WHERE session_key = ?
                ORDER BY created_at ASC
            """, (session_key,))
            
            rows = c.fetchall()
            return [r[0] for r in rows if r[0]]
    
    def _get_most_common_flow(self) -> List[str]:
        """Obtiene flujo más frecuente (camino esperado)."""
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            # Obtener todas las sesiones
            c.execute("""
                SELECT session_key
                FROM accessibility_data
                WHERE app_name = ?
                GROUP BY session_key
                ORDER BY COUNT(*) DESC
                LIMIT 100
            """, (self.app_name,))
            
            sessions = c.fetchall()
        
        # Buscar la secuencia más común
        flow_counter = Counter()
        for (session_key,) in sessions:
            flow = tuple(self._get_session_flow(session_key))
            flow_counter[flow] += 1
        
        if flow_counter:
            most_common = flow_counter.most_common(1)[0][0]
            return list(most_common)
        
        return []
    
    def _calculate_flow_similarity(self, flow1: List[str], flow2: List[str]) -> float:
        """Calcula similitud entre dos flujos (0.0-1.0)."""
        if not flow1 or not flow2:
            return 0.0
        
        # Algoritmo simple: porcentaje de pantallas coincidentes en orden
        matches = sum(1 for a, b in zip(flow1, flow2) if a == b)
        max_len = max(len(flow1), len(flow2))
        
        return min(matches / max_len, 1.0) if max_len > 0 else 0.0
    
    def _find_recovery_path(self, current_screen: str, expected_screen: str, 
                           remaining_flow: List[str]) -> List[str]:
        """Sugiere cómo volver del camino anómalo."""
        # Buscar puntos de reconexión
        recovery = []
        for screen in remaining_flow:
            recovery.append(screen)
            if screen == expected_screen:
                break
        
        return recovery[:5]  # Limitar a 5 pasos sugeridos
    
    def _generate_suggestions(self, is_deviated: bool, similarity: float, 
                            flow: List[str]) -> List[str]:
        """Genera sugerencias basadas en el análisis."""
        suggestions = []
        
        if not is_deviated:
            suggestions.append("✅ Flujo correcto: sigue adelante con confianza")
        elif similarity > 0.8:
            suggestions.append("⚠️ Leve desviación: intenta volver al flujo principal")
            suggestions.append("💡 El siguiente paso esperado está disponible")
        elif similarity > 0.5:
            suggestions.append("❌ Desviación moderada: revisa los pasos anteriores")
            suggestions.append("🔄 Considera reiniciar desde la pantalla anterior")
        else:
            suggestions.append("❌ Flujo completamente desviado")
            suggestions.append("🏠 Regresa a la pantalla de inicio e intenta de nuevo")
        
        if len(flow) > 10:
            suggestions.append("⏱️ Sesión larga: considera dividir en pasos más pequeños")
        
        return suggestions
    
    def _get_tester_frequent_flows(self, tester_id: str, days: int, 
                                  limit: int = 5) -> List[Dict]:
        """Obtiene flujos más frecuentes de un tester."""
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            c.execute("""
                SELECT DISTINCT session_key
                FROM accessibility_data
                WHERE tester_id = ? AND app_name = ?
                  AND created_at > datetime('now', '-' || ? || ' days')
            """, (tester_id, self.app_name, days))
            
            sessions = c.fetchall()
        
        flow_counter = Counter()
        for (session_key,) in sessions:
            flow = tuple(self._get_session_flow(session_key))
            flow_counter[flow] += 1
        
        result = []
        for flow, count in flow_counter.most_common(limit):
            result.append({
                "flow": list(flow),
                "frequency": count,
                "percentage": round(count / len(sessions) * 100, 1) if sessions else 0
            })
        
        return result
    
    def _calculate_interruption_points(self) -> Dict[str, int]:
        """Identifica pantallas donde se interrumpen flujos."""
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            c.execute("""
                SELECT header_text, COUNT(*) as interruptions
                FROM (
                    SELECT DISTINCT session_key, header_text
                    FROM accessibility_data
                    WHERE app_name = ?
                    GROUP BY session_key
                    HAVING COUNT(*) > 1
                )
                GROUP BY header_text
                ORDER BY interruptions DESC
                LIMIT 10
            """, (self.app_name,))
            
            rows = c.fetchall()
            return {screen: count for screen, count in rows}
    
    def _get_flow_variations(self) -> List[Dict]:
        """Obtiene variaciones de flujos (caminos alternativos válidos)."""
        with sqlite3.connect(self.db) as conn:
            c = conn.cursor()
            
            c.execute("""
                SELECT session_key
                FROM accessibility_data
                WHERE app_name = ?
                GROUP BY session_key
            """, (self.app_name,))
            
            sessions = c.fetchall()
        
        flow_counter = Counter()
        for (session_key,) in sessions:
            flow = tuple(self._get_session_flow(session_key))
            flow_counter[flow] += 1
        
        result = []
        for flow, count in flow_counter.most_common(5):
            result.append({
                "flow": list(flow),
                "frequency": count
            })
        
        return result
    
    def _generate_recommendations(self, anomaly_count: int, total_sessions: int,
                                 frequent_flows: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en el análisis."""
        recommendations = []
        
        if total_sessions == 0:
            return ["Insuficientes datos para hacer recomendaciones"]
        
        anomaly_rate = anomaly_count / total_sessions
        
        if anomaly_rate > 0.5:
            recommendations.append(
                "⚠️ CRÍTICO: Más del 50% de sesiones tienen deviaciones. "
                "Revisa el flujo esperado o la interfaz de la app."
            )
        elif anomaly_rate > 0.2:
            recommendations.append(
                "🔍 Múltiples deviaciones detectadas. Algunas pantallas pueden ser confusas."
            )
        else:
            recommendations.append(
                "✅ Flujos generalmente correctos. Excelente trabajo de testing."
            )
        
        if frequent_flows:
            recommendations.append(
                f"📊 Flujo más común: {' → '.join(frequent_flows[0]['flow'][:3])}"
            )
        
        return recommendations


# ========================================
# ENDPOINT HELPER
# ========================================

def create_flow_analytics_for_app(app_name: str) -> FlowAnalyticsEngine:
    """Factory function."""
    return FlowAnalyticsEngine(app_name)
