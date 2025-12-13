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

import logging
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional
from db import get_conn_cm

logger = logging.getLogger(__name__)
FLOW_MODEL_DIR = "models/flows"
# DB_NAME = "accessibility.db"

class FlowAnalyticsEngine:
    """
    Motor de análisis de flujos con PostgreSQL:
    - Detección de desviaciones
    - Generación de reportes por tester
    - Historial de anomalías
    - Sugerencias de flujos correctos
    """

    def __init__(self, app_name: str):
        self.app_name = app_name

    # ========================================
    # 1. ANÁLISIS DE DESVIACIONES
    # ========================================
    def analyze_deviation(self, session_key: str, expected_flow: List[str]) -> Dict:
        observed_flow = self._get_session_flow(session_key)
        if not observed_flow:
            return {
                "session_key": session_key,
                "error": "No flow data found",
                "is_deviated": None
            }

        deviation_point = None
        for i, (expected, actual) in enumerate(zip(expected_flow, observed_flow)):
            if expected != actual:
                deviation_point = i
                break

        is_deviated = deviation_point is not None
        similarity = self._calculate_flow_similarity(expected_flow, observed_flow)

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
    # def generate_tester_flow_report(self, tester_id: str, days: int = 7) -> Dict:
    #     # conn = get_conn()
    #     try:
    #         # cur = conn.cursor()
    #         with get_conn_cm() as conn:
	#             with conn.cursor() as c:
            
    #                     c.execute("""
    #                         SELECT DISTINCT session_key, MIN(created_at)
    #                         FROM accessibility_data
    #                         WHERE tester_id = %s AND app_name = %s
    #                         AND created_at > NOW() - INTERVAL '%s days'
    #                         GROUP BY session_key
    #                         ORDER BY MIN(created_at) DESC
    #                     """, (tester_id, self.app_name, days))
    #                     sessions = c.fetchall()
    #     # finally:
    #     #     release_conn(conn)

    #     except Exception as e:
    #         logger.error(f"❌ Error generate_tester_flow_report: {e}")

    #     if not sessions:
    #         return {
    #             "tester_id": tester_id,
    #             "app_name": self.app_name,
    #             "period_days": days,
    #             "total_sessions": 0,
    #             "message": "No sessions found"
    #         }

    #     session_analyses = []
    #     anomaly_count = 0
    #     for session_key, _ in sessions:
    #         expected = self._get_most_common_flow()
    #         analysis = self.analyze_deviation(session_key, expected)
    #         session_analyses.append(analysis)
    #         if analysis.get("is_deviated"):
    #             anomaly_count += 1

    #     avg_similarity = sum(s.get("similarity_score", 1.0) for s in session_analyses) / max(1, len(session_analyses))
    #     frequent_flows = self._get_tester_frequent_flows(tester_id, days, limit=5)

    #     return {
    #         "tester_id": tester_id,
    #         "app_name": self.app_name,
    #         "period_days": days,
    #         "total_sessions": len(sessions),
    #         "anomalous_sessions": anomaly_count,
    #         "anomaly_rate": round(anomaly_count / len(sessions), 3),
    #         "avg_flow_similarity": round(avg_similarity, 3),
    #         "quality_score": round(avg_similarity * 100, 1),
    #         "session_analyses": session_analyses[:10],
    #         "frequent_flows": frequent_flows,
    #         "recommendations": self._generate_recommendations(anomaly_count, len(sessions), frequent_flows),
    #         "generated_at": datetime.now().isoformat()
    #     }
    
    def generate_tester_flow_report(self, tester_id: str, days: int = 7) -> Dict:
        sessions = []  # <-- garantiza que siempre exista

        try:
            with get_conn_cm() as conn:
                with conn.cursor() as c:
                    c.execute("""
                        SELECT DISTINCT session_key, MIN(created_at)
                        FROM accessibility_data
                        WHERE tester_id = %s AND app_name = %s
                        AND created_at > NOW() - INTERVAL %s
                        GROUP BY session_key
                        ORDER BY MIN(created_at) DESC
                    """, (tester_id, self.app_name, f"{days} days"))

                    sessions = c.fetchall()

        except Exception as e:
            logger.error(f"❌ Error generate_tester_flow_report: {e}")
            return {
                "tester_id": tester_id,
                "app_name": self.app_name,
                "period_days": days,
                "total_sessions": 0,
                "message": "Query failed"
            }

        # Si no hay sesiones
        if not sessions:
            return {
                "tester_id": tester_id,
                "app_name": self.app_name,
                "period_days": days,
                "total_sessions": 0,
                "message": "No sessions found"
            }

        session_analyses = []
        anomaly_count = 0

        for session_key, _ in sessions:
            expected = self._get_most_common_flow()
            analysis = self.analyze_deviation(session_key, expected)
            session_analyses.append(analysis)
            if analysis.get("is_deviated"):
                anomaly_count += 1

        avg_similarity = sum(s.get("similarity_score", 1.0) for s in session_analyses) / max(1, len(session_analyses))
        frequent_flows = self._get_tester_frequent_flows(tester_id, days, limit=5)

        return {
            "tester_id": tester_id,
            "app_name": self.app_name,
            "period_days": days,
            "total_sessions": len(sessions),
            "anomalous_sessions": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(sessions), 3),
            "avg_flow_similarity": round(avg_similarity, 3),
            "quality_score": round(avg_similarity * 100, 1),
            "session_analyses": session_analyses[:10],
            "frequent_flows": frequent_flows,
            "recommendations": self._generate_recommendations(anomaly_count, len(sessions), frequent_flows),
            "generated_at": datetime.now().isoformat()
        }


    # ========================================
    # 3. DASHBOARD DE FLUJOS
    # ========================================
    def get_flow_analytics_dashboard(self, include_testers: Optional[List[str]] = None) -> Dict:
        # conn = get_conn()
        try:
            # cur = conn.cursor()
            with get_conn_cm() as conn:
	            with conn.cursor() as c:
                        query = "SELECT COUNT(DISTINCT session_key) FROM accessibility_data WHERE app_name = %s"
                        params = [self.app_name]
                        if include_testers:
                            query += f" AND tester_id IN ({','.join(['%s']*len(include_testers))})"
                            params.extend(include_testers)
                        c.execute(query, params)
                        total_sessions = c.fetchone()[0]

                        c.execute("""
                            SELECT header_text, COUNT(*) as count
                            FROM accessibility_data
                            WHERE app_name = %s
                            GROUP BY header_text
                            ORDER BY count DESC
                            LIMIT 10
                        """, (self.app_name,))
                        screen_distribution = c.fetchall()
        # finally:
        #     release_conn(conn)
        except Exception as e:
            logger.error(f"❌ Error flow_analytics_dashboard: {e}")
        
        interruption_points = self._calculate_interruption_points()

        return {
            "app_name": self.app_name,
            "total_sessions": total_sessions,
            "screen_distribution": {screen: count for screen, count in screen_distribution},
            "interruption_hotspots": interruption_points,
            "most_common_flow": self._get_most_common_flow(),
            "flow_variations": self._get_flow_variations(),
            "generated_at": datetime.now().isoformat()
        }

    # ========================================
    # 4. HISTORIAL DE ANOMALÍAS
    # ========================================
    def log_flow_anomaly(self, tester_id: str, session_key: str, deviation_details: Dict, severity: str = "medium"):
        # conn = get_conn()
        try:
            # cur = conn.cursor()
            with get_conn_cm() as conn:
	            with conn.cursor() as c:
                        c.execute("""
                            CREATE TABLE IF NOT EXISTS flow_anomalies (
                                id SERIAL PRIMARY KEY,
                                app_name TEXT NOT NULL,
                                tester_id TEXT NOT NULL,
                                session_key TEXT NOT NULL,
                                deviation_point INT,
                                expected_screen TEXT,
                                actual_screen TEXT,
                                severity TEXT DEFAULT 'medium',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                resolved BOOLEAN DEFAULT FALSE
                            )
                        """)
                        c.execute("""
                            INSERT INTO flow_anomalies
                            (app_name, tester_id, session_key, deviation_point, expected_screen, actual_screen, severity)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
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
        except Exception as e:
            logger.error(f"❌ Error log_flow_anomaly: {e}")
        # finally:
        #     release_conn(conn)

    def get_anomaly_history(self, tester_id: str = None, days: int = 30) -> List[Dict]:
        # conn = get_conn()
        try:
            with get_conn_cm() as conn:
	            with conn.cursor() as c:
            # cur = conn.cursor()
                        query = """
                            SELECT id, tester_id, session_key, deviation_point,
                                expected_screen, actual_screen, severity, created_at
                            FROM flow_anomalies
                            WHERE app_name = %s AND created_at > NOW() - INTERVAL '%s days'
                        """
                        params = [self.app_name, days]
                        if tester_id:
                            query += " AND tester_id = %s"
                            params.append(tester_id)
                        query += " ORDER BY created_at DESC"
                        c.execute(query, params)
                        rows = c.fetchall()
                        return [
                            {
                                "id": r[0], "tester_id": r[1], "session_key": r[2],
                                "deviation_point": r[3], "expected_screen": r[4], "actual_screen": r[5],
                                "severity": r[6], "created_at": r[7]
                            } for r in rows
                        ]
                
        except Exception as e:
            logger.error(f"❌ Error get_anomaly_history: {e}")
        # finally:
        #     release_conn(conn)

    # ========================================
    # 5. HELPERS PRIVADOS
    # ========================================
    def _get_session_flow(self, session_key: str) -> List[str]:
        try:
            with get_conn_cm() as conn:
                with conn.cursor() as c:
                    c.execute("""
                        SELECT header_text
                        FROM accessibility_data
                        WHERE session_key = %s
                        ORDER BY created_at ASC
                    """, (session_key,))
                    rows = c.fetchall()
                    return [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.error(f"❌ Error _get_session_flow: {e}")
            return []

    def _get_most_common_flow(self) -> List[str]:
        # conn = get_conn()
        flow_counter = Counter()
        try:
            # cur = conn.cursor()
            with get_conn_cm() as conn:
                with conn.cursor() as c:
                        c.execute("""
                            SELECT session_key
                            FROM accessibility_data
                            WHERE app_name = %s
                            GROUP BY session_key
                            ORDER BY COUNT(*) DESC
                            LIMIT 100
                        """, (self.app_name,))
                        sessions = c.fetchall()
                        for (session_key,) in sessions:
                            flow = tuple(self._get_session_flow(session_key))
                            flow_counter[flow] += 1
        # finally:
        #     release_conn(conn)

        except Exception as e:
            logger.error(f"❌ Error _get_most_common_flow: {e}")

        return list(flow_counter.most_common(1)[0][0]) if flow_counter else []

    def _calculate_flow_similarity(self, flow1: List[str], flow2: List[str]) -> float:
        if not flow1 or not flow2:
            return 0.0
        matches = sum(1 for a, b in zip(flow1, flow2) if a == b)
        max_len = max(len(flow1), len(flow2))
        return min(matches / max_len, 1.0) if max_len > 0 else 0.0

    def _find_recovery_path(self, current_screen: str, expected_screen: str, remaining_flow: List[str]) -> List[str]:
        recovery = []
        for screen in remaining_flow:
            recovery.append(screen)
            if screen == expected_screen:
                break
        return recovery[:5]

    def _generate_suggestions(self, is_deviated: bool, similarity: float, flow: List[str]) -> List[str]:
        suggestions = []
        if not is_deviated:
            suggestions.append("✅ Flujo correcto: sigue adelante con confianza")
        elif similarity > 0.8:
            suggestions.append("⚠️ Leve desviación: intenta volver al flujo principal")
        elif similarity > 0.5:
            suggestions.append("❌ Desviación moderada: revisa los pasos anteriores")
        else:
            suggestions.append("❌ Flujo completamente desviado")
        if len(flow) > 10:
            suggestions.append("⏱️ Sesión larga: considera dividir en pasos más pequeños")
        return suggestions

    def _get_tester_frequent_flows(self, tester_id: str, days: int, limit: int = 5) -> List[Dict]:
        
        flow_counter = Counter()
        
        with get_conn_cm() as conn:
            with conn.cursor() as c:   
        
                    try:
                        cur = conn.cursor()
                        cur.execute("""
                            SELECT DISTINCT session_key
                            FROM accessibility_data
                            WHERE tester_id = %s AND app_name = %s
                            AND created_at > NOW() - INTERVAL '%s days'
                        """, (tester_id, self.app_name, days))
                        sessions = cur.fetchall()
                        for (session_key,) in sessions:
                            flow = tuple(self._get_session_flow(session_key))
                            flow_counter[flow] += 1

                    except Exception as e:
                        logger.error(f"❌ Error _get_most_common_flow: {e}")        

        # finally:
        #     release_conn(conn)

        result = []
        for flow, count in flow_counter.most_common(limit):
            result.append({
                "flow": list(flow),
                "frequency": count,
                "percentage": round(count / len(sessions) * 100, 1) if sessions else 0
            })

        return result

    def _calculate_interruption_points(self) -> Dict[str, int]:
        # conn = get_conn()
        try:
            # cur = conn.cursor()
            with get_conn_cm() as conn:
                with conn.cursor() as c:

                        c.execute("""
                            SELECT header_text, COUNT(*) as interruptions
                            FROM (
                                SELECT DISTINCT session_key, header_text
                                FROM accessibility_data
                                WHERE app_name = %s
                                GROUP BY session_key
                                HAVING COUNT(*) > 1
                            ) t
                            GROUP BY header_text
                            ORDER BY interruptions DESC
                            LIMIT 10
                        """, (self.app_name,))
                        rows = c.fetchall()
                        return {screen: count for screen, count in rows}
        # finally:
        #     release_conn(conn)
        except Exception as e:
            logger.error(f"❌ Error _calculate_interruption_points: {e}")   

    def _get_flow_variations(self) -> List[Dict]:

        flow_counter = Counter()

        with get_conn_cm() as conn:
            with conn.cursor() as c:
        
                    try:
                        cur = conn.cursor()
                        cur.execute("""
                            SELECT session_key
                            FROM accessibility_data
                            WHERE app_name = %s
                            GROUP BY session_key
                        """, (self.app_name,))
                        sessions = cur.fetchall()
                        for (session_key,) in sessions:
                            flow = tuple(self._get_session_flow(session_key))
                            flow_counter[flow] += 1
                    
                    except Exception as e:
                        logger.error(f"❌ Error _calculate_interruption_points: {e}")
        # finally:
        #     release_conn(conn)
        result = []
        for flow, count in flow_counter.most_common(5):
            result.append({"flow": list(flow), "frequency": count})
        return result

    def _generate_recommendations(self, anomaly_count: int, total_sessions: int, frequent_flows: List[Dict]) -> List[str]:
        recommendations = []
        if total_sessions == 0:
            return ["Insuficientes datos para hacer recomendaciones"]
        anomaly_rate = anomaly_count / total_sessions
        if anomaly_rate > 0.5:
            recommendations.append("⚠️ CRÍTICO: Más del 50% de sesiones tienen desviaciones.")
        elif anomaly_rate > 0.2:
            recommendations.append("🔍 Múltiples desviaciones detectadas. Revisar pantallas confusas.")
        else:
            recommendations.append("✅ Flujos generalmente correctos.")
        if frequent_flows:
            recommendations.append(f"📊 Flujo más común: {' → '.join(frequent_flows[0]['flow'][:3])}")
        return recommendations

# ========================================
# FACTORY
# ========================================
def create_flow_analytics_for_app(app_name: str) -> FlowAnalyticsEngine:
    return FlowAnalyticsEngine(app_name)
