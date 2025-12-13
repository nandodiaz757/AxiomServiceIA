"""
🤖 QA AI DASHBOARD - Análisis Inteligente de Cambios UI
Proporciona insights basados en IA para toma de decisiones en testing
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import hashlib
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
import math
from db import get_conn_cm
from psycopg2.extras import RealDictCursor
import plotly.graph_objs as go
from plotly.subplots import make_subplots

qa_ai_router = APIRouter(prefix="/api/qa-ai", tags=["QA AI"])

# =========================================================================
# 🧠 ANÁLISIS INTELIGENTE DE CAMBIOS
# =========================================================================

class ChangeAnalyzer:
    """Analiza cambios entre versiones con inteligencia artificial"""
    


    @staticmethod
    def calculate_stability_score(
        removed_count: int,
        added_count: int,
        modified_count: int,
        text_changes: int = 0
    ) -> float:
        """
        Calcula la estabilidad de una pantalla según los cambios estructurales y de texto.

        - Cada cambio estructural reduce 25 puntos de estabilidad.
        - Si no hay cambios estructurales pero hay cambios de texto, estabilidad = 90.
        - Si no hay cambios, estabilidad = 100.
        """

        # Contar solo cambios estructurales
        structural_changes = int(removed_count + added_count + modified_count)

        if structural_changes > 0:
            # Cada cambio estructural reduce 25 puntos
            score = 100 - structural_changes * 25
            score = max(0, score)  # mínimo 0
        elif text_changes > 0:
            score = 90  # cambios solo de texto → alta estabilidad
        else:
            score = 100  # sin cambios → máxima estabilidad

        return round(float(score), 2)



    @staticmethod
    def calculate_risk_score(
        stability: float,
        frequency: float,
        modification_intensity: float,
        historical_failures: int = 0,
        text_changes: int = 0
    ) -> Dict:
        W_STABILITY = 0.4
        W_FREQUENCY = 0.3
        W_INTENSITY = 0.2
        W_HISTORY = 0.1

        stability_risk = (100 - stability) * W_STABILITY
        frequency_risk = min(frequency / 10, 1.0) * 100 * W_FREQUENCY
        intensity_risk = min(modification_intensity, 1.0) * 100 * W_INTENSITY
        historical_risk = min(historical_failures / 5, 1.0) * 100 * W_HISTORY

        # 🔹 Solo texto → máximo 5 puntos
        if modification_intensity == 0:
            total_risk = 5
        else:
            total_risk = stability_risk + frequency_risk + intensity_risk + historical_risk

        return {
            "total_risk": round(total_risk, 2),
            "stability_factor": round(stability_risk, 2),
            "frequency_factor": round(frequency_risk, 2),
            "intensity_factor": round(intensity_risk, 2),
            "historical_factor": round(historical_risk, 2)
        }


    @staticmethod
    def predict_failure_probability(
        risk_score: float,
        change_magnitude: float,
        similar_past_issues: int = 0
    ) -> Dict:
        """
        Predice probabilidad de fallos futuros (0-100%)
        Penaliza menos cuando solo hay cambios textuales
        """

        base_probability = risk_score / 100.0

        # Si no hay intensidad → cambios únicamente textuales
        if change_magnitude == 0:
            change_factor = 0  # impacto funcional mínimo
        else:
            change_factor = min(change_magnitude, 1.0)

        history_factor = 1 - math.exp(-max(similar_past_issues, 0))

        failure_prob = (
            base_probability * 0.55 +
            change_factor * 0.25 +
            history_factor * 0.20
        ) * 100

        confidence = min(50 + similar_past_issues * 10, 95)

        return {
            "failure_probability": round(min(failure_prob, 100.0), 2),
            "confidence": confidence,
            "recommendation": ChangeAnalyzer._get_risk_recommendation(failure_prob)
        }

    
    @staticmethod
    def _get_risk_recommendation(probability: float) -> str:
        """Retorna recomendación basada en probabilidad de fallo"""
        if probability >= 80:
            return "🔴 CRÍTICO: Requiere testing exhaustivo inmediato"
        elif probability >= 60:
            return "🟠 ALTO: Testing intensivo recomendado"
        elif probability >= 40:
            return "🟡 MEDIO: Testing estándar + casos edge"
        elif probability >= 20:
            return "🟢 BAJO: Testing estándar suficiente"
        else:
            return "✅ MÍNIMO: Cambios menores, testing básico"
    
    

    @staticmethod
    def calculate_change_impact(removed, added, modified) -> Dict:
        """
        Impacto UI basado en severidad del componente y cantidad.
        Cambios solo de texto (TextView) que ya estén marcados en modified_texts no se cuentan.
        """
        weights = {
            "android.widget.Button": 3,
            "android.widget.EditText": 3,
            "android.widget.ImageButton": 3,
            "android.widget.ImageView": 2,
            "android.widget.TextView": 1
        }
        
        total_score = 0
        areas = defaultdict(int)

        # Solo contar nodos estructurales (no text_changes)
        for node in removed + added:
            c = node.get("node", node).get("class", "unknown")
            if c != "android.widget.TextView":  # ignorar solo texto
                total_score += weights.get(c, 1)
                areas[c] += weights.get(c, 1)

        for change in modified:
            c = change.get("node", {}).get("class", "unknown")
            if c != "android.widget.TextView":  # ignorar solo texto
                total_score += weights.get(c, 1)
                areas[c] += weights.get(c, 1)

        # Clasificación mucho más útil para QA
        if total_score >= 20:
            level, severity = "CRITICAL", 3
        elif total_score >= 10:
            level, severity = "HIGH", 2
        else:
            level, severity = "LOW", 1

        return {
            "impact_level": level,
            "severity": severity,
            "impact_score": total_score,
            "areas_affected": sorted(areas.items(), key=lambda x: x[1], reverse=True),
        }


    @staticmethod
    def find_trending_issues(history: List[Dict]) -> Dict:
        patterns = defaultdict(list)

        for diff in history:
            screen = diff.get("screen")
            timestamp = diff.get("created_at")

            for n in diff.get("removed", []) + diff.get("added", []):
                comp = n.get("node", {}).get("class", "unknown")
                key = f"{screen}::{comp}"
                patterns[key].append(timestamp)

            for change in diff.get("modified", []):
                comp = change.get("node", {}).get("class", "unknown")
                key = f"{screen}::{comp}"
                patterns[key].append(timestamp)

        trending = {}
        now = max((d.get("created_at") for d in history), default=None)

        for key, times in patterns.items():
            if len(times) >= 3:
                recent = sum(1 for t in times if (now - t).days <= 30)
                trending[key] = {
                    "total_occurrences": len(times),
                    "recent_occurrences": recent,
                    "last_change": max(times),
                    "risk": "High" if recent >= 2 else "Medium"
                }

        return dict(sorted(trending.items(), key=lambda x: x[1]["recent_occurrences"], reverse=True))
   

from typing import List, Dict

class MetricsCalculator:
    """Métricas inteligentes para el dashboard"""

    @staticmethod
    def calculate_test_coverage_gap(last_build_diffs: List[Dict]) -> Dict:
        total_screens = len(last_build_diffs)

        high_risk_untested = sum(
            1 for d in last_build_diffs
            if d["risk_score"] >= 40 and not d.get("approval")
        )

        tested_screens = total_screens - high_risk_untested
        coverage_pct = (tested_screens / total_screens * 100) if total_screens > 0 else 0

        return {
            "total_coverage": round(coverage_pct, 1),
            "untested_screens": high_risk_untested,
            "critical_gap": high_risk_untested > 0,
            "priority": "🔴 CRÍTICO" if high_risk_untested > 0 else "🟢 OK"
        }

    @staticmethod
    def calculate_regression_risk(
        last_build_diffs: List[Dict],
        historical_diffs: List[List[Dict]],
        lookback: int = 5
    ) -> Dict:
        similar_risk_repeats = 0

        for history in historical_diffs[:lookback]:
            for diff in last_build_diffs:
                match = next(
                    (h for h in history
                     if h["screen"] == diff["screen"]
                     and abs(h["risk_score"] - diff["risk_score"]) < 15),
                    None
                )
                if match:
                    similar_risk_repeats += 1

        total_possible = len(last_build_diffs) * min(len(historical_diffs), lookback)
        regression_pct = (similar_risk_repeats / total_possible * 100) if total_possible > 0 else 0

        return {
            "regression_risk": round(regression_pct, 1),
            "repeated_patterns": similar_risk_repeats,
            "recommendation": "⚠️ Riesgo de regresión" if regression_pct > 50 else "Normal"
        }

    @staticmethod
    def find_trending_issues(history: List[Dict]) -> Dict:
        patterns = defaultdict(list)

        for diff in history:
            screen = diff.get("screen")
            timestamp = diff.get("created_at")

            for n in diff.get("removed", []) + diff.get("added", []):
                comp = n.get("node", {}).get("class", "unknown")
                key = f"{screen}::{comp}"
                patterns[key].append(timestamp)

            for change in diff.get("modified", []):
                comp = change.get("node", {}).get("class", "unknown")
                key = f"{screen}::{comp}"
                patterns[key].append(timestamp)

        trending = {}
        now = max((d.get("created_at") for d in history), default=None)

        for key, times in patterns.items():
            if len(times) >= 3:
                recent = sum(1 for t in times if (now - t).days <= 30)
                trending[key] = {
                    "total_occurrences": len(times),
                    "recent_occurrences": recent,
                    "last_change": max(times),
                    "risk": "High" if recent >= 2 else "Medium"
                }

        return dict(sorted(trending.items(), key=lambda x: x[1]["recent_occurrences"], reverse=True))

    @staticmethod
    def _get_resource_level(hours: float) -> str:
        if hours <= 4:
            return "👤 1 Tester (rápida verificación, 1-4h)"
        elif hours <= 16:
            return "👥 2 Testers (smoke + regresión parcial, 4-16h)"
        elif hours <= 40:
            return "👥👥 3-4 Testers (regresión completa, 1-5 días)"
        else:
            return "👥👥👥 5+ Testers + Team Lead (release crítico / coordinación, >5 días)"



class MetricsCalculator:

    @staticmethod
    def get_resource_suggestion(
        failure_probability: float,
        impact_score: float,
        severity: int,
        confidence: float,
        structural_changes: int = 1,
        text_changes: int = 0
    ) -> dict:
        """
        Asigna horas y recursos según nivel de riesgo, impacto, severidad y
        tipo de cambio (solo texto = esfuerzo mínimo).
        """

        # 🕐 Estimar horas base
        base_hours = (
            (failure_probability / 100) * 30 +
            (impact_score / 10) * 15 +
            severity * 3
        )

        # 🟡 Solo texto → esfuerzo MUY bajo
        if structural_changes == 0:
            # reducción progresiva según cantidad de textos
            reduction_factor = min(1.0, 0.10 + (text_changes * 0.04))
            base_hours *= reduction_factor

            # nunca más de 6h cuando es solo texto
            base_hours = min(base_hours, 6.0)

            # nunca menos de 1h (siempre QA touch)
            base_hours = max(base_hours, 1.0)

        estimated_hours = round(base_hours, 1)

        # 📊 Selección de recursos según horas
        if estimated_hours > 40:
            resource_label = "👥👥👥 5+ Testers + Team Lead (release crítico / coordinación, >5 días)"
        elif estimated_hours > 24:
            resource_label = "👥👥 3-4 Testers (2-3 días)"
        elif estimated_hours > 8:
            resource_label = "👥 2 Testers (1-2 días)"
        elif estimated_hours > 3:
            resource_label = "👤 1 Tester (medio día - 1 día)"
        else:
            resource_label = "⚪ QA rápido: 1 Tester (≤3h)"

        return {
            "estimated_hours": estimated_hours,
            "resource_label": resource_label
        }


    @staticmethod
    def estimate_required_hours(
        failure_probability: float,
        impact_score: float,
        severity: int,
        confidence: float = 50.0,
        structural_changes: int = 1,
        text_changes: int = 0
    ) -> float:
        """
        Estima horas necesarias en función de riesgo, impacto y severidad.
        - Si el cambio es SOLO textual ⇒ reducción drástica de esfuerzo
        """

        # 🧮 Fórmula base
        base_hours = (
            (failure_probability / 100) * 30 +   # probabilidad de fallos
            (impact_score / 10) * 15 +          # criticidad del cambio
            severity * 3                        # nivel de severidad definido por QA
        )

        # 🟦 Si NO hay cambios estructurales → solo texto
        if structural_changes == 0:
            # Reducción suave según cantidad de textos cambiados
            reduction_factor = min(1.0, 0.15 + (text_changes * 0.03))
            base_hours *= reduction_factor

            # Nunca puede pasar de 6h si solo es texto
            base_hours = min(base_hours, 6.0)

        # Limitación global
        return round(max(base_hours, 1.0), 1)


    @staticmethod
    def calculate_effort_estimate(last_build_diffs: List[Dict]) -> Dict:
        total_screens = len(last_build_diffs)
        total_changes = sum(
            len(d["removed"]) + len(d["added"]) + len(d["modified"])
            for d in last_build_diffs
        )
        avg_stability = sum(d["stability_score"] for d in last_build_diffs) / total_screens if total_screens > 0 else 100
        high_risk_count = sum(1 for d in last_build_diffs if d["risk_score"] >= 70)

        base_time = 25  # min/pantalla

        stability_m = (100 - avg_stability) / 100 * 1.5 + 1
        change_m = 1 + (min(total_changes, 30) / 30)
        risk_m = 1 + (high_risk_count / 5)

        hours = (base_time * stability_m * change_m * risk_m) / 60

        return {
            "estimated_hours": round(hours, 1),
            "estimated_days": round(hours / 8, 1),
            "test_cases_recommended": int(total_changes * 3 + high_risk_count * 4),
            "resource_level": MetricsCalculator._get_resource_level(hours)
        }




# =========================================================================
# 📊 DASHBOARD ENDPOINT
# =========================================================================

@qa_ai_router.get("/dashboard-advanced/{tester_id}", response_class=HTMLResponse)
def qa_ai_dashboard_advanced(
    tester_id: str,
    builds_to_compare: int = Query(5, ge=1, le=20),
    show_predictions: bool = Query(True)
):
    try:
        with get_conn_cm() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as c:

                # 1️⃣ Historial de builds
                c.execute("""
                    SELECT build_id, MAX(created_at) AS last_seen
                    FROM screen_diffs
                    WHERE tester_id = %s
                    GROUP BY build_id
                    ORDER BY last_seen DESC
                    LIMIT %s
                """, (tester_id, builds_to_compare))

                builds_rows = c.fetchall()
                builds = [row['build_id'] for row in builds_rows]

                if not builds:
                    return HTMLResponse("<h1>No hay datos para este tester</h1>")

                builds_data = {}

                # 2️⃣ Diffs por build
                for build_id in builds:
                    c.execute("""
                        SELECT id, screen_name, removed, added, modified,
                            created_at, anomaly_score, cluster_id, approval_status
                        FROM screen_diffs
                        WHERE tester_id = %s AND build_id = %s
                        ORDER BY created_at DESC
                    """, (tester_id, build_id))

                    diffs = []

                    for row in c.fetchall():
                        removed = json.loads(row['removed']) if row['removed'] else []
                        added   = json.loads(row['added']) if row['added'] else []
                        modified = json.loads(row['modified']) if row['modified'] else []

                        # --- Detectar nodos que SON SOLO TEXTO ---
                        def is_text_only(node, change_type="modified"):
                            cls = node.get("node", {}).get("class", "")
                            changes = node.get("changes", {})

                            # Solo los TextView modificados *solo en texto* no son estructurales
                            if cls != "android.widget.TextView":
                                return False  # no TextView → estructural

                            if change_type in ["removed", "added"]:
                                return False  # todo removed o added TextView es estructural

                            # change_type == "modified"
                            if not changes:
                                return False  # sin cambios → estructural
                            if list(changes.keys()) == ["text"]:
                                return True   # solo cambio de texto → no estructural

                            return False

                        # ✔ Filtrado: quitar nodos que son solo texto
                        # removed_nodes = [n for n in removed if not is_text_only(n, removed=True)]
                        # added_nodes   = [n for n in added if not is_text_only(n, added=True)]
                        # modified_nodes = [m for m in modified if not is_text_only(m)]

                        # Removed y Added siempre estructurales → no necesitan is_text_only
                        removed_nodes = [n for n in removed]  
                        added_nodes   = [n for n in added]    

                        # Modified → filtrar TextView que solo cambiaron texto
                        modified_nodes = [m for m in modified if not is_text_only(m, change_type="modified")]

                        # ✔ Contar cambios de texto REAL
                        modified_texts = [
                            m for m in modified_nodes
                            if "text" in m.get("changes", {}) 
                            and float(m["changes"].get("similarity", 1.0)) < 0.8
                        ]
                        text_changes = len(modified_texts)

                        # --- Cambios estructurales ---
                        def is_structural(node):
                            cls = node.get("node", {}).get("class", "")
                            changes = node.get("changes", {})

                            if cls == "android.widget.TextView" and (not changes or list(changes.keys()) == ["text"]):
                                return False
                            return True

                        structural_changes = sum(
                            1 for n in removed_nodes + added_nodes + modified_nodes if is_structural(n)
                        )

                        # ---- MÉTRICAS IA ---
                        stability = ChangeAnalyzer.calculate_stability_score(
                            removed_count=len(removed_nodes),
                            added_count=len(added_nodes),
                            modified_count=len(modified_nodes),
                            text_changes=text_changes
                        )

                        impact = ChangeAnalyzer.calculate_change_impact(
                            removed_nodes, added_nodes, modified_nodes
                        )

                        impact_score = impact["impact_score"]
                        impact_severity = impact["severity"]
                        impact_level = impact["impact_level"]

                        # Ajustes si solo texto
                        if structural_changes == 0:
                            impact_score = 0
                            impact_severity = 1
                            impact_level = "LOW"

                        modification_intensity = (
                            0 if structural_changes == 0 else min(impact_score / 10, 1.0)
                        )

                        # Riesgo
                        risk_factors = ChangeAnalyzer.calculate_risk_score(
                            stability=stability,
                            frequency=1,
                            modification_intensity=modification_intensity,
                            historical_failures=(
                                1 if structural_changes > 0 and row["approval_status"] == "rejected"
                                else 0
                            ),
                            text_changes=text_changes
                        )
                        risk_score = risk_factors["total_risk"]

                        # Predicción de fallos
                        prediction = ChangeAnalyzer.predict_failure_probability(
                            risk_score=risk_score,
                            change_magnitude=impact_score / 10,
                            similar_past_issues=1 if row['approval_status'] == "rejected" else 0
                        )
                        failure_prob = prediction["failure_probability"]
                        confidence = prediction["confidence"]
                        recommendation = prediction["recommendation"]

                        # --- FIX GLOBAL: sin cambios estructurales ni de texto ---
                        if structural_changes == 0 and text_changes == 0:
                            risk_score = 0
                            failure_prob = 0
                            impact_score = 0
                            impact_severity = 1
                            impact_level = "LOW"

                        # Recursos QA
                        resources = MetricsCalculator.get_resource_suggestion(
                            failure_probability=failure_prob,
                            impact_score=impact_score,
                            severity=impact_severity,
                            confidence=confidence,
                            structural_changes=structural_changes,
                            text_changes=text_changes
                        )

                        # Reducir horas si solo texto
                        if structural_changes == 0:
                            resources["estimated_hours"] = max(resources["estimated_hours"] * 0.20, 1.0)
                            resources["resource_label"] = "⚪ QA rápido: 1 Tester (≤2h)"

                        # Prioridad
                        if risk_score >= 70:
                            diff_priority = "high"
                        elif risk_score >= 40:
                            diff_priority = "medium"
                        else:
                            diff_priority = "low"

                        diffs.append({
                            "id": row['id'],
                            "screen": row['screen_name'],
                            "created_at": row['created_at'],
                            "stability_score": stability,
                            "impact_score": impact_score,
                            "impact_severity": impact_severity,
                            "impact_level": impact_level,
                            "risk_score": risk_score,
                            "failure_probability": failure_prob,
                            "confidence": confidence,
                            "recommendation": recommendation,
                            "testing_hours": resources["estimated_hours"],
                            "testing_resources": resources["resource_label"],
                            "anomaly_score": row['anomaly_score'],
                            "cluster_id": row['cluster_id'],
                            "approval": row['approval_status'],
                            "diff_priority": diff_priority,
                            "added_nodes": added_nodes,
                            "removed_nodes": removed_nodes,
                            "modified_nodes": modified_nodes,
                            "structural_changes": structural_changes,
                            "text_changes": text_changes,
                            "modified_texts": modified_texts
                        })

                    builds_data[build_id] = diffs

            # KPIs del último build
            last_build = builds[0]
            last_build_data = builds_data.get(last_build, [])

            total_screens = len(set(d["screen"] for d in last_build_data))
            high_risk_screens = sum(1 for d in last_build_data if d["risk_score"] >= 70)

            html_content = _generate_ai_dashboard_html_v6(
                tester_id=tester_id,
                builds=builds,
                builds_data=builds_data,
                total_screens=total_screens,
                high_risk_screens=high_risk_screens,
                show_predictions=show_predictions
            )

            return HTMLResponse(html_content)

    except Exception as e:
        print("🔥 Error en dashboard avanzado:", e)
        return HTMLResponse(f"<pre>Error: {str(e)}</pre>")



def _generate_ai_dashboard_html_v5(
    tester_id: str,
    builds: list,
    builds_data: dict,
    total_screens: int,
    high_risk_screens: int,
    show_predictions: bool = True
) -> str:

    priorities = {"high": 0, "medium": 0, "low": 0}
    clusters = {}
    total_hours = 0
    rows_html = ""
    seen_screens = set()

    # JS para toggle de detalles
    js_script = """
    <script>
    function toggleDetails(id) {
        const row = document.getElementById(id);
        row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
    }
    </script>
    """

    for build_id in builds:
        diffs = builds_data.get(build_id, [])
        if not diffs:
            continue

        # Fila de separación por build
        rows_html += f"""
        <tr style="background:#f0f0f0;font-weight:bold">
            <td colspan="10">Build: {build_id}</td>
        </tr>
        """

        # ✅ Contar priorities y clusters **antes de filtrar**
        for d in diffs:
            priorities[d["diff_priority"]] += 1
            cid = d.get("cluster_id")
            if cid:
                clusters[cid] = clusters.get(cid, 0) + 1

        for d in diffs:
            # Filtrar para mostrar solo una fila por pantalla+build
            screen_key = f"{d['screen']}_{build_id}"
            if screen_key in seen_screens:
                continue
            seen_screens.add(screen_key)

            total_hours += d.get("testing_hours", 0)
            badge_color = {"high": "#ff3b30", "medium": "#ff9500", "low": "#34c759"}[d["diff_priority"]]

            structural_changes = d.get("structural_changes", 0)
            text_changes = d.get("text_changes", 0)
            changes_summary = f"S:{structural_changes} / T:{text_changes}"

            # Crear fila de detalles ocultos
            detail_id = f"detail-{d['id']}"
            detail_html = f"""
            <tr id="{detail_id}" style="display:none;background:#f9f9f9;">
                <td colspan="10" style="text-align:left;">
                    <b>Added:</b> {json.dumps(d.get('added_nodes', []))}<br>
                    <b>Removed:</b> {json.dumps(d.get('removed_nodes', []))}<br>
                    <b>Modified:</b> {json.dumps(d.get('modified_nodes', []))}<br>
                    <b>Text Diff:</b> {json.dumps(d.get('modified_texts', []))}
                </td>
            </tr>
            """

            # Fila principal
            rows_html += f"""
            <tr>
                <td>
                    <a href="javascript:void(0);" onclick="toggleDetails('{detail_id}')">
                        {d['screen']}
                    </a>
                </td>
                <td>{d['stability_score']:.1f}</td>
                <td><span style="color:{badge_color};font-weight:bold">{d['risk_score']:.0f}</span></td>
                <td>{d['impact_level']}</td>
                <td>{d['failure_probability']:.1f}%</td>
                <td>{d['confidence']:.0f}%</td>
                <td>{d['testing_hours']:.1f} h</td>
                <td>{d['testing_resources']}</td>
                <td>{d['recommendation']}</td>
                <td title="Cantidad de cambios estructurales (S) y solo de texto (T)">{changes_summary}</td>
            </tr>
            {detail_html}
            """

    total_screens = len(seen_screens)
    high_risk_screens = priorities["high"]

    # Gráficos reflejando todos los diffs
    donut = go.Figure(data=[go.Pie(
        labels=["High", "Medium", "Low"],
        values=[priorities["high"], priorities["medium"], priorities["low"]],
        hole=.55
    )])
    donut.update_layout(width=380, height=300, margin=dict(t=10, b=10), title="Priority")
    donut_html = donut.to_html(include_plotlyjs="cdn", full_html=False)

    cluster_fig = go.Figure(data=[go.Bar(x=list(clusters.keys()), y=list(clusters.values()))])
    cluster_fig.update_layout(width=480, height=300, title="Clusters with the most changes")
    cluster_html = cluster_fig.to_html(include_plotlyjs=False, full_html=False)

    return f"""
    <html>
    <head>
        <title>QA AI Dashboard 👩‍💻🤖</title>
        <style>
            body {{ font-family: Arial; padding:20px; }}
            table {{ width:100%; border-collapse: collapse; margin-top:20px; }}
            th,td {{ border:1px solid #ccc; padding:6px; text-align:center; }}
            th {{ background:#eee }}
            .kpis {{ display:flex; gap:20px; margin-bottom:20px; }}
            .box {{ padding:12px 18px; background:#fafafa; border-radius:8px; font-size:17px; }}
            .charts {{ display:flex; gap:25px; margin:20px 0; }}
            a {{ cursor:pointer; text-decoration:none; color:#007bff; }}
        </style>
        {js_script}
    </head>
    <body>
        <h1>QA AI Dashboard v5</h1>
        <h3>Tester: {tester_id}</h3>

        <div class="kpis">
            <div class="box">📱 Screens: <b>{total_screens}</b></div>
            <div class="box">🔴 High Risk: <b>{high_risk_screens}</b></div>
            <div class="box">⏱ Total Effort: <b>{total_hours:.1f}h</b></div>
        </div>

        <div class="charts">
            {donut_html}
            {cluster_html}
        </div>

        <h3>Screen Risk & Testing Matrix</h3>
        <table>
            <tr>
                <th>Screen</th>
                <th>Stability</th>
                <th>Risk</th>
                <th>Impact</th>
                <th>Failure %</th>
                <th>Conf.</th>
                <th>Hours</th>
                <th>Resources</th>
                <th>Recommendation</th>
                <th>#Changes (S/T)</th>
            </tr>
            {rows_html}
        </table>
    </body>
    </html>
    """


def _generate_ai_dashboard_html(
    tester_id: str,
    builds: list,
    builds_data: dict,
    total_screens: int = None,
    high_risk_screens: int = None,
    show_predictions: bool = True,
    max_builds_display: int = 10
) -> str:
    """
    Genera un HTML avanzado y visual del QA Dashboard con IA.
    Parámetros:
        tester_id: ID del tester
        builds: lista de builds a mostrar
        builds_data: diccionario con info de cambios por build
        total_screens: opcional, total de pantallas
        high_risk_screens: opcional, total de pantallas de alto riesgo
        show_predictions: mostrar recomendaciones IA
        max_builds_display: máximo número de builds a mostrar en gráficos
    """
    import numpy as np
    import json
    from datetime import datetime

    # Ordenar y limitar builds
    builds_sorted = sorted(builds, reverse=True)[:max_builds_display]

    # Métricas por build
    metrics_by_build = {}
    all_diffs = []
    for build_id in builds_sorted:
        diffs = builds_data.get(build_id, [])
        all_diffs.extend(diffs)
        total_removed = sum(len(d["removed"]) for d in diffs)
        total_added = sum(len(d["added"]) for d in diffs)
        total_modified = sum(len(d["modified"]) for d in diffs)
        avg_risk = np.mean([d["risk_score"] for d in diffs]) if diffs else 0
        avg_stability = np.mean([d["stability_score"] for d in diffs]) if diffs else 100
        total_changes = total_removed + total_added + total_modified
        metrics_by_build[build_id] = {
            "removed": total_removed,
            "added": total_added,
            "modified": total_modified,
            "avg_risk": float(avg_risk),
            "avg_stability": float(avg_stability),
            "total_changes": total_changes,
            "total_screens": len(diffs)
        }

    # Calcular total screens y high risk si no se pasó
    if total_screens is None:
        total_screens = sum(m['total_screens'] for m in metrics_by_build.values())
    if high_risk_screens is None:
        high_risk_screens = len([d for d in all_diffs if d['risk_score'] > 60])

    # Top 10 pantallas críticas
    critical_screens = sorted(
        [(d["screen"], d["risk_score"], d["anomaly_score"]) for d in all_diffs if d["risk_score"] > 60],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Tendencias componentes problemáticos
    trending_issues = ChangeAnalyzer.find_trending_issues(all_diffs)[:8]

    # HTML ultra legible y responsive
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 QA AI Dashboard - {tester_id}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin:0; padding:10px; background:#f3f4f6; }}
            .container {{ max-width:1600px; margin:auto; }}
            .metrics-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:15px; margin-bottom:20px; }}
            .metric-card {{ background:white; padding:15px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size:28px; font-weight:bold; margin:5px 0; }}
            .metric-label {{ font-size:12px; color:#555; }}
            .metric-description {{ font-size:11px; color:#777; }}
            .chart-container {{ width:100%; overflow-x:auto; height:300px; }}
            canvas {{ min-width:300px; height:300px !important; }}
            table {{ width:100%; border-collapse:collapse; font-size:12px; }}
            th, td {{ padding:8px; border-bottom:1px solid #ddd; }}
            th {{ background:#f9fafb; }}
            .risk-badge {{ padding:3px 8px; border-radius:12px; font-size:11px; }}
            .risk-critical {{ background:#fee2e2; color:#991b1b; }}
            .risk-high {{ background:#fef3c7; color:#92400e; }}
            .risk-medium {{ background:#fce7f3; color:#831843; }}
            .risk-low {{ background:#dbeafe; color:#1e3a8a; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🤖 QA AI Dashboard - Tester: {tester_id}</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Average Risk</div>
                    <div class="metric-value">{np.mean([m['avg_risk'] for m in metrics_by_build.values()]):.1f}%</div>
                    <div class="metric-description">Estimated risk level</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">High Risk Screens</div>
                    <div class="metric-value">{high_risk_screens}</div>
                    <div class="metric-description">Screens requiring intensive testing</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Changes</div>
                    <div class="metric-value">{sum(m['total_changes'] for m in metrics_by_build.values())}</div>
                    <div class="metric-description">Components modified across builds</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Stability</div>
                    <div class="metric-value">{np.mean([m['avg_stability'] for m in metrics_by_build.values()]):.1f}%</div>
                    <div class="metric-description">Average stability score</div>
                </div>
            </div>

            <div class="chart-container">
                <canvas id="trendsChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="riskChart"></canvas>
            </div>

            <h3>Top Critical Screens</h3>
            <table>
                <thead><tr><th>Screen</th><th>Risk</th><th>Anomaly</th><th>Level</th></tr></thead>
                <tbody>
    """
    for screen, risk, anomaly in critical_screens:
        if risk>=80: level,badge="CRITICAL","risk-critical"
        elif risk>=60: level,badge="HIGH","risk-high"
        elif risk>=40: level,badge="MEDIUM","risk-medium"
        else: level,badge="LOW","risk-low"
        html+=f"<tr><td>{screen[:40]}</td><td>{risk:.1f}%</td><td>{anomaly:.2f}</td><td><span class='risk-badge {badge}'>{level}</span></td></tr>"

    html+=f"""
                </tbody>
            </table>
        </div>

        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const builds = {json.dumps(builds_sorted)};
            const removedData = {json.dumps([metrics_by_build[b]["removed"] for b in builds_sorted])};
            const addedData = {json.dumps([metrics_by_build[b]["added"] for b in builds_sorted])};
            const modifiedData = {json.dumps([metrics_by_build[b]["modified"] for b in builds_sorted])};
            const riskData = {json.dumps([metrics_by_build[b]["avg_risk"] for b in builds_sorted])};

            new Chart(document.getElementById('trendsChart'), {{
                type:'line',
                data:{{ labels: builds, datasets:[
                    {{label:'Removed', data:removedData, borderColor:'#ef4444', backgroundColor:'rgba(239,68,68,0.1)', fill:true, tension:0.4}},
                    {{label:'Added', data:addedData, borderColor:'#10b981', backgroundColor:'rgba(16,185,129,0.1)', fill:true, tension:0.4}},
                    {{label:'Modified', data:modifiedData, borderColor:'#f59e0b', backgroundColor:'rgba(245,158,11,0.1)', fill:true, tension:0.4}}
                ]}},
                options:{{ responsive:true, maintainAspectRatio:false, scales:{{ x:{{ ticks:{{ maxRotation:45, minRotation:30, autoSkip:true }} }}, y:{{ beginAtZero:true }} }} }}
            }});

            new Chart(document.getElementById('riskChart'), {{
                type:'bar',
                data:{{ labels: builds, datasets:[{{ label:'Risk %', data:riskData, backgroundColor:riskData.map(v=>v>=80?'rgba(239,68,68,0.8)':v>=60?'rgba(245,158,11,0.8)':v>=40?'rgba(168,85,247,0.8)':'rgba(16,185,129,0.8)') }}] }},
                options:{{ responsive:true, maintainAspectRatio:false, indexAxis:'y', scales:{{ x:{{ max:100 }} }} }}
            }});
        }});
        </script>
    </body>
    </html>
    """
    return html

def _generate_ai_dashboard_html_v6(
    tester_id: str,
    builds: list,
    builds_data: dict,
    total_screens: int = None,
    high_risk_screens: int = None,
    show_predictions: bool = True
) -> str:
    import json
    import plotly.graph_objects as go

    priorities = {"high": 0, "medium": 0, "low": 0}
    clusters = {}
    total_hours = 0
    rows_html = ""
    seen_screens = set()

    # JS para toggle de detalles
    js_script = """
    <script>
    function toggleDetails(id) {
        const row = document.getElementById(id);
        row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
    }
    </script>
    """

    for build_id in builds:
        diffs = builds_data.get(build_id, [])
        if not diffs:
            continue

        # Fila de separación por build
        rows_html += f"""
        <tr style="background:#f0f0f0;font-weight:bold">
            <td colspan="10">Build: {build_id}</td>
        </tr>
        """

        for d in diffs:
            # Filtrar para mostrar solo una fila por pantalla+build
            screen_key = f"{d.get('screen', 'unknown')}_{build_id}"
            if screen_key in seen_screens:
                continue
            seen_screens.add(screen_key)

            # Contar prioridades y clusters SOLO para los diffs visibles
            diff_priority = d.get("diff_priority", "low")
            priorities[diff_priority] += 1

            cid = d.get("cluster_id")
            if cid:
                clusters[cid] = clusters.get(cid, 0) + 1

            total_hours += d.get("testing_hours", 0)

            badge_color = {"high": "#ff3b30", "medium": "#ff9500", "low": "#34c759"}.get(diff_priority, "#34c759")

            structural_changes = d.get("structural_changes", 0)
            text_changes = d.get("text_changes", 0)
            changes_summary = f"S:{structural_changes} / T:{text_changes}"

            # Crear fila de detalles ocultos
            detail_id = f"detail-{d.get('id', '0')}"
            detail_html = f"""
            <tr id="{detail_id}" style="display:none;background:#f9f9f9;">
                <td colspan="10" style="text-align:left;">
                    <b>Added:</b> {json.dumps(d.get('added_nodes', []))}<br>
                    <b>Removed:</b> {json.dumps(d.get('removed_nodes', []))}<br>
                    <b>Modified:</b> {json.dumps(d.get('modified_nodes', []))}<br>
                    <b>Text Diff:</b> {json.dumps(d.get('modified_texts', []))}
                </td>
            </tr>
            """

            # Fila principal
            rows_html += f"""
            <tr>
                <td>
                    <a href="javascript:void(0);" onclick="toggleDetails('{detail_id}')">
                        {d.get('screen', 'Unknown')}
                    </a>
                </td>
                <td>{d.get('stability_score', 0.0):.1f}</td>
                <td><span style="color:{badge_color};font-weight:bold">{d.get('risk_score', 0):.0f}</span></td>
                <td>{d.get('impact_level', 'LOW')}</td>
                <td>{d.get('failure_probability', 0.0):.1f}%</td>
                <td>{d.get('confidence', 0):.0f}%</td>
                <td>{d.get('testing_hours', 0.0):.1f} h</td>
                <td>{d.get('testing_resources', 'N/A')}</td>
                <td>{d.get('recommendation', '')}</td>
                <td title="Cantidad de cambios estructurales (S) y solo de texto (T)">{changes_summary}</td>
            </tr>
            {detail_html}
            """

    total_screens = len(seen_screens)
    high_risk_screens = priorities.get("high", 0)

    # Gráficos
    donut = go.Figure(data=[go.Pie(
        labels=["High", "Medium", "Low"],
        values=[priorities.get("high", 0), priorities.get("medium", 0), priorities.get("low", 0)],
        hole=.55
    )])
    donut.update_layout(width=380, height=300, margin=dict(t=10, b=10), title="Priority")
    donut_html = donut.to_html(include_plotlyjs="cdn", full_html=False)

    cluster_fig = go.Figure(data=[go.Bar(
        x=list(clusters.keys()), 
        y=list(clusters.values())
    )])
    cluster_fig.update_layout(width=480, height=300, title="Clusters with the most changes")
    cluster_html = cluster_fig.to_html(include_plotlyjs=False, full_html=False)

    # HTML final
    return f"""
    <html>
    <head>
        <title>QA AI Dashboard 👩‍💻🤖</title>
        <style>
            body {{ font-family: Arial; padding:20px; }}
            table {{ width:100%; border-collapse: collapse; margin-top:20px; }}
            th,td {{ border:1px solid #ccc; padding:6px; text-align:center; }}
            th {{ background:#eee }}
            .kpis {{ display:flex; gap:20px; margin-bottom:20px; }}
            .box {{ padding:12px 18px; background:#fafafa; border-radius:8px; font-size:17px; }}
            .charts {{ display:flex; gap:25px; margin:20px 0; }}
            a {{ cursor:pointer; text-decoration:none; color:#007bff; }}
        </style>
        {js_script}
    </head>
    <body>
        <h1>QA AI Dashboard v6</h1>
        <h3>Tester: {tester_id}</h3>

        <div class="kpis">
            <div class="box">📱 Screens: <b>{total_screens}</b></div>
            <div class="box">🔴 High Risk: <b>{high_risk_screens}</b></div>
            <div class="box">⏱ Total Effort: <b>{total_hours:.1f}h</b></div>
        </div>

        <div class="charts">
            {donut_html}
            {cluster_html}
        </div>

        <h3>Screen Risk & Testing Matrix</h3>
        <table>
            <tr>
                <th>Screen</th>
                <th>Stability</th>
                <th>Risk</th>
                <th>Impact</th>
                <th>Failure %</th>
                <th>Conf.</th>
                <th>Hours</th>
                <th>Resources</th>
                <th>Recommendation</th>
                <th>#Changes (S/T)</th>
            </tr>
            {rows_html}
        </table>
    </body>
    </html>
    """



# =========================================================================
# 📋 ENDPOINT DE REPORTE EN PDF
# =========================================================================

@qa_ai_router.get("/report-pdf/{tester_id}")
def generate_qa_ai_report_pdf(tester_id: str, builds_to_compare: int = Query(5)):
    """
    Genera reporte en PDF con insights de IA
    """
    # Esta función puede integrarse con librería como reportlab
    # por ahora retorna HTML que se puede convertir a PDF
    return {
        "status": "generating",
        "message": f"Reporte en PDF para {tester_id} se está generando",
        "url": f"/api/qa-ai/report-html/{tester_id}?builds={builds_to_compare}"
    }
