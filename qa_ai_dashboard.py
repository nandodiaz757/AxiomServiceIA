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

qa_ai_router = APIRouter(prefix="/api/qa-ai", tags=["QA AI"])

# =========================================================================
# 🧠 ANÁLISIS INTELIGENTE DE CAMBIOS
# =========================================================================

class ChangeAnalyzer:
    """Analiza cambios entre versiones con inteligencia artificial"""
    
    @staticmethod
    def calculate_stability_score(removed_count: int, added_count: int, modified_count: int) -> float:
        """
        Calcula score de estabilidad (0-100)
        - 100: Sin cambios
        - 0: Muchos cambios (inestable)
        """
        total_changes = removed_count + added_count + modified_count
        if total_changes == 0:
            return 100.0
        
        # Penalty por cambios: mayor cambio = menor score
        # Formula: 100 - (cambios * 20)
        score = max(0, 100 - (total_changes * 20))
        return float(score)
    
    @staticmethod
    def calculate_risk_score(
        stability: float,
        frequency: int,
        modification_intensity: float,
        historical_failures: int = 0
    ) -> Dict:
        """
        Calcula score de riesgo (0-100) basado en múltiples factores
        
        Factores:
        - Estabilidad: pantallas inestables = mayor riesgo
        - Frecuencia: cambios frecuentes = mayor riesgo
        - Intensidad: cambios grandes = mayor riesgo
        - Historial: fallos previos = mayor riesgo
        """
        # Normalizar métricas
        stability_risk = (100 - stability) * 0.4  # 40% del peso
        frequency_risk = min(frequency / 10, 1.0) * 100 * 0.3  # 30% del peso
        intensity_risk = min(modification_intensity, 1.0) * 100 * 0.2  # 20% del peso
        historical_risk = min(historical_failures / 5, 1.0) * 100 * 0.1  # 10% del peso
        
        total_risk = stability_risk + frequency_risk + intensity_risk + historical_risk
        
        return {
            "total_risk": min(total_risk, 100.0),
            "stability_factor": stability_risk,
            "frequency_factor": frequency_risk,
            "intensity_factor": intensity_risk,
            "historical_factor": historical_risk
        }
    
    @staticmethod
    def predict_failure_probability(
        risk_score: float,
        change_magnitude: float,
        similar_past_issues: int = 0
    ) -> Dict:
        """
        Predice probabilidad de fallos futuros (0-100%)
        
        Basado en:
        - Risk score actual
        - Magnitud del cambio
        - Patrones históricos similares
        """
        # Fórmula base: risk_score normalizado
        base_probability = risk_score / 100.0
        
        # Factor de magnitud del cambio
        change_factor = min(change_magnitude, 1.0)
        
        # Factor de patrones históricos
        # Si hay issues similares previas, aumenta probabilidad
        history_factor = min(similar_past_issues * 0.15, 1.0)
        
        # Probabilidad final (ponderada)
        failure_prob = (
            base_probability * 0.5 +
            change_factor * 0.3 +
            history_factor * 0.2
        ) * 100
        
        return {
            "failure_probability": min(failure_prob, 100.0),
            "confidence": min(80 + similar_past_issues * 5, 99),
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
    def calculate_change_impact(
        removed: List, added: List, modified: List
    ) -> Dict:
        """
        Analiza el impacto de cambios en diferentes dimensiones
        """
        total_changes = len(removed) + len(added) + len(modified)
        
        if total_changes == 0:
            return {
                "impact_level": "NONE",
                "areas_affected": [],
                "severity": 0
            }
        
        # Extraer áreas afectadas
        areas = defaultdict(int)
        for node in removed + added:
            class_name = node.get("class", "unknown")
            areas[class_name] += 1
        for change in modified:
            class_name = change.get("node", {}).get("class", "unknown")
            areas[class_name] += 1
        
        # Clasificar por severidad
        if total_changes >= 10:
            severity = 3  # Crítico
            level = "CRITICAL"
        elif total_changes >= 5:
            severity = 2  # Alto
            level = "HIGH"
        else:
            severity = 1  # Bajo
            level = "LOW"
        
        return {
            "impact_level": level,
            "areas_affected": sorted(areas.items(), key=lambda x: x[1], reverse=True)[:5],
            "severity": severity,
            "total_components_changed": total_changes
        }
    
    @staticmethod
    def find_trending_issues(
        screen_diffs_history: List[Dict]
    ) -> Dict:
        """
        Identifica patrones y tendencias en problemas
        """
        issue_patterns = defaultdict(list)
        
        for diff in screen_diffs_history:
            components = set()
            
            for node in diff.get("removed", []) + diff.get("added", []):
                components.add(node.get("class", "unknown"))
            
            for change in diff.get("modified", []):
                components.add(change.get("node", {}).get("class", "unknown"))
            
            for component in components:
                issue_patterns[component].append(diff.get("created_at"))
        
        # Identificar componentes problemáticos (cambian frecuentemente)
        trending = {}
        for component, timestamps in issue_patterns.items():
            if len(timestamps) >= 3:
                trending[component] = {
                    "occurrences": len(timestamps),
                    "frequency": "Alta" if len(timestamps) >= 5 else "Media",
                    "last_change": max(timestamps) if timestamps else None
                }
        
        return sorted(
            trending.items(),
            key=lambda x: x[1]["occurrences"],
            reverse=True
        )


class MetricsCalculator:
    """Calcula métricas de IA para decisiones"""
    
    @staticmethod
    def calculate_test_coverage_gap(
        total_screens: int,
        tested_screens: int,
        high_risk_untested: int
    ) -> Dict:
        """Identifica gaps en cobertura de testing"""
        coverage_pct = (tested_screens / total_screens * 100) if total_screens > 0 else 0
        
        return {
            "total_coverage": coverage_pct,
            "untested_screens": total_screens - tested_screens,
            "high_risk_gap": high_risk_untested,
            "priority": "🔴 CRÍTICO" if high_risk_untested > 0 else "🟢 OK"
        }
    
    @staticmethod
    def calculate_regression_risk(
        current_build_changes: Dict,
        previous_builds_history: List[Dict],
        max_lookback_builds: int = 5
    ) -> Dict:
        """
        Calcula riesgo de regresión analizando historial
        """
        if not previous_builds_history:
            return {"regression_risk": 0, "similar_past_issues": 0}
        
        # Buscar patrones similares en builds anteriores
        similar_changes = 0
        for past_build in previous_builds_history[-max_lookback_builds:]:
            # Comparar áreas afectadas
            if _compare_change_patterns(current_build_changes, past_build):
                similar_changes += 1
        
        risk_pct = (similar_changes / len(previous_builds_history[-max_lookback_builds:])) * 100
        
        return {
            "regression_risk": min(risk_pct, 100),
            "similar_past_issues": similar_changes,
            "recommendation": "High regression risk detected" if risk_pct > 50 else "Normal regression risk"
        }
    
    @staticmethod
    def calculate_effort_estimate(
        stability_score: float,
        total_changes: int,
        high_risk_count: int
    ) -> Dict:
        """
        Estima esfuerzo de testing requerido
        """
        # Base: 30 minutos por pantalla
        base_time = 30
        
        # Multiplicador por estabilidad (inestable = más testing)
        stability_multiplier = (100 - stability_score) / 100 * 2 + 1
        
        # Multiplicador por cambios
        change_multiplier = 1 + (min(total_changes, 20) / 20)
        
        # Multiplicador por riesgo
        risk_multiplier = 1 + (high_risk_count / 5)
        
        total_hours = (base_time * stability_multiplier * change_multiplier * risk_multiplier) / 60
        
        return {
            "estimated_hours": round(total_hours, 1),
            "estimated_days": round(total_hours / 8, 1),
            "test_cases_recommended": int(total_changes * 3 + high_risk_count * 5),
            "resource_level": _get_resource_level(total_hours)
        }


def _compare_change_patterns(current: Dict, historical: Dict) -> bool:
    """Compara si dos conjuntos de cambios son similares"""
    # Implementar lógica de similitud
    return False


def _get_resource_level(hours: float) -> str:
    """Retorna nivel de recursos necesarios"""
    if hours <= 4:
        return "👤 1 Tester"
    elif hours <= 16:
        return "👥 2 Testers"
    elif hours <= 40:
        return "👥👥 3-4 Testers"
    else:
        return "👥👥👥 5+ Testers + Team Lead"


# =========================================================================
# 📊 DASHBOARD ENDPOINT
# =========================================================================

@qa_ai_router.get("/dashboard-advanced/{tester_id}", response_class=HTMLResponse)
def qa_ai_dashboard_advanced(
    tester_id: str,
    builds_to_compare: int = Query(5, ge=1, le=20),
    show_predictions: bool = Query(True)
):
    """
    Dashboard IA avanzado con:
    - Análisis de cambios entre versiones
    - Predicción de fallos
    - Métricas de riesgo
    - Recomendaciones de testing
    - Análisis de tendencias
    """
    
    conn = sqlite3.connect("accessibility.db")
    c = conn.cursor()
    
    # 1️⃣ OBTENER HISTORIAL DE BUILDS
    c.execute("""
        SELECT DISTINCT build_id 
        FROM screen_diffs 
        WHERE tester_id = ?
        ORDER BY created_at DESC 
        LIMIT ?
    """, (tester_id, builds_to_compare))
    builds = [row[0] for row in c.fetchall()]
    
    if not builds:
        conn.close()
        return HTMLResponse(content="<h1>No data found for this tester</h1>")
    
    # 2️⃣ OBTENER DATOS DE DIFFS POR BUILD
    builds_data = {}
    for build_id in builds:
        c.execute("""
            SELECT id, screen_name, removed, added, modified, created_at, 
                   anomaly_score, cluster_id, approval_status
            FROM screen_diffs 
            WHERE tester_id = ? AND build_id = ?
            ORDER BY created_at DESC
        """, (tester_id, build_id))
        
        diffs = []
        for row in c.fetchall():
            diff_id, screen, removed_raw, added_raw, modified_raw, created_at, anomaly, cluster, approval = row
            
            removed = json.loads(removed_raw) if removed_raw else []
            added = json.loads(added_raw) if added_raw else []
            modified = json.loads(modified_raw) if modified_raw else []
            
            stability = ChangeAnalyzer.calculate_stability_score(
                len(removed), len(added), len(modified)
            )
            
            # impact = ChangeAnalyzer.calculate_impact(removed, added, modified)
            impact = ChangeAnalyzer.calculate_change_impact(removed, added, modified)

            
            risk_factors = ChangeAnalyzer.calculate_risk_score(
                stability=stability,
                frequency=1,
                modification_intensity=len(modified) / max(10, len(removed) + len(added) + len(modified)),
                historical_failures=1 if approval == "rejected" else 0
            )
            
            diffs.append({
                "id": diff_id,
                "screen": screen,
                "removed": removed,
                "added": added,
                "modified": modified,
                "created_at": created_at,
                "stability_score": stability,
                "impact": impact,
                "risk_score": risk_factors["total_risk"],
                "anomaly_score": anomaly or 0,
                "cluster_id": cluster,
                "approval": approval
            })
        
        builds_data[build_id] = diffs
    
    conn.close()
    
    # 3️⃣ ANÁLISIS AGREGADO
    total_screens = len(set(d["screen"] for build_diffs in builds_data.values() for d in build_diffs))
    high_risk_screens = sum(1 for build_diffs in builds_data.values() for d in build_diffs if d["risk_score"] > 60)
    
    # 4️⃣ GENERAR HTML CON CHART.JS Y PLOTLY
    html_content = _generate_ai_dashboard_html(
        tester_id=tester_id,
        builds=builds,
        builds_data=builds_data,
        total_screens=total_screens,
        high_risk_screens=high_risk_screens,
        show_predictions=show_predictions
    )
    
    return HTMLResponse(content=html_content)


def _generate_ai_dashboard_html(
    tester_id: str,
    builds: List[str],
    builds_data: Dict,
    total_screens: int,
    high_risk_screens: int,
    show_predictions: bool
) -> str:
    """Genera HTML del dashboard con todas las visualizaciones"""
    
    # Preparar datos para gráficos
    builds_sorted = sorted(builds, reverse=True)
    
    # Métricas por build
    metrics_by_build = {}
    for build_id in builds_sorted:
        diffs = builds_data.get(build_id, [])
        
        if not diffs:
            continue
        
        total_removed = sum(len(d["removed"]) for d in diffs)
        total_added = sum(len(d["added"]) for d in diffs)
        total_modified = sum(len(d["modified"]) for d in diffs)
        avg_risk = np.mean([d["risk_score"] for d in diffs]) if diffs else 0
        avg_stability = np.mean([d["stability_score"] for d in diffs]) if diffs else 100
        
        metrics_by_build[build_id] = {
            "removed": total_removed,
            "added": total_added,
            "modified": total_modified,
            "avg_risk": float(avg_risk),
            "avg_stability": float(avg_stability),
            "total_changes": total_removed + total_added + total_modified,
            "total_screens": len(diffs)
        }
    
    # Identificar pantallas críticas
    all_diffs = [d for diffs in builds_data.values() for d in diffs]
    critical_screens = sorted(
        [(d["screen"], d["risk_score"], d["anomaly_score"]) for d in all_diffs 
         if d["risk_score"] > 60],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Tendencias
    trending_issues = ChangeAnalyzer.find_trending_issues(all_diffs)[:8]
    
    # HTML Template
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 QA IA Dashboard - {tester_id}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
        <style>
            :root {{
                --primary: #6366f1;
                --success: #10b981;
                --danger: #ef4444;
                --warning: #f59e0b;
                --info: #3b82f6;
                --dark: #1f2937;
                --light: #f9fafb;
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1600px;
                margin: 0 auto;
            }}
            
            .header {{
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                border-left: 5px solid var(--primary);
            }}
            
            .header h1 {{
                color: var(--dark);
                margin-bottom: 5px;
                font-size: 28px;
            }}
            
            .header .subtitle {{
                color: #6b7280;
                font-size: 14px;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-top: 4px solid;
                transition: transform 0.2s;
            }}
            
            .metric-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.15);
            }}
            
            .metric-card.danger {{ border-top-color: var(--danger); }}
            .metric-card.warning {{ border-top-color: var(--warning); }}
            .metric-card.success {{ border-top-color: var(--success); }}
            .metric-card.info {{ border-top-color: var(--info); }}
            
            .metric-value {{
                font-size: 32px;
                font-weight: bold;
                color: var(--dark);
                margin: 10px 0;
            }}
            
            .metric-label {{
                color: #6b7280;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .metric-description {{
                font-size: 12px;
                color: #9ca3af;
                margin-top: 8px;
            }}
            
            .progress-bar {{
                width: 100%;
                height: 8px;
                background: #e5e7eb;
                border-radius: 4px;
                margin-top: 10px;
                overflow: hidden;
            }}
            
            .progress-bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, var(--primary), var(--info));
                transition: width 0.3s ease;
            }}
            
            .section {{
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            
            .section-title {{
                font-size: 20px;
                font-weight: bold;
                color: var(--dark);
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid var(--light);
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .chart-container {{
                position: relative;
                height: 400px;
                margin-bottom: 20px;
            }}
            
            .table-responsive {{
                overflow-x: auto;
                margin-top: 15px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            
            th {{
                background: var(--light);
                padding: 12px;
                text-align: left;
                font-weight: 600;
                color: var(--dark);
                border-bottom: 2px solid #e5e7eb;
            }}
            
            td {{
                padding: 12px;
                border-bottom: 1px solid #e5e7eb;
            }}
            
            tr:hover {{
                background: #f9fafb;
            }}
            
            .risk-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                text-align: center;
                min-width: 80px;
            }}
            
            .risk-critical {{
                background: #fee2e2;
                color: #991b1b;
            }}
            
            .risk-high {{
                background: #fef3c7;
                color: #92400e;
            }}
            
            .risk-medium {{
                background: #fce7f3;
                color: #831843;
            }}
            
            .risk-low {{
                background: #dbeafe;
                color: #1e3a8a;
            }}
            
            .recommendation-box {{
                background: #f0f9ff;
                border-left: 4px solid var(--info);
                padding: 15px;
                border-radius: 6px;
                margin: 10px 0;
                font-size: 13px;
                color: #1e3a8a;
            }}
            
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            
            @media (max-width: 1200px) {{
                .grid-2 {{ grid-template-columns: 1fr; }}
            }}
            
            .badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                background: var(--light);
                color: var(--dark);
            }}
            
            .badge-success {{ background: #d1fae5; color: #065f46; }}
            .badge-danger {{ background: #fee2e2; color: #991b1b; }}
            .badge-warning {{ background: #fef3c7; color: #92400e; }}
            
            .insights-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}
            
            .insight-item {{
                background: var(--light);
                padding: 15px;
                border-radius: 8px;
                border-left: 3px solid var(--primary);
            }}
            
            .insight-item.critical {{
                border-left-color: var(--danger);
                background: #fef2f2;
            }}
            
            .insight-item.warning {{
                border-left-color: var(--warning);
                background: #fffbeb;
            }}
            
            .insight-title {{
                font-weight: 600;
                color: var(--dark);
                margin-bottom: 5px;
            }}
            
            .insight-text {{
                font-size: 13px;
                color: #6b7280;
            }}
            
            .timestamp {{
                font-size: 11px;
                color: #9ca3af;
                margin-top: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- HEADER -->
            <div class="header">
                <h1>🤖 QA IA Dashboard</h1>
                <div class="subtitle">Análisis Inteligente de Cambios UI - Tester: <strong>{tester_id}</strong></div>
                <div class="subtitle">Análisis de {len(builds_sorted)} builds | {total_screens} pantallas | Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
            </div>
            
            <!-- MÉTRICAS PRINCIPALES -->
            <div class="metrics-grid">
                <div class="metric-card danger">
                    <div class="metric-label">🔴 Riesgo Promedio</div>
                    <div class="metric-value">{np.mean([m['avg_risk'] for m in metrics_by_build.values()]):.1f}%</div>
                    <div class="metric-description">Riesgo de fallo estimado</div>
                    <div class="progress-bar">
                        <div class="progress-bar-fill" style="width: {np.mean([m['avg_risk'] for m in metrics_by_build.values()])}%"></div>
                    </div>
                </div>
                
                <div class="metric-card warning">
                    <div class="metric-label">⚠️ Pantallas Críticas</div>
                    <div class="metric-value">{high_risk_screens}</div>
                    <div class="metric-description">Requieren testing intensivo</div>
                </div>
                
                <div class="metric-card info">
                    <div class="metric-label">📊 Total de Cambios</div>
                    <div class="metric-value">{sum(m['total_changes'] for m in metrics_by_build.values())}</div>
                    <div class="metric-description">Componentes modificados</div>
                </div>
                
                <div class="metric-card success">
                    <div class="metric-label">✅ Estabilidad</div>
                    <div class="metric-value">{np.mean([m['avg_stability'] for m in metrics_by_build.values()]):.1f}%</div>
                    <div class="metric-description">Score promedio</div>
                    <div class="progress-bar">
                        <div class="progress-bar-fill" style="width: {np.mean([m['avg_stability'] for m in metrics_by_build.values()])}%; background: linear-gradient(90deg, var(--success), var(--info));"></div>
                    </div>
                </div>
            </div>
            
            <!-- VISUALIZACIONES PRINCIPALES -->
            <div class="grid-2">
                <!-- Tendencia de Cambios -->
                <div class="section">
                    <div class="section-title">📈 Tendencia de Cambios por Build</div>
                    <div class="chart-container">
                        <canvas id="trendsChart"></canvas>
                    </div>
                </div>
                
                <!-- Distribución de Riesgo -->
                <div class="section">
                    <div class="section-title">🎯 Distribución de Riesgo</div>
                    <div class="chart-container">
                        <canvas id="riskDistributionChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- PANTALLAS CRÍTICAS -->
            <div class="section">
                <div class="section-title">🚨 Top 10 Pantallas Críticas</div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Pantalla</th>
                                <th>Score Riesgo</th>
                                <th>Anomaly Score</th>
                                <th>Nivel</th>
                                <th>Acción Recomendada</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    for screen, risk, anomaly in critical_screens:
        if risk >= 80:
            level = "CRÍTICO"
            badge_class = "risk-critical"
            action = "🔴 Testing exhaustivo + Code Review"
        elif risk >= 60:
            level = "ALTO"
            badge_class = "risk-high"
            action = "🟠 Testing intensivo + Edge cases"
        elif risk >= 40:
            level = "MEDIO"
            badge_class = "risk-medium"
            action = "🟡 Testing estándar"
        else:
            level = "BAJO"
            badge_class = "risk-low"
            action = "🟢 Testing básico"
        
        html += f"""
                            <tr>
                                <td><strong>{screen[:50]}</strong></td>
                                <td><strong>{risk:.1f}%</strong></td>
                                <td>{anomaly:.2f}</td>
                                <td><span class="risk-badge {badge_class}">{level}</span></td>
                                <td>{action}</td>
                            </tr>
        """
    
    html += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- COMPONENTES CON PROBLEMAS RECURRENTES -->
            <div class="section">
                <div class="section-title">🔄 Componentes Problemáticos (Patrones Recurrentes)</div>
                <div class="insights-grid">
    """
    
    for component, info in trending_issues:
        html += f"""
                    <div class="insight-item {'critical' if info['occurrences'] >= 5 else 'warning'}">
                        <div class="insight-title">{component}</div>
                        <div class="insight-text">
                            Apariciones: <strong>{info['occurrences']}</strong><br>
                            Frecuencia: <strong>{info['frequency']}</strong><br>
                            Último cambio: <strong>{info.get('last_change', 'N/A')}</strong>
                        </div>
                    </div>
        """
    
    html += """
                </div>
            </div>
            
            <!-- ANÁLISIS DE IMPACTO POR BUILD -->
            <div class="section">
                <div class="section-title">📊 Análisis Comparativo por Build</div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Build</th>
                                <th>Pantallas</th>
                                <th>Removidos</th>
                                <th>Agregados</th>
                                <th>Modificados</th>
                                <th>Riesgo Promedio</th>
                                <th>Estabilidad</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    for build_id, metrics in metrics_by_build.items():
        html += f"""
                            <tr>
                                <td><strong>{build_id}</strong></td>
                                <td>{metrics['total_screens']}</td>
                                <td><span class="badge badge-danger">{metrics['removed']}</span></td>
                                <td><span class="badge badge-success">{metrics['added']}</span></td>
                                <td><span class="badge badge-warning">{metrics['modified']}</span></td>
                                <td><strong>{metrics['avg_risk']:.1f}%</strong></td>
                                <td><strong>{metrics['avg_stability']:.1f}%</strong></td>
                            </tr>
        """
    
    html += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- RECOMENDACIONES DE IA -->
            <div class="section">
                <div class="section-title">💡 Recomendaciones Inteligentes</div>
    """
    
    # Generar recomendaciones basadas en análisis
    avg_risk = np.mean([m['avg_risk'] for m in metrics_by_build.values()])
    avg_stability = np.mean([m['avg_stability'] for m in metrics_by_build.values()])
    total_changes = sum(m['total_changes'] for m in metrics_by_build.values())
    
    effort = MetricsCalculator.calculate_effort_estimate(
        stability_score=avg_stability,
        total_changes=total_changes,
        high_risk_count=high_risk_screens
    )
    
    html += f"""
                <div class="recommendation-box" style="background: #f0fdf4; border-left-color: var(--success); color: #1f2d0c;">
                    <strong>📋 Esfuerzo Estimado de Testing:</strong><br>
                    ⏱️ {effort['estimated_hours']} horas ({effort['estimated_days']} días)<br>
                    👥 {effort['resource_level']}<br>
                    🧪 {effort['test_cases_recommended']} casos de test recomendados
                </div>
                
                <div class="recommendation-box">
                    <strong>🎯 Estrategia de Testing Recomendada:</strong><br>
    """
    
    if avg_risk > 70:
        html += "🔴 <strong>MODO CRÍTICO:</strong> Ejecutar suite de tests completa + exploratory testing<br>"
    elif avg_risk > 50:
        html += "🟠 <strong>MODO INTENSIVO:</strong> Enfocarse en áreas críticas + edge cases<br>"
    else:
        html += "🟢 <strong>MODO ESTÁNDAR:</strong> Suite de tests normal + smoke tests<br>"
    
    if high_risk_screens > 0:
        html += f"⚠️ Priorizar testing en {high_risk_screens} pantallas de alto riesgo<br>"
    
    if total_changes > 20:
        html += f"📊 Alto volumen de cambios ({total_changes}): considerar regresión completa<br>"
    
    html += """
                </div>
                
                <div class="recommendation-box" style="background: #fef3c7; border-left-color: var(--warning); color: #78350f;">
                    <strong>⚡ Acciones Inmediatas:</strong><br>
                    1️⃣ Ejecutar smoke tests en todas las pantallas críticas<br>
                    2️⃣ Validar componentes con patrones recurrentes de problemas<br>
                    3️⃣ Crear tests específicos para cambios de alto riesgo<br>
                    4️⃣ Revisar resultados de builds previas similares
                </div>
            </div>
            
            <!-- COMPARACIÓN INTERACTIVA -->
            <div class="section">
                <div class="section-title">🔍 Comparación Interactiva de Builds</div>
                <div id="comparisonChart" style="width: 100%; height: 500px;"></div>
            </div>
        </div>
        
        <script>
            // Chart 1: Tendencia de Cambios
            const trendsCtx = document.getElementById('trendsChart').getContext('2d');
            const builds = {json.dumps(builds_sorted)};
            const removedData = {json.dumps([metrics_by_build.get(b, {}).get('removed', 0) for b in builds_sorted])};
            const addedData = {json.dumps([metrics_by_build.get(b, {}).get('added', 0) for b in builds_sorted])};
            const modifiedData = {json.dumps([metrics_by_build.get(b, {}).get('modified', 0) for b in builds_sorted])};
            
            new Chart(trendsCtx, {{
                type: 'line',
                data: {{
                    labels: builds,
                    datasets: [
                        {{
                            label: 'Removidos',
                            data: removedData,
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            tension: 0.4,
                            fill: true
                        }},
                        {{
                            label: 'Agregados',
                            data: addedData,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4,
                            fill: true
                        }},
                        {{
                            label: 'Modificados',
                            data: modifiedData,
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            tension: 0.4,
                            fill: true
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: false }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }},
                        x: {{ grid: {{ display: false }} }}
                    }}
                }}
            }});
            
            // Chart 2: Distribución de Riesgo
            const riskCtx = document.getElementById('riskDistributionChart').getContext('2d');
            const riskData = {json.dumps([metrics_by_build.get(b, {}).get('avg_risk', 0) for b in builds_sorted])};
            
            new Chart(riskCtx, {{
                type: 'bar',
                data: {{
                    labels: builds,
                    datasets: [{{
                        label: 'Score de Riesgo (%)',
                        data: riskData,
                        backgroundColor: riskData.map(v => {{
                            if (v >= 80) return 'rgba(239, 68, 68, 0.8)';
                            if (v >= 60) return 'rgba(245, 158, 11, 0.8)';
                            if (v >= 40) return 'rgba(168, 85, 247, 0.8)';
                            return 'rgba(16, 185, 129, 0.8)';
                        }})
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {{
                        x: {{ max: 100 }}
                    }},
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
            
            // Plotly Comparison Chart
            const comparisonTrace = {{
                x: builds,
                y: {json.dumps([metrics_by_build.get(b, {}).get('total_changes', 0) for b in builds_sorted])},
                name: 'Total de Cambios',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: '#6366f1', width: 3 }},
                marker: {{ size: 10 }}
            }};
            
            Plotly.newPlot('comparisonChart', [comparisonTrace], {{
                title: 'Evolución de Cambios por Build',
                xaxis: {{ title: 'Build' }},
                yaxis: {{ title: 'Número de Cambios' }},
                hovermode: 'closest',
                plot_bgcolor: '#f9fafb',
                paper_bgcolor: 'white'
            }}, {{ responsive: true }});
        </script>
    </body>
    </html>
    """
    
    return html


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
