#!/usr/bin/env python3
"""
🤖 QA IA Dashboard - Script de Prueba y Ejemplos
Ejecuta: python test_qa_ai_dashboard.py
"""

import requests
import json
from datetime import datetime
from typing import Optional

BASE_URL = "http://localhost:8000"

print("""
╔══════════════════════════════════════════════════════╗
║  🤖 QA IA DASHBOARD - SCRIPT DE PRUEBAS             ║
╚══════════════════════════════════════════════════════╝
""")

# =========================================================================
# EJEMPLOS DE CURL
# =========================================================================

CURL_EXAMPLES = {
    "1. Dashboard Básico (5 builds)": """
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz"
    """,
    
    "2. Dashboard Detallado (10 builds)": """
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz?builds_to_compare=10"
    """,
    
    "3. Con Predicciones Deshabilitadas": """
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz?show_predictions=false"
    """,
    
    "4. Máximo de Comparación (20 builds)": """
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz?builds_to_compare=20"
    """,
    
    "5. Tester Diferente": """
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/otro_tester?builds_to_compare=5"
    """,
}

# =========================================================================
# FUNCIÓN DE PRUEBA
# =========================================================================

def test_dashboard(tester_id: str = "luis_diaz", builds: int = 5):
    """Prueba el dashboard IA"""
    
    url = f"{BASE_URL}/api/qa-ai/dashboard-advanced/{tester_id}"
    params = {
        "builds_to_compare": builds,
        "show_predictions": True
    }
    
    print(f"\n📡 Conectando a: {url}")
    print(f"   Parámetros: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"📏 Content Length: {len(response.text)} bytes")
        print(f"⏱️  Tiempo de respuesta: OK")
        
        # Verificar estructura HTML
        if "<html" in response.text:
            print("✅ HTML válido recibido")
        if "canvas" in response.text:
            print("✅ Gráficos (Chart.js) incluidos")
        if "Plotly" in response.text:
            print("✅ Visualizaciones (Plotly) incluidas")
        
        # Guardar en archivo
        with open("dashboard_output.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("\n💾 Dashboard guardado en: dashboard_output.html")
        
        return response.text
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a servidor")
        print("   ¿Está corriendo? python backend.py")
    except Exception as e:
        print(f"❌ Error: {e}")

# =========================================================================
# EJEMPLOS EN PYTHON
# =========================================================================

def example_1_basic():
    """Ejemplo 1: Acceso básico"""
    print("\n" + "="*70)
    print("EJEMPLO 1: Acceso Básico al Dashboard")
    print("="*70)
    
    tester_id = "luis_diaz"
    test_dashboard(tester_id=tester_id, builds=5)

def example_2_detailed():
    """Ejemplo 2: Dashboard detallado con más builds"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Dashboard Detallado (10 builds)")
    print("="*70)
    
    tester_id = "luis_diaz"
    test_dashboard(tester_id=tester_id, builds=10)

def example_3_multiple_testers():
    """Ejemplo 3: Varios testers"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Comparar Múltiples Testers")
    print("="*70)
    
    testers = ["luis_diaz", "maria_garcia", "juan_lopez"]
    
    for tester in testers:
        print(f"\n📊 Analizando tester: {tester}")
        try:
            response = requests.get(
                f"{BASE_URL}/api/qa-ai/dashboard-advanced/{tester}",
                params={"builds_to_compare": 5},
                timeout=30
            )
            print(f"   ✅ Status: {response.status_code}")
        except:
            print(f"   ⚠️  No hay datos para: {tester}")

def example_4_save_reports():
    """Ejemplo 4: Guardar reportes HTML"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Guardar Reportes en HTML")
    print("="*70)
    
    tester_id = "luis_diaz"
    
    for num_builds in [5, 10, 15]:
        print(f"\n📄 Generando reporte con {num_builds} builds...")
        
        try:
            response = requests.get(
                f"{BASE_URL}/api/qa-ai/dashboard-advanced/{tester_id}",
                params={"builds_to_compare": num_builds},
                timeout=30
            )
            
            filename = f"qa_report_{tester_id}_{num_builds}builds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            print(f"   ✅ Guardado: {filename}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def example_5_parse_metrics():
    """Ejemplo 5: Parsear métricas del HTML"""
    print("\n" + "="*70)
    print("EJEMPLO 5: Extraer Métricas del Dashboard")
    print("="*70)
    
    tester_id = "luis_diaz"
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/qa-ai/dashboard-advanced/{tester_id}",
            params={"builds_to_compare": 5},
            timeout=30
        )
        
        html = response.text
        
        # Búsquedas simples
        metrics = {}
        
        # Risk Score (buscar en el HTML)
        if "Riesgo Promedio" in html:
            print("✅ Métrica 'Riesgo Promedio' encontrada")
        
        if "Pantallas Críticas" in html:
            print("✅ Métrica 'Pantallas Críticas' encontrada")
        
        if "Total de Cambios" in html:
            print("✅ Métrica 'Total de Cambios' encontrada")
        
        if "Estabilidad" in html:
            print("✅ Métrica 'Estabilidad' encontrada")
        
        print("\n💡 Nota: Para parsing más avanzado, usar BeautifulSoup:")
        print("""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        risk_value = soup.find('div', class_='metric-value').text
        """)
        
    except Exception as e:
        print(f"❌ Error: {e}")

# =========================================================================
# BENCHMARK
# =========================================================================

def benchmark_performance():
    """Prueba de performance"""
    print("\n" + "="*70)
    print("BENCHMARK: Performance del Dashboard")
    print("="*70)
    
    import time
    
    tester_id = "luis_diaz"
    builds_options = [5, 10, 15, 20]
    
    results = []
    
    for num_builds in builds_options:
        print(f"\n⏱️  Midiendo con {num_builds} builds...")
        
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{BASE_URL}/api/qa-ai/dashboard-advanced/{tester_id}",
                params={"builds_to_compare": num_builds},
                timeout=60
            )
            
            elapsed = time.time() - start_time
            html_size = len(response.text) / 1024  # KB
            
            results.append({
                "builds": num_builds,
                "time": elapsed,
                "size_kb": html_size,
                "status": response.status_code
            })
            
            print(f"   ✅ {elapsed:.2f}s | {html_size:.1f} KB | Status: {response.status_code}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Resumen
    print("\n📊 RESUMEN DE PERFORMANCE:")
    print("=" * 50)
    for result in results:
        print(f"  {result['builds']} builds: {result['time']:.2f}s ({result['size_kb']:.1f} KB)")

# =========================================================================
# MENÚ PRINCIPAL
# =========================================================================

def main():
    print("""
EJEMPLOS DISPONIBLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Acceso Básico
2. Dashboard Detallado
3. Múltiples Testers
4. Guardar Reportes
5. Parsear Métricas
6. Benchmark de Performance
7. Ver Ejemplos CURL
8. Ejecutar Todo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    try:
        choice = input("Selecciona opción (1-8): ").strip()
        
        if choice == "1":
            example_1_basic()
        elif choice == "2":
            example_2_detailed()
        elif choice == "3":
            example_3_multiple_testers()
        elif choice == "4":
            example_4_save_reports()
        elif choice == "5":
            example_5_parse_metrics()
        elif choice == "6":
            benchmark_performance()
        elif choice == "7":
            print("\n📋 EJEMPLOS CURL:")
            for title, curl_cmd in CURL_EXAMPLES.items():
                print(f"\n{title}:")
                print(curl_cmd)
        elif choice == "8":
            print("\n⚙️  Ejecutando todos los ejemplos...\n")
            example_1_basic()
            example_2_detailed()
            example_3_multiple_testers()
            example_4_save_reports()
            benchmark_performance()
        else:
            print("❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n\n✋ Cancelado por usuario")

if __name__ == "__main__":
    # Verificar que el servidor está corriendo
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=2)
        if response.status_code == 200:
            print("✅ Servidor Backend: Conectado\n")
            main()
        else:
            print("❌ Servidor Backend: No disponible")
    except:
        print("""
❌ ERROR: El servidor backend no está corriendo

Para iniciar el servidor ejecuta:
  python backend.py

Luego vuelve a ejecutar este script:
  python test_qa_ai_dashboard.py
        """)

