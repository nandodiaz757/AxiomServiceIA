#!/usr/bin/env python3
"""
Script de prueba para validar los 3 nuevos endpoints de FlowAnalyticsEngine.

Endpoints:
1. POST /flow-analyze/{app_name}/{tester_id} - Análisis de flujos del tester
2. GET /flow-dashboard/{app_name} - Dashboard global de flujos
3. GET /flow-anomalies/{tester_id} - Historial de anomalías
"""

import asyncio
import json
import httpx
import sys
from datetime import datetime

# Configuración
BASE_URL = "http://localhost:8000"
APP_NAME = "com.grability.rappi"
TESTER_ID = "tester_001"

print("=" * 80)
print("🧪 PRUEBAS DE ENDPOINTS: FlowAnalyticsEngine")
print("=" * 80)
print(f"Base URL: {BASE_URL}")
print(f"App Name: {APP_NAME}")
print(f"Tester ID: {TESTER_ID}")
print()


async def test_flow_analyze():
    """Test endpoint: POST /flow-analyze/{app_name}/{tester_id}"""
    print("\n" + "=" * 80)
    print("📊 TEST 1: POST /flow-analyze/{app_name}/{tester_id}")
    print("=" * 80)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{BASE_URL}/flow-analyze/{APP_NAME}/{TESTER_ID}"
            
            # Body opcional
            payload = {
                "session_key": f"{TESTER_ID}_minute_block_xyz",
                "flow_sequence": ["home", "profile", "settings"]
            }
            
            print(f"URL: POST {url}")
            print(f"Body: {json.dumps(payload, indent=2)}")
            print()
            
            response = await client.post(url, json=payload)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            
            if response.status_code == 200:
                print("✅ TEST PASSED")
            else:
                print("❌ TEST FAILED")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")


async def test_flow_dashboard():
    """Test endpoint: GET /flow-dashboard/{app_name}"""
    print("\n" + "=" * 80)
    print("📈 TEST 2: GET /flow-dashboard/{app_name}")
    print("=" * 80)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{BASE_URL}/flow-dashboard/{APP_NAME}"
            
            print(f"URL: GET {url}")
            print()
            
            response = await client.get(url)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            
            if response.status_code == 200:
                print("✅ TEST PASSED")
            else:
                print("❌ TEST FAILED")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")


async def test_flow_anomalies():
    """Test endpoint: GET /flow-anomalies/{tester_id}"""
    print("\n" + "=" * 80)
    print("📋 TEST 3: GET /flow-anomalies/{tester_id}")
    print("=" * 80)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{BASE_URL}/flow-anomalies/{TESTER_ID}"
            
            # Query params
            params = {
                "limit": 20,
                "severity": "high"  # Filtrar por severidad (opcional)
            }
            
            print(f"URL: GET {url}")
            print(f"Query Params: {params}")
            print()
            
            response = await client.get(url, params=params)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            
            if response.status_code == 200:
                print("✅ TEST PASSED")
            else:
                print("❌ TEST FAILED")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")


async def test_all_endpoints():
    """Ejecutar todas las pruebas"""
    
    # Verificar conexión
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE_URL}/status")
            if response.status_code >= 400:
                print("⚠️ El servidor responde pero con error")
    except httpx.ConnectError:
        print("❌ ERROR: No se puede conectar al servidor FastAPI")
        print("   Asegúrate de que el servidor está corriendo en:", BASE_URL)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR al verificar conexión: {e}")
    
    # Ejecutar pruebas
    await test_flow_analyze()
    await test_flow_dashboard()
    await test_flow_anomalies()
    
    # Resumen
    print("\n" + "=" * 80)
    print("✅ PRUEBAS COMPLETADAS")
    print("=" * 80)
    print()
    print("Resumen de endpoints:")
    print("1. POST /flow-analyze/{app_name}/{tester_id} - Genera reporte de flujos")
    print("2. GET /flow-dashboard/{app_name} - Dashboard de análisis global")
    print("3. GET /flow-anomalies/{tester_id} - Historial de anomalías detectadas")
    print()


if __name__ == "__main__":
    print("\n🚀 Iniciando pruebas de endpoints FlowAnalyticsEngine...\n")
    asyncio.run(test_all_endpoints())
