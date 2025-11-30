#!/usr/bin/env python3
# =====================================================
# Test Script - Sistema de Configuración
# =====================================================
"""
Script para testear el ConfigManager y los endpoints de notificación.
Prueba:
1. Carga de configuración
2. Endpoints de configuración
3. Funciones de notificación
4. Health checks
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class ConfigTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = []
    
    def log(self, test_name: str, success: bool, message: str = "", response: Any = None):
        """Registra resultado de un test."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{status} | {test_name}")
        if message:
            print(f"   → {message}")
        if response and not success:
            print(f"   Response: {response}")
        self.results.append((test_name, success))
    
    def test_get_config(self):
        """Test: Obtener configuración actual."""
        try:
            response = requests.get(f"{self.base_url}/api/config", timeout=5)
            success = response.status_code == 200
            data = response.json() if success else None
            
            self.log(
                "GET /api/config",
                success,
                f"Config loaded. File: {data.get('file_path', 'N/A')}" if data else "",
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("GET /api/config", False, f"Exception: {str(e)}")
            return False, None
    
    def test_get_notifications_config(self):
        """Test: Obtener configuración de notificaciones."""
        try:
            response = requests.get(f"{self.base_url}/api/config/notifications", timeout=5)
            success = response.status_code == 200
            data = response.json() if success else None
            
            if success:
                slack_ready = data.get('slack', {}).get('enabled')
                teams_ready = data.get('teams', {}).get('enabled')
                jira_ready = data.get('jira', {}).get('enabled')
                msg = f"Slack: {slack_ready} | Teams: {teams_ready} | Jira: {jira_ready}"
            else:
                msg = ""
            
            self.log(
                "GET /api/config/notifications",
                success,
                msg,
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("GET /api/config/notifications", False, f"Exception: {str(e)}")
            return False, None
    
    def test_get_ci_config(self):
        """Test: Obtener configuración de CI."""
        try:
            response = requests.get(f"{self.base_url}/api/config/ci", timeout=5)
            success = response.status_code == 200
            data = response.json() if success else None
            
            if success:
                threshold = data.get('ci', {}).get('similarity_threshold')
                msg = f"Similarity threshold: {threshold}"
            else:
                msg = ""
            
            self.log(
                "GET /api/config/ci",
                success,
                msg,
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("GET /api/config/ci", False, f"Exception: {str(e)}")
            return False, None
    
    def test_get_ml_config(self):
        """Test: Obtener configuración de ML."""
        try:
            response = requests.get(f"{self.base_url}/api/config/ml", timeout=5)
            success = response.status_code == 200
            data = response.json() if success else None
            
            if success:
                batch_size = data.get('ml', {}).get('batch_size')
                msg = f"Batch size: {batch_size}"
            else:
                msg = ""
            
            self.log(
                "GET /api/config/ml",
                success,
                msg,
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("GET /api/config/ml", False, f"Exception: {str(e)}")
            return False, None
    
    def test_health_check(self):
        """Test: Health check de configuración."""
        try:
            response = requests.get(f"{self.base_url}/api/config/health", timeout=5)
            success = response.status_code == 200
            data = response.json() if success else None
            
            if success:
                overall = data.get('overall', 'unknown')
                msg = f"Overall status: {overall}"
            else:
                msg = ""
            
            self.log(
                "GET /api/config/health",
                success,
                msg,
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("GET /api/config/health", False, f"Exception: {str(e)}")
            return False, None
    
    def test_test_slack(self):
        """Test: Enviar mensaje de prueba a Slack."""
        try:
            response = requests.post(f"{self.base_url}/api/config/test-slack", timeout=10)
            success = response.status_code == 200
            data = response.json() if success else None
            
            msg = data.get('message', '') if success else data.get('error', '') if data else ''
            
            self.log(
                "POST /api/config/test-slack",
                success,
                msg,
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("POST /api/config/test-slack", False, f"Exception: {str(e)}")
            return False, None
    
    def test_test_teams(self):
        """Test: Enviar mensaje de prueba a Teams."""
        try:
            response = requests.post(f"{self.base_url}/api/config/test-teams", timeout=10)
            success = response.status_code == 200
            data = response.json() if success else None
            
            msg = data.get('message', '') if success else data.get('error', '') if data else ''
            
            self.log(
                "POST /api/config/test-teams",
                success,
                msg,
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("POST /api/config/test-teams", False, f"Exception: {str(e)}")
            return False, None
    
    def test_reload_config(self):
        """Test: Recargar configuración."""
        try:
            response = requests.post(f"{self.base_url}/api/config/reload", timeout=5)
            success = response.status_code == 200
            data = response.json() if success else None
            
            msg = data.get('message', '') if success else data.get('error', '') if data else ''
            
            self.log(
                "POST /api/config/reload",
                success,
                msg,
                response.text if not success else None
            )
            return success, data
        except Exception as e:
            self.log("POST /api/config/reload", False, f"Exception: {str(e)}")
            return False, None
    
    def run_all_tests(self):
        """Ejecutar todos los tests."""
        print("=" * 70)
        print("🧪 CONFIG MANAGER TEST SUITE")
        print("=" * 70)
        
        print("\n📋 CONFIGURATION TESTS")
        print("-" * 70)
        self.test_get_config()
        self.test_get_notifications_config()
        self.test_get_ci_config()
        self.test_get_ml_config()
        
        print("\n🏥 HEALTH TESTS")
        print("-" * 70)
        self.test_health_check()
        
        print("\n📨 NOTIFICATION TESTS")
        print("-" * 70)
        self.test_test_slack()
        time.sleep(1)  # Pequeña pausa entre tests
        self.test_test_teams()
        
        print("\n🔄 CONFIG MANAGEMENT TESTS")
        print("-" * 70)
        self.test_reload_config()
        
        # Resumen
        print("\n" + "=" * 70)
        passed = sum(1 for _, success in self.results if success)
        total = len(self.results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        print(f"📊 RESULTS: {passed}/{total} tests passed ({percentage:.1f}%)")
        print("=" * 70)
        
        return passed == total


def main():
    """Función principal."""
    print("\n🚀 Starting Config Manager Test Suite...")
    print(f"📡 Target: {BASE_URL}\n")
    
    # Verificar que el servidor esté corriendo
    try:
        response = requests.get(f"{BASE_URL}/api/config/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        print(f"   Make sure the server is running at {BASE_URL}")
        return False
    
    # Ejecutar tests
    tester = ConfigTester(BASE_URL)
    all_passed = tester.run_all_tests()
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
