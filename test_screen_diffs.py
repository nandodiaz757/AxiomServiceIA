#!/usr/bin/env python3
"""
Test script para validar el endpoint /screen/diffs mejorado
Ejecutar después de iniciar el servidor backend
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
ENDPOINT = "/screen/diffs"

def test_endpoint(params=None, test_name="Test"):
    """Ejecuta un test del endpoint y valida la respuesta"""
    print(f"\n{'='*70}")
    print(f"🧪 {test_name}")
    print(f"{'='*70}")
    
    try:
        url = f"{BASE_URL}{ENDPOINT}"
        print(f"GET {url}")
        if params:
            print(f"Params: {params}")
        
        response = requests.get(url, params=params, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return False
        
        data = response.json()
        
        # Validar estructura
        if "screen_diffs" not in data:
            print("❌ Missing 'screen_diffs' key")
            return False
        
        print(f"✅ Response structure valid")
        
        # Validar metadata si existe
        if "metadata" in data:
            metadata = data["metadata"]
            print(f"\n📊 Metadata:")
            print(f"  - Pending: {metadata.get('pending', 'N/A')}")
            print(f"  - Approved: {metadata.get('approved', 'N/A')}")
            print(f"  - Rejected: {metadata.get('rejected', 'N/A')}")
            print(f"  - Total diffs: {metadata.get('total_diffs', 'N/A')}")
            print(f"  - Total changes: {metadata.get('total_changes', 'N/A')}")
            print(f"  - Has changes: {metadata.get('has_changes', 'N/A')}")
        
        # Validar filtros si existen
        if "request_filters" in data:
            filters = data["request_filters"]
            print(f"\n🔧 Applied Filters:")
            print(f"  - only_pending: {filters.get('only_pending', 'N/A')}")
            print(f"  - only_approved: {filters.get('only_approved', 'N/A')}")
            print(f"  - only_rejected: {filters.get('only_rejected', 'N/A')}")
        
        # Validar primer diff si existe
        diffs = data.get("screen_diffs", [])
        if diffs:
            first_diff = diffs[0]
            print(f"\n📋 First Diff Sample:")
            print(f"  - ID: {first_diff.get('id', 'N/A')}")
            print(f"  - Screen: {first_diff.get('screen_name', 'N/A')}")
            print(f"  - Created: {first_diff.get('created_at', 'N/A')}")
            print(f"  - Has changes: {first_diff.get('has_changes', 'N/A')}")
            
            # Validar object 'approval'
            if "approval" in first_diff:
                approval = first_diff["approval"]
                print(f"\n✨ Approval Object (NUEVO):")
                print(f"  - Status: {approval.get('status', 'N/A')}")
                print(f"  - Is pending: {approval.get('is_pending', 'N/A')}")
                print(f"  - Approved at: {approval.get('approved_at', 'N/A')}")
                print(f"  - Rejected at: {approval.get('rejected_at', 'N/A')}")
                print(f"  - Rejection reason: {approval.get('rejection_reason', 'N/A')}")
            else:
                print(f"❌ Missing 'approval' object in diff")
                return False
            
            # Validar detailed_changes
            detailed = first_diff.get("detailed_changes", [])
            print(f"\n📝 Detailed Changes: {len(detailed)} items")
            if detailed:
                change = detailed[0]
                print(f"  - Sample action: {change.get('action', 'N/A')}")
                print(f"  - Sample attribute: {change.get('attribute', 'N/A')}")
                print(f"  - Sample node_class: {change.get('node_class', 'N/A')}")
            
            # Validar NO hay emojis
            response_str = json.dumps(data)
            emoji_chars = ['🗑️', '🆕', '✏️', '⚠️', '✅', '🧩', '🆕']
            has_emoji = any(emoji in response_str for emoji in emoji_chars)
            if has_emoji:
                print(f"\n❌ ALERT: Response contains emoji characters!")
                return False
            else:
                print(f"\n✅ No emoji characters found in response")
        
        else:
            print(f"⚠️  No diffs found (might be expected)")
        
        print(f"\n✅ Test PASSED")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error. Is server running on {BASE_URL}?")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print(f"\n🚀 SCREEN_DIFFS Endpoint Validation")
    print(f"Started at: {datetime.now().isoformat()}")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Default (only_pending=True)
    tests_total += 1
    if test_endpoint(None, "Test 1: Default (pending)"):
        tests_passed += 1
    
    # Test 2: Only pending explicit
    tests_total += 1
    if test_endpoint({"only_pending": True}, "Test 2: only_pending=True"):
        tests_passed += 1
    
    # Test 3: Only approved
    tests_total += 1
    if test_endpoint({"only_pending": False, "only_approved": True}, "Test 3: only_approved=True"):
        tests_passed += 1
    
    # Test 4: Only rejected
    tests_total += 1
    if test_endpoint({"only_pending": False, "only_rejected": True}, "Test 4: only_rejected=True"):
        tests_passed += 1
    
    # Test 5: Combined filters
    tests_total += 1
    if test_endpoint({"only_pending": False, "only_approved": True, "only_rejected": False}, 
                     "Test 5: Multiple filters"):
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print(f"Success rate: {(tests_passed/tests_total)*100:.1f}%")
    print(f"Completed at: {datetime.now().isoformat()}")
    
    if tests_passed == tests_total:
        print(f"\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
