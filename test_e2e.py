import json
import sqlite3
import time
import urllib.request
from datetime import datetime

# Evento de prueba
event = {
    "actualDevice": "Pixel 6",
    "version": "8.20.0",
    "timestamp": int(time.time() * 1000),
    "eventType": 0,
    "eventTypeName": "WINDOW_STATE_CHANGED",
    "packageName": "com.grability.rappi",
    "className": "MainActivity",
    "text": "Test Screen",
    "contentDescription": "Test Description",
    "screensId": "test_screen_001",
    "screenNames": "TestScreen",
    "headerText": "Test Header - " + datetime.now().strftime("%H:%M:%S"),
    "actualDevices": "Pixel 6",
    "versions": "8.20.0",
    "actions": [],
    "collectNodeTree": [],
    "additionalInfo": {},
    "treeData": {
        "id": "root",
        "className": "android.widget.FrameLayout",
        "text": "Root",
        "children": [
            {
                "id": "btn_test",
                "className": "android.widget.Button",
                "text": "Test Button",
                "clickable": True,
                "enabled": True
            }
        ]
    },
    "session_key": "test_session_001"
}

print("=== ENVIANDO EVENTO A /collect ===")
try:
    url = 'http://127.0.0.1:8000/collect'
    data = json.dumps(event).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=10) as r:
        response = r.read().decode('utf-8')
        print(f"STATUS: {r.status}")
        print(f"RESPONSE: {response}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n=== ESPERANDO 2 SEGUNDOS ===")
time.sleep(2)

print("\n=== VERIFICANDO ÚLTIMO DIFF INSERTADO ===")
conn = sqlite3.connect('accessibility.db')
c = conn.cursor()
row = c.execute("""
    SELECT id, header_text, diff_priority, similarity_to_approved, approved_before, created_at
    FROM screen_diffs
    ORDER BY created_at DESC
    LIMIT 1
""").fetchone()
print(row)
conn.close()

if row:
    print(f"\n✅ ÚLTIMO DIFF:")
    print(f"   ID: {row[0]}")
    print(f"   Header: {row[1]}")
    print(f"   Priority: {row[2]}")
    print(f"   Similarity: {row[3]}")
    print(f"   Approved Before: {row[4]}")
    print(f"   Created: {row[5]}")
else:
    print("❌ No se encontraron diffs")
