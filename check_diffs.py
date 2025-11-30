import sqlite3, json
print('=== PRAGMA table_info(screen_diffs) ===')
conn = sqlite3.connect('accessibility.db')
c = conn.cursor()
cols = c.execute("PRAGMA table_info(screen_diffs)").fetchall()
for col in cols:
    print(col)
print('\n=== ROW id=1 ===')
row = c.execute("SELECT id, diff_hash, diff_priority, similarity_to_approved, approved_before, created_at FROM screen_diffs WHERE id=1").fetchone()
print(row)
print('\n=== LAST 5 DIFFS ===')
rows = c.execute("SELECT id, diff_hash, diff_priority, similarity_to_approved, approved_before, created_at, header_text FROM screen_diffs ORDER BY created_at DESC LIMIT 5").fetchall()
for r in rows:
    print(r)
conn.close()
