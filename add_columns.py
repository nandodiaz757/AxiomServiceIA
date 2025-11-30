import sqlite3
print('=== Agregando columnas nuevas a screen_diffs ===')
conn = sqlite3.connect('accessibility.db')
c = conn.cursor()

try:
    c.execute("ALTER TABLE screen_diffs ADD COLUMN diff_priority TEXT DEFAULT 'high'")
    print('✅ Columna diff_priority agregada')
except Exception as e:
    print(f'⚠️  diff_priority: {e}')

try:
    c.execute("ALTER TABLE screen_diffs ADD COLUMN similarity_to_approved REAL DEFAULT 0.0")
    print('✅ Columna similarity_to_approved agregada')
except Exception as e:
    print(f'⚠️  similarity_to_approved: {e}')

try:
    c.execute("ALTER TABLE screen_diffs ADD COLUMN approved_before INTEGER DEFAULT 0")
    print('✅ Columna approved_before agregada')
except Exception as e:
    print(f'⚠️  approved_before: {e}')

conn.commit()
conn.close()

# Verificar
conn = sqlite3.connect('accessibility.db')
c = conn.cursor()
print('\n=== Verificación post-ALTER ===')
cols = c.execute("PRAGMA table_info(screen_diffs)").fetchall()
for col in cols[-5:]:  # últimas 5 columnas
    print(col)
conn.close()
