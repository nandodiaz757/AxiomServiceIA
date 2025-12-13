# import torch
# import torch.nn as nn
# import torch.optim as optim
# import sqlite3
# import json
# import numpy as np
# from SiameseEncoder import SiameseEncoder  # tu clase encoder

# # ---------------------------------------------------------------------
# # Crear base de datos de ejemplo si no existe
# # ---------------------------------------------------------------------

# DB_NAME = "accessibility.db"  # 👈 ajusta si tu base de datos tiene otro nombre


# # ---------------------------------------------------------------------
# # Función para obtener pares (árbol_A, árbol_B, etiqueta)
# # ---------------------------------------------------------------------
# def load_training_pairs(limit=200):
#     pairs = []
#     with sqlite3.connect(DB_NAME) as conn:
#         conn.row_factory = sqlite3.Row
#         rows = conn.execute("""
#             SELECT collect_node_tree, build_id, header_text
#             FROM accessibility_data
#             ORDER BY created_at DESC
#             LIMIT ?
#         """, (limit,)).fetchall()

#     for i in range(len(rows) - 1):
#         try:
#             tree_a = json.loads(rows[i]["collect_node_tree"] or "[]")
#             tree_b = json.loads(rows[i + 1]["collect_node_tree"] or "[]")
#             label = 1.0 if rows[i]["header_text"] == rows[i + 1]["header_text"] else 0.0
#             pairs.append((tree_a, tree_b, label))
#         except Exception:
#             continue

#     print(f"✅ {len(pairs)} pares cargados para entrenamiento.")
#     return pairs

# # ---------------------------------------------------------------------
# # Función de pérdida contrastiva
# # ---------------------------------------------------------------------
# def contrastive_loss(similarity, label, margin=0.5):
#     pos_loss = (1 - similarity) ** 2
#     neg_loss = torch.clamp(similarity - margin, min=0) ** 2
#     return torch.mean(label * pos_loss + (1 - label) * neg_loss)

# # ---------------------------------------------------------------------
# # Entrenamiento
# # ---------------------------------------------------------------------
# def train_model(epochs=5):
#     model = SiameseEncoder()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     training_pairs = load_training_pairs()

#     for epoch in range(epochs):
#         total_loss = 0.0
#         for (tree_a, tree_b, label) in training_pairs:
#             optimizer.zero_grad()
#             sim = model(tree_a, tree_b)
#             lbl = torch.tensor(label, dtype=torch.float32)
#             loss = contrastive_loss(sim, lbl)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"🧠 Epoch {epoch+1}/{epochs} | Loss promedio: {total_loss/len(training_pairs):.4f}")

#     model.save("ui_encoder.pt")
#     print("✅ Modelo guardado en ui_encoder.pt")

# # ---------------------------------------------------------------------
# if __name__ == "__main__":
#     train_model()

import torch
import torch.nn as nn
import torch.optim as optim
import psycopg2
import json
import numpy as np
from SiameseEncoder import SiameseEncoder  # tu clase encoder
from db import get_conn_cm

# ---------------------------------------------------------------------
# Parámetros de conexión a PostgreSQL
# ---------------------------------------------------------------------
PG_CONN_PARAMS = {
    "host": "localhost",
    "dbname": "accessibility",
    "user": "postgres",
    "password": "password",
    "port": 5432
}

# ---------------------------------------------------------------------
# Función para obtener pares (árbol_A, árbol_B, etiqueta)
# ---------------------------------------------------------------------
def load_training_pairs(limit=200):
    pairs = []
    conn = psycopg2.connect(**PG_CONN_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        SELECT collect_node_tree, build_id, header_text
        FROM accessibility_data
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()
    # rows[i] = (collect_node_tree, build_id, header_text)

    for i in range(len(rows) - 1):
        try:
            tree_a = json.loads(rows[i][0] or "[]")
            tree_b = json.loads(rows[i + 1][0] or "[]")
            label = 1.0 if rows[i][2] == rows[i + 1][2] else 0.0
            pairs.append((tree_a, tree_b, label))
        except Exception:
            continue

    cur.close()
    conn.close()

    print(f"✅ {len(pairs)} pares cargados para entrenamiento.")
    return pairs

# ---------------------------------------------------------------------
# Función de pérdida contrastiva
# ---------------------------------------------------------------------
def contrastive_loss(similarity, label, margin=0.5):
    pos_loss = (1 - similarity) ** 2
    neg_loss = torch.clamp(similarity - margin, min=0) ** 2
    return torch.mean(label * pos_loss + (1 - label) * neg_loss)

# ---------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------
def train_model(epochs=5):
    model = SiameseEncoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    training_pairs = load_training_pairs()

    for epoch in range(epochs):
        total_loss = 0.0
        for (tree_a, tree_b, label) in training_pairs:
            optimizer.zero_grad()
            sim = model(tree_a, tree_b)
            lbl = torch.tensor(label, dtype=torch.float32)
            loss = contrastive_loss(sim, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"🧠 Epoch {epoch+1}/{epochs} | Loss promedio: {total_loss/len(training_pairs):.4f}")

    model.save("ui_encoder.pt")
    print("✅ Modelo guardado en ui_encoder.pt")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
