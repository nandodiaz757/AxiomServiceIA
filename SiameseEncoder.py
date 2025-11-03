import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim  # 👈 importante para usar desde fuera

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    # ----------------------------------------------------------
    # Conversión de árbol UI a vector fijo de features
    # ----------------------------------------------------------
    def tree_to_vector(self, ui_tree):
        """
        Convierte un árbol de accesibilidad (lista de nodos JSON)
        en un vector numérico simple, tolerante a None o tipos incorrectos.
        """
        if not ui_tree or not isinstance(ui_tree, list):
            return np.zeros(128, dtype=np.float32)

        features = []
        for node in ui_tree[:50]:
            if not isinstance(node, dict):
                # intenta parsear si es string JSON
                if isinstance(node, str):
                    try:
                        node = json.loads(node)
                        if not isinstance(node, dict):
                            continue
                    except Exception:
                        continue
                else:
                    continue  # ignora nodos inválidos

            cls = node.get("className", "") or ""
            txt = node.get("text", "") or ""
            clickable = 1.0 if node.get("clickable") else 0.0
            enabled = 1.0 if node.get("enabled", True) else 0.0

            bounds = node.get("bounds") or {}
            width = float(bounds.get("width", 0) or 0)
            height = float(bounds.get("height", 0) or 0)
            size = np.log1p(width * height) / 10.0

            text_hash = (sum(ord(c) for c in txt[:10]) % 1000) / 1000.0
            cls_hash = (sum(ord(c) for c in cls[:10]) % 1000) / 1000.0

            node_vec = [clickable, enabled, size, text_hash, cls_hash]
            features.append(node_vec)

        if not features:
            return np.zeros(128, dtype=np.float32)

        flat = np.array(features, dtype=np.float32).flatten()
        if len(flat) < 128:
            pad = np.zeros(128 - len(flat), dtype=np.float32)
            flat = np.concatenate([flat, pad])
        else:
            flat = flat[:128]

        return flat


    # ----------------------------------------------------------
    # Encodea un árbol a embedding
    # ----------------------------------------------------------
    # def encode_tree(self, ui_tree):
    #     vec = torch.tensor(self.tree_to_vector(ui_tree), dtype=torch.float32)
    #     with torch.no_grad():
    #         emb = self.encoder(vec)
    #         emb = F.normalize(emb, p=2, dim=0)  # normaliza L2
    #     return emb
    
    def encode_batch(self, trees: list):
        vecs = [self.tree_to_vector(t) for t in trees]
        vecs = torch.tensor(vecs, dtype=torch.float32)
        with torch.no_grad():
            emb = self.encoder(vecs)
            norms = emb.norm(p=2, dim=1, keepdim=True)
            emb = torch.where(norms == 0, torch.zeros_like(emb), emb / norms)
        return emb
    
    def encode_tree(self, ui_tree):
        vec = torch.tensor(self.tree_to_vector(ui_tree), dtype=torch.float32)

        with torch.no_grad():
            emb = self.encoder(vec)
            norm = emb.norm(p=2)

            # Evita divisiones por cero o NaN
            if norm.item() == 0 or torch.isnan(norm):
                emb = torch.zeros_like(emb)
            else:
                emb = emb / norm

            emb = emb.unsqueeze(0)  # siempre 2D: (1, embedding_dim)

        return emb
    # ----------------------------------------------------------
    # Forward siamés: compara dos árboles y devuelve similitud
    # ----------------------------------------------------------
    def forward(self, tree_a, tree_b):
        va = torch.tensor(self.tree_to_vector(tree_a), dtype=torch.float32)
        vb = torch.tensor(self.tree_to_vector(tree_b), dtype=torch.float32)
        ea = F.normalize(self.encoder(va), p=2, dim=0)
        eb = F.normalize(self.encoder(vb), p=2, dim=0)
        sim = F.cosine_similarity(ea, eb, dim=0)
        return sim

    # ----------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------
    def save(self, path="ui_encoder.pt"):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path="ui_encoder.pt"):
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
