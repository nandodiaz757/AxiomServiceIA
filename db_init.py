import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Modelo de embeddings liviano
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_diff_features(diff_row: dict) -> np.ndarray:
    """
    Genera un vector de features a partir de un registro de screen_diffs
    con bounding boxes, ratios por clase y embeddings semánticos.
    """
    removed = diff_row.get("removed", [])
    added   = diff_row.get("added", [])
    modified= diff_row.get("modified", [])

    # Ratios por tipo de cambio
    total_nodes = max(1, len(removed) + len(added) + len(modified))
    ratio_removed  = len(removed)  / total_nodes
    ratio_added    = len(added)    / total_nodes
    ratio_modified = len(modified) / total_nodes

    # Promedio de áreas de bounding boxes
    def avg_area(nodes):
        if not nodes: return 0.0
        areas = []
        for n in nodes:
            bbox = n.get("bounds")  # [x1, y1, x2, y2]
            if bbox and len(bbox) == 4:
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                areas.append(w * h)
        return np.mean(areas) if areas else 0.0

    avg_removed_area  = avg_area(removed)
    avg_added_area    = avg_area(added)
    avg_modified_area = avg_area(modified)

    # Embedding semántico de texto
    text_concat = " ".join([
        n.get("text", "") or n.get("role", "")
        for n in removed + added + modified
    ])
    text_embedding = _embedder.encode(text_concat)

    numeric_feats = np.array([
        ratio_removed, ratio_added, ratio_modified,
        avg_removed_area, avg_added_area, avg_modified_area
    ])
    return np.concatenate([numeric_feats, text_embedding])
