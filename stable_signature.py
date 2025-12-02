# stable_signature.py
import hashlib
import json
from typing import List, Dict

SAFE_KEYS = ["className", "text", "desc", "viewId", "pkg"]

def normalize_node(node: Dict) -> Dict:
    """Filtra solo las claves estables y convierte None en cadena vacía."""
    return {k: (node.get(k) or "") for k in SAFE_KEYS}

# def normalize_tree(nodes: List[Dict]) -> List[Dict]:
#     """Normaliza y ordena la lista de nodos para que el orden no afecte el hash."""
#     normalized = [normalize_node(n) for n in nodes]
#     return sorted(normalized, key=lambda n: (n["className"], n["text"]))

def normalize_tree(nodes: List[Dict]) -> List[Dict]:
    """Normaliza y ordena la lista de nodos para que el orden no afecte el hash."""
    normalized = [normalize_node(n) for n in nodes]

    return sorted(
        normalized,
        key=lambda n: (
            n.get("className") or "",
            n.get("text") or ""
        )
    )

def stable_signature(nodes: List[Dict]) -> str:
    """Genera un hash estable del árbol normalizado."""
    norm = normalize_tree(nodes)
    return hashlib.sha256(json.dumps(norm, sort_keys=True, ensure_ascii=False).encode()).hexdigest()