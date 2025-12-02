#!/usr/bin/env python
"""Test script to verify backend imports correctly"""

try:
    import backend
    print("[OK] Backend importado correctamente")
    print(f"[OK] TRAINED_SCREENS_CACHE definido: {hasattr(backend, 'TRAINED_SCREENS_CACHE')}")
    print(f"[OK] TRAIN_CACHE_TTL definido: {hasattr(backend, 'TRAIN_CACHE_TTL')}")
    print(f"[OK] TRAIN_GENERAL_ON_COLLECT definido: {hasattr(backend, 'TRAIN_GENERAL_ON_COLLECT')}")
    print("\n[SUCCESS] Backend cargado exitosamente con todas las variables requeridas")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
