#!/usr/bin/env python3
"""
ROLLBACK SCRIPT - Revertir cambios del sistema de retroalimentación incremental
Ejecutar: python ROLLBACK_INCREMENTAL_FEEDBACK.py
"""

import os
import sys
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

# Colores para output
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'

def log_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def log_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def log_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def log_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def log_section(title):
    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.RESET}\n")

def restore_from_backup(original_file, backup_file):
    """Restaura archivo desde backup"""
    if not os.path.exists(backup_file):
        log_warning(f"No se encontró backup: {backup_file}")
        return False
    
    try:
        shutil.copy2(backup_file, original_file)
        log_success(f"Restaurado: {original_file}")
        return True
    except Exception as e:
        log_error(f"Error restaurando {original_file}: {e}")
        return False

def remove_file(filepath):
    """Elimina un archivo de forma segura"""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            log_success(f"Eliminado: {filepath}")
            return True
        except Exception as e:
            log_error(f"Error eliminando {filepath}: {e}")
            return False
    return True

def rollback_database():
    """Revierte cambios en la base de datos"""
    log_section("REVERTIR BASE DE DATOS")
    
    db_file = "accessibility.db"
    
    if not os.path.exists(db_file):
        log_warning(f"Base de datos no encontrada: {db_file}")
        return False
    
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        
        # Verificar si existen las columnas agregadas
        c.execute("PRAGMA table_info(screen_diffs)")
        columns = {row[1] for row in c.fetchall()}
        
        log_info(f"Columnas actuales en screen_diffs: {', '.join(sorted(columns))}")
        
        # Crear tabla sin las columnas nuevas
        if 'diff_priority' in columns or 'similarity_to_approved' in columns or 'approved_before' in columns:
            log_warning("Se detectaron columnas de retroalimentación. Revertiendo...")
            
            # Crear tabla temporal sin las columnas nuevas
            c.execute("""
                CREATE TABLE screen_diffs_old AS
                SELECT 
                    id, tester_id, build_id, screen_name, header_text,
                    removed, added, modified, text_diff, diff_hash,
                    text_overlap, overlap_ratio, ui_structure_similarity, screen_status,
                    created_at, updated_at
                FROM screen_diffs
                WHERE diff_priority IS NOT NULL OR diff_priority IS NULL
            """)
            
            # Eliminar tabla vieja
            c.execute("DROP TABLE screen_diffs")
            
            # Renombrar tabla temporal
            c.execute("ALTER TABLE screen_diffs_old RENAME TO screen_diffs")
            
            log_success("Tabla screen_diffs revertida a estado original")
        else:
            log_info("Columnas de retroalimentación no encontradas - no hay cambios que revertir")
        
        # Eliminar tablas de feedback_model.db si existe
        conn.commit()
        conn.close()
        
    except Exception as e:
        log_error(f"Error revertiendo base de datos: {e}")
        return False
    
    return True

def rollback_feedback_database():
    """Elimina la base de datos de feedback"""
    log_section("ELIMINAR BD DE FEEDBACK")
    
    db_files = [
        "feedback_model.db",
        "feedback_model.db-journal"
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                log_success(f"Eliminado: {db_file}")
            except Exception as e:
                log_error(f"Error eliminando {db_file}: {e}")
                return False
    
    return True

def main():
    """Ejecuta rollback completo"""
    
    print(f"\n{Colors.RED}")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  ROLLBACK - SISTEMA DE RETROALIMENTACIÓN INCREMENTAL  ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print(Colors.RESET)
    
    log_warning("Este script revertirá TODOS los cambios realizados")
    log_warning("Se requiere confirmación del usuario")
    
    confirm = input(f"\n{Colors.YELLOW}¿Deseas continuar con el ROLLBACK? (s/n): {Colors.RESET}")
    
    if confirm.lower() != 's':
        log_info("Rollback cancelado por el usuario")
        return 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_section(f"INICIANDO ROLLBACK [{timestamp}]")
    
    total_steps = 5
    completed = 0
    
    # PASO 1: Restaurar backend.py
    log_section("PASO 1: RESTAURAR BACKEND.PY")
    backup_file = "backend.py.backup_incremental"
    if os.path.exists(backup_file):
        if restore_from_backup("backend.py", backup_file):
            completed += 1
        else:
            log_warning("Restauración manual puede requerirse")
    else:
        log_warning("No hay backup de backend.py - revisar manualmente")
    
    # PASO 2: Restaurar db_init.py
    log_section("PASO 2: RESTAURAR DB_INIT.PY")
    backup_file = "db_init.py.backup_incremental"
    if os.path.exists(backup_file):
        if restore_from_backup("db_init.py", backup_file):
            completed += 1
        else:
            log_warning("Restauración manual puede requerirse")
    else:
        log_warning("No hay backup de db_init.py - revisar manualmente")
    
    # PASO 3: Revertir cambios en base de datos
    log_section("PASO 3: REVERTIR BASE DE DATOS")
    if rollback_database():
        completed += 1
    else:
        log_warning("Revisor manual de la BD requerido")
    
    # PASO 4: Eliminar archivos nuevos
    log_section("PASO 4: ELIMINAR ARCHIVOS NUEVOS")
    new_files = [
        "incremental_feedback_system.py",
        "feedback_model.db",
        "feedback_model.db-journal"
    ]
    
    for filepath in new_files:
        remove_file(filepath)
    completed += 1
    
    # PASO 5: Limpiar caché de Python
    log_section("PASO 5: LIMPIAR CACHE DE PYTHON")
    try:
        pycache_dirs = [
            "__pycache__",
        ]
        for pycache in pycache_dirs:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)
                log_success(f"Eliminado: {pycache}")
        completed += 1
    except Exception as e:
        log_error(f"Error limpiando caché: {e}")
    
    # RESUMEN
    log_section("RESUMEN DEL ROLLBACK")
    log_success(f"Pasos completados: {completed}/{total_steps}")
    
    if completed == total_steps:
        print(f"\n{Colors.GREEN}")
        print("╔" + "="*58 + "╗")
        print("║" + "  ROLLBACK COMPLETADO EXITOSAMENTE  ".center(58) + "║")
        print("║" + " "*58 + "║")
        print("║" + f"  Timestamp: {timestamp}".ljust(58) + "║")
        print("╚" + "="*58 + "╝")
        print(Colors.RESET)
        
        log_info("Próximos pasos:")
        log_info("1. Reinicia el servidor: python server.py")
        log_info("2. Verifica logs para errores")
        log_info("3. Prueba endpoints para confirmar revert")
        
        return 0
    else:
        print(f"\n{Colors.YELLOW}")
        print("╔" + "="*58 + "╗")
        print("║" + "  ROLLBACK PARCIAL - REVISIÓN MANUAL REQUERIDA  ".center(58) + "║")
        print("║" + " "*58 + "║")
        print("║" + f"  {completed}/{total_steps} pasos completados".ljust(58) + "║")
        print("╚" + "="*58 + "╝")
        print(Colors.RESET)
        
        log_warning("Algunos archivos pueden requerir restauración manual")
        log_warning("Revisa los archivos .backup_incremental")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
