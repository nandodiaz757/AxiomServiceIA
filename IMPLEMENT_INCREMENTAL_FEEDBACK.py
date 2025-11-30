#!/usr/bin/env python3
"""
Script de implementación segura del sistema de retroalimentación incremental
Realiza cambios precisos en backend.py y db_init.py con validaciones
"""

import os
import sys
import re
from pathlib import Path

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

def modify_backend_analyze_and_train():
    """Agregar lógica de retroalimentación en analyze_and_train"""
    log_section("MODIFICAR ANALYZE_AND_TRAIN")
    
    backend_file = "backend.py"
    
    with open(backend_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar el patrón: has_changes = True (después de except)
    pattern = r"(        except Exception as e:\s+logger\.error\(f\"Error comparando árboles: \{e\}\"\)\s+has_changes = True)\s+(    else:\s+has_changes = True)"
    
    replacement = r"""\1
            mark_as_low_priority = False  # 🔹 NUEVA LÍNEA
            similarity_to_approved = 0.0   # 🔹 NUEVA LÍNEA

\2
        mark_as_low_priority = False  # 🔹 NUEVA LÍNEA
        similarity_to_approved = 0.0   # 🔹 NUEVA LÍNEA"""
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        log_error("No se encontró el patrón a reemplazar")
        return False
    
    # Ahora agregar la lógica antes del except
    pattern2 = r"(                or diff_result\.get\(\"has_changes\"\)\s+\))"
    
    replacement2 = r"""\1
            
            # 🔹 NUEVA SECCIÓN: Retroalimentación Incremental
            mark_as_low_priority = False
            similarity_to_approved = 0.0
            
            if has_changes and feedback_system is not None:
                try:
                    approval_info = check_approved_diff_pattern(
                        diff_signature=diff_signature,
                        app_name=app_name,
                        tester_id=t_id,
                        feedback_system=feedback_system
                    )
                    logger.info(f"📊 Análisis de retroalimentación: {approval_info['reason']}")
                    
                    # Guardar el estado para usar al insertar
                    mark_as_low_priority = not approval_info['should_show']
                    similarity_to_approved = approval_info['similarity_score']
                except Exception as e:
                    logger.warning(f"⚠️  Error en check_approved_diff_pattern: {e}")
                    mark_as_low_priority = False
                    similarity_to_approved = 0.0"""
    
    new_content = re.sub(pattern2, replacement2, new_content)
    
    # Guardar cambios
    with open(backend_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    log_success("Lógica de retroalimentación agregada a analyze_and_train")
    return True

def modify_backend_insert_screen_diffs():
    """Agregar columnas de retroalimentación al INSERT screen_diffs"""
    log_section("MODIFICAR INSERT SCREEN_DIFFS")
    
    backend_file = "backend.py"
    
    with open(backend_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar el INSERT original
    pattern = r"""(                cur\.execute\(\"\"\"\s+INSERT INTO screen_diffs \(\s+tester_id, build_id, screen_name, header_text,\s+removed, added, modified, text_diff, diff_hash,\s+text_overlap, overlap_ratio, ui_structure_similarity, screen_status\s+\)\s+VALUES \(\?,\?,\?,\?,\?,\?,\?,\?,\?,\?,\?,\?,\?\))"""
    
    replacement = r"""\1"""
    
    # Buscar dónde termina este INSERT
    pattern_insert_complete = r"""(text_overlap, overlap_ratio,\s+ui_sim, screen_status\s+\)\))"""
    
    replacement_insert = r"""text_overlap, overlap_ratio,
                    ui_sim, screen_status,
                    mark_as_low_priority, similarity_to_approved
                ))"""
    
    # Lo haré más simple: buscar la línea del INSERT VALUES
    pattern_simple = r"(VALUES \(\?,\?,\?,\?,\?,\?,\?,\?,\?,\?,\?,\?,\?\))"
    replacement_simple = r"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    
    new_content = re.sub(pattern_simple, replacement_simple, content)
    
    # Ahora agregar las columnas al INSERT
    pattern_cols = r"""(INSERT INTO screen_diffs \(\s+tester_id, build_id, screen_name, header_text,\s+removed, added, modified, text_diff, diff_hash,\s+text_overlap, overlap_ratio, ui_structure_similarity, screen_status)"""
    
    replacement_cols = r"""\1,
                        diff_priority, similarity_to_approved"""
    
    new_content = re.sub(pattern_cols, replacement_cols, new_content)
    
    if new_content == content:
        log_warning("Patrones de INSERT no encontrados exactamente, requiere manual review")
        return False
    
    with open(backend_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    log_success("Columnas de retroalimentación agregadas al INSERT screen_diffs")
    return True

def add_feedback_endpoints():
    """Agregar nuevos endpoints para approve/reject"""
    log_section("AGREGAR ENDPOINTS DE FEEDBACK")
    
    backend_file = "backend.py"
    
    endpoints_code = '''
# ============================================================
# 🔹 NUEVOS ENDPOINTS: SISTEMA DE RETROALIMENTACIÓN
# ============================================================

@app.post("/diff/{diff_id}/approve")
async def approve_diff(diff_id: int):
    """
    Registra aprobación de un diff.
    El modelo aprenderá a no mostrar similares en el futuro.
    """
    global feedback_system
    if feedback_system is None:
        return {"error": "Feedback system not initialized", "status": 503}
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            
            # Obtener detalles del diff
            c.execute("""
                SELECT diff_hash, tester_id, app_name, build_id, screen_name
                FROM screen_diffs WHERE id = ? LIMIT 1
            """, (diff_id,))
            
            diff_row = c.fetchone()
            if not diff_row:
                return {"error": "Diff not found", "status": 404}
            
            diff_hash, tester, app, build, screen = diff_row
            
            # Registrar aprobación en feedback system
            record_diff_decision(
                diff_hash=diff_hash,
                diff_signature=diff_hash,
                app_name=app,
                tester_id=tester,
                build_version=build,
                decision='approved',
                user_approved=True,
                feedback_system=feedback_system
            )
            
            # Actualizar BD
            c.execute("""
                UPDATE screen_diffs 
                SET diff_priority = 'low', approved_before = 1
                WHERE id = ?
            """, (diff_id,))
            
            conn.commit()
        
        logger.info(f"✅ Diff {diff_id} aprobado por tester {tester}")
        return {
            "success": True,
            "message": f"Diff approved - modelo mejorado",
            "diff_id": diff_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error aprobando diff: {e}")
        return {"error": str(e), "status": 500}


@app.post("/diff/{diff_id}/reject")
async def reject_diff(diff_id: int):
    """
    Registra rechazo de un diff (falso positivo).
    El modelo aprenderá a no mostrar similares.
    """
    global feedback_system
    if feedback_system is None:
        return {"error": "Feedback system not initialized", "status": 503}
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            
            c.execute("""
                SELECT diff_hash, tester_id, app_name, build_id
                FROM screen_diffs WHERE id = ? LIMIT 1
            """, (diff_id,))
            
            diff_row = c.fetchone()
            if not diff_row:
                return {"error": "Diff not found", "status": 404}
            
            diff_hash, tester, app, build = diff_row
            
            # Registrar rechazo
            record_diff_decision(
                diff_hash=diff_hash,
                diff_signature=diff_hash,
                app_name=app,
                tester_id=tester,
                build_version=build,
                decision='rejected',
                user_approved=False,
                feedback_system=feedback_system
            )
            
            # Marcar como baja prioridad (falso positivo)
            c.execute("""
                UPDATE screen_diffs 
                SET diff_priority = 'low'
                WHERE id = ?
            """, (diff_id,))
            
            conn.commit()
        
        logger.info(f"❌ Diff {diff_id} rechazado por tester {tester} - falso positivo")
        return {
            "success": True,
            "message": "Diff marcado como falso positivo",
            "diff_id": diff_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error rechazando diff: {e}")
        return {"error": str(e), "status": 500}


@app.get("/learning-insights/{app_name}/{tester_id}")
async def get_learning_insights(app_name: str, tester_id: str):
    """
    Retorna métricas de aprendizaje del modelo para un tester.
    Muestra cómo está mejorando la precisión.
    """
    global feedback_system
    if feedback_system is None:
        return {"error": "Feedback system not initialized", "status": 503}
    
    try:
        insights = feedback_system.get_learning_insights(app_name, tester_id)
        return {
            "app_name": app_name,
            "tester_id": tester_id,
            "insights": insights,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo insights: {e}")
        return {"error": str(e), "status": 500}
'''
    
    with open(backend_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar el final del archivo para agregar endpoints
    if "NUEVOS ENDPOINTS: SISTEMA DE RETROALIMENTACIÓN" in content:
        log_info("Endpoints ya existen, saltando...")
        return True
    
    # Agregar antes del último return o final del archivo
    # Buscar un buen lugar: después de todos los @app routes, antes de la configuración de pruebas
    insert_point = content.rfind("if __name__")
    
    if insert_point != -1:
        new_content = content[:insert_point] + endpoints_code + "\n\n" + content[insert_point:]
    else:
        new_content = content + endpoints_code
    
    with open(backend_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    log_success("Endpoints de feedback agregados")
    return True

def modify_db_init():
    """Agregar ALTER TABLE para screen_diffs"""
    log_section("MODIFICAR DB_INIT.PY")
    
    db_init_file = "db_init.py"
    
    with open(db_init_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar después de CREATE TABLE screen_diffs
    pattern = r"(c\.execute\(\"\"\"\s+CREATE TABLE screen_diffs.*?\)\"\"\"[^)]*\).*?)(\n\s+c\.execute\(\"\"\"\s+CREATE TABLE)"
    
    # Más simple: buscar CREATE TABLE screen_diffs y agregar después
    if "ALTER TABLE screen_diffs ADD COLUMN IF NOT EXISTS diff_priority" not in content:
        alter_statements = """
    # 🔹 Agregar columnas para retroalimentación incremental
    c.execute('''
        ALTER TABLE screen_diffs ADD COLUMN IF NOT EXISTS
        diff_priority TEXT DEFAULT 'high'
    ''')
    
    c.execute('''
        ALTER TABLE screen_diffs ADD COLUMN IF NOT EXISTS
        similarity_to_approved REAL DEFAULT 0.0
    ''')
    
    c.execute('''
        ALTER TABLE screen_diffs ADD COLUMN IF NOT EXISTS
        approved_before INTEGER DEFAULT 0
    ''')
"""
        
        # Buscar línea de commit
        commit_pattern = r"(    conn\.commit\(\))"
        new_content = re.sub(commit_pattern, alter_statements + r"\n\1", content)
        
        if new_content == content:
            log_warning("No se encontró patrón de commit en db_init.py")
            return False
        
        with open(db_init_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        log_success("Columnas de retroalimentación agregadas a screen_diffs")
        return True
    else:
        log_info("Columnas de retroalimentación ya existen")
        return True

def validate_syntax():
    """Validar sintaxis de Python en archivos modificados"""
    log_section("VALIDAR SINTAXIS")
    
    files_to_check = ["backend.py", "db_init.py"]
    
    for filepath in files_to_check:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, filepath, 'exec')
            log_success(f"✓ {filepath} - Sintaxis correcta")
        except SyntaxError as e:
            log_error(f"✗ {filepath} - Error de sintaxis: {e}")
            return False
    
    return True

def main():
    print(f"\n{Colors.RED}")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  IMPLEMENTACIÓN - RETROALIMENTACIÓN INCREMENTAL  ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print(Colors.RESET)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("backend.py"):
        log_error("backend.py no encontrado en el directorio actual")
        return 1
    
    steps = [
        ("Agregar lógica de retroalimentación en analyze_and_train", modify_backend_analyze_and_train),
        ("Modificar INSERT screen_diffs", modify_backend_insert_screen_diffs),
        ("Agregar endpoints de feedback", add_feedback_endpoints),
        ("Modificar db_init.py con ALTER TABLE", modify_db_init),
        ("Validar sintaxis", validate_syntax),
    ]
    
    completed = 0
    for step_name, step_func in steps:
        log_section(step_name)
        try:
            if step_func():
                completed += 1
            else:
                log_warning(f"Paso parcialmente completado: {step_name}")
        except Exception as e:
            log_error(f"Error en {step_name}: {e}")
            return 1
    
    # Resumen final
    log_section("RESUMEN DE IMPLEMENTACIÓN")
    log_success(f"Pasos completados: {completed}/{len(steps)}")
    
    if completed == len(steps):
        print(f"\n{Colors.GREEN}")
        print("╔" + "="*58 + "╗")
        print("║" + "  IMPLEMENTACIÓN EXITOSA  ".center(58) + "║")
        print("║" + " "*58 + "║")
        print("║" + "  Sistema de retroalimentación iniciado  ".center(58) + "║")
        print("╚" + "="*58 + "╝")
        print(Colors.RESET)
        
        log_info("\nPróximos pasos:")
        log_info("1. python server.py  (reiniciar servidor)")
        log_info("2. Verifica logs para confirmar carga del sistema")
        log_info("3. En caso de error: python ROLLBACK_INCREMENTAL_FEEDBACK.py")
        
        return 0
    else:
        log_error("Implementación parcial - requiere revisión manual")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
