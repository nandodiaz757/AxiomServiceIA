"""
Endpoints para integración de automatización de pruebas
"""

from fastapi import APIRouter, Query, Body, HTTPException
from session_manager import (
    get_session_manager, 
    SessionStatus,
    init_session_manager
)
from typing import List, Dict, Optional
import logging

router = APIRouter(prefix="/api/automation", tags=["automation"])
logger = logging.getLogger(__name__)


# Inicializar session manager
def setup_automation_routes():
    """Debe ser llamado al startup de la app"""
    try:
        init_session_manager()
        logger.info("✅ Automation routes inicializadas")
    except Exception as e:
        logger.error(f"Error inicializando automation routes: {e}")


# ====================================
# ENDPOINTS DE SESIÓN
# ====================================

@router.post("/session/create")
async def create_test_session(
    test_name: str = Body(..., embed=True),
    tester_id: str = Body(..., embed=True),
    build_id: str = Body(..., embed=True),
    app_name: str = Body(..., embed=True),
    expected_flow: List[str] = Body(..., embed=True),
    metadata: Optional[Dict] = Body(None, embed=True)
):
    """
    Crea una nueva sesión de prueba automatizada.
    
    Args:
        test_name: Nombre descriptivo del test
        tester_id: ID del tester/bot de automatización
        build_id: ID del build bajo prueba
        app_name: Nombre de la aplicación
        expected_flow: Lista de pantallas esperadas en orden
        metadata: Datos adicionales (browser, ambiente, etc.)
    
    Returns:
        Información de la sesión creada
    """
    try:
        manager = get_session_manager()
        session = manager.create_session(
            test_name=test_name,
            tester_id=tester_id,
            build_id=build_id,
            app_name=app_name,
            expected_flow=expected_flow,
            metadata=metadata
        )

        return {
            "session_id": session.session_id,
            "test_name": session.test_name,
            "status": session.status.value,
            "expected_flow": session.expected_flow,
            "created_at": session.created_at,
            "message": "✅ Sesión creada exitosamente"
        }

    except Exception as e:
        logger.error(f"Error creando sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/start")
async def start_test_session(session_id: str):
    """Inicia una sesión de prueba (la marca como RUNNING)"""
    try:
        manager = get_session_manager()
        success = manager.start_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")

        return {
            "session_id": session_id,
            "status": SessionStatus.RUNNING.value,
            "message": "▶️ Sesión iniciada"
        }

    except Exception as e:
        logger.error(f"Error iniciando sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/event")
async def record_session_event(
    session_id: str,
    screen_name: str = Body(..., embed=True),
    header_text: str = Body("", embed=True),
    event_type: str = Body("screen_change", embed=True),
    additional_data: Optional[Dict] = Body(None, embed=True)
):
    """
    Registra un evento (cambio de pantalla) en la sesión.
    Valida automáticamente contra el flujo esperado.
    
    Args:
        session_id: ID de la sesión
        screen_name: Nombre/ID de la pantalla
        header_text: Texto del header/título
        event_type: Tipo de evento
        additional_data: Datos adicionales del evento
    
    Returns:
        Resultado de validación del evento
    """
    try:
        manager = get_session_manager()
        success, validation_result, message = await manager.process_event(
            session_id=session_id,
            screen_name=screen_name,
            header_text=header_text,
            event_type=event_type,
            additional_data=additional_data
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return {
            "session_id": session_id,
            "screen_name": screen_name,
            "validation_result": validation_result.value,
            "message": message,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error registrando evento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/validation")
async def add_session_validation(
    session_id: str,
    validation_name: str = Body(..., embed=True),
    rule: Dict = Body(..., embed=True),
    passed: bool = Body(..., embed=True),
    error_message: Optional[str] = Body(None, embed=True)
):
    """
    Registra una validación (assertion) adicional en la sesión.
    
    Args:
        session_id: ID de la sesión
        validation_name: Nombre de la validación
        rule: Diccionario con criterios
        passed: Si la validación pasó
        error_message: Mensaje de error si falló
    
    Returns:
        Confirmación de registro
    """
    try:
        manager = get_session_manager()
        success = manager.add_validation(
            session_id=session_id,
            validation_name=validation_name,
            rule=rule,
            passed=passed,
            error_message=error_message
        )

        if not success:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")

        return {
            "session_id": session_id,
            "validation_name": validation_name,
            "passed": passed,
            "message": f"✓ Validación '{validation_name}' registrada",
            "success": True
        }

    except Exception as e:
        logger.error(f"Error agregando validación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/end")
async def end_test_session(
    session_id: str,
    success: bool = Body(..., embed=True),
    final_status: str = Body("completed", embed=True)
):
    """
    Finaliza una sesión y genera reporte.
    
    Args:
        session_id: ID de la sesión
        success: Si el test finalizó exitosamente
        final_status: Estado final (completed, failed, error)
    
    Returns:
        Reporte de la sesión
    """
    try:
        manager = get_session_manager()
        
        # Convertir string a enum
        status_map = {
            "completed": SessionStatus.COMPLETED,
            "failed": SessionStatus.FAILED,
            "error": SessionStatus.ERROR
        }
        status = status_map.get(final_status, SessionStatus.COMPLETED)

        report = manager.end_session(session_id, status)

        if not report:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")

        return report

    except Exception as e:
        logger.error(f"Error finalizando sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================
# ENDPOINTS DE CONSULTA
# ====================================

@router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Obtiene el estado actual de una sesión"""
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")

        return session.to_dict()

    except Exception as e:
        logger.error(f"Error obteniendo sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_test_sessions(
    status: Optional[str] = Query(None),
    tester_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Lista sesiones con filtros opcionales.
    
    Query Parameters:
        status: Filtrar por estado (created, running, passed, failed, etc.)
        tester_id: Filtrar por ID del tester
        limit: Límite de resultados (máximo 500)
    
    Returns:
        Lista de sesiones
    """
    try:
        manager = get_session_manager()
        
        # Convertir string a enum si se proporciona
        status_enum = None
        if status:
            try:
                status_enum = SessionStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Estado inválido: {status}")

        sessions = manager.list_sessions(
            status=status_enum,
            tester_id=tester_id,
            limit=limit
        )

        return {
            "total": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        logger.error(f"Error listando sesiones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/report")
async def get_session_report(session_id: str):
    """Obtiene el reporte completo de una sesión"""
    try:
        manager = get_session_manager()
        report = manager.get_session_report(session_id)

        if not report:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")

        return report

    except Exception as e:
        logger.error(f"Error obteniendo reporte: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================
# ENDPOINTS DE MANTENIMIENTO
# ====================================

@router.post("/cleanup/expired")
async def cleanup_expired_sessions(max_age_hours: int = Query(24, ge=1)):
    """
    Limpia sesiones antiguas sin cerrar.
    
    Query Parameters:
        max_age_hours: Edad máxima en horas (default 24)
    
    Returns:
        Número de sesiones limpias
    """
    try:
        manager = get_session_manager()
        count = manager.cleanup_expired_sessions(max_age_hours)

        return {
            "cleaned": count,
            "message": f"🧹 {count} sesiones expiradas limpias"
        }

    except Exception as e:
        logger.error(f"Error en limpieza: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_automation_stats():
    """Obtiene estadísticas generales del sistema de automatización"""
    try:
        manager = get_session_manager()
        
        all_sessions = manager.sessions
        
        stats = {
            "total_sessions": len(all_sessions),
            "active_sessions": sum(1 for s in all_sessions.values() if s.status == SessionStatus.RUNNING),
            "completed_sessions": sum(1 for s in all_sessions.values() if s.status == SessionStatus.COMPLETED),
            "failed_sessions": sum(1 for s in all_sessions.values() if s.status == SessionStatus.FAILED),
            "total_events": sum(s.events_received for s in all_sessions.values()),
            "avg_flow_completion": (
                sum(s.flow_position / len(s.expected_flow) * 100 
                    for s in all_sessions.values() 
                    if s.expected_flow) / len(all_sessions)
                if all_sessions else 0
            )
        }

        return stats

    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
