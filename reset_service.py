import random, time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

# Estructura temporal en memoria
# email -> {"code": "123456", "timestamp": 1697052400}
reset_codes = {}



def generate_code(email: str) -> str:
    """Genera y guarda un código temporal de 6 dígitos."""
    code = str(random.randint(100000, 999999))
    reset_codes[email] = {"code": code, "timestamp": time.time()}
    return code

def validate_code(email: str, code: str, expiration_seconds: int = 300) -> bool:
    """Valida que el código exista, coincida y no haya expirado."""
    data = reset_codes.get(email)
    if not data:
        return False
    if data["code"] != code:
        return False
    if time.time() - data["timestamp"] > expiration_seconds:
        # Elimina el código si expiró
        reset_codes.pop(email, None)
        return False
    return True
