from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from db import get_conn_cm, init_db 
import bcrypt


routers = APIRouter()
templates = Jinja2Templates(directory="templates")


@routers.get("/login", response_class=HTMLResponse)
def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@routers.post("/login", response_class=HTMLResponse)
def post_login(request: Request, username: str = Form(...), password: str = Form(...)):

    # Buscar usuario por email
    with get_conn_cm() as conn:
        with conn.cursor() as c:
            c.execute("SELECT id, email, hash_password, udid FROM usuarios WHERE email=%s;", (username,))
            user = c.fetchone()

    if not user:
        return HTMLResponse(
            "<h3>Usuario no encontrado</h3><a href='/login'>Volver</a>"
        )

    stored_hash = user[2]  # hash_password
    if not bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return HTMLResponse(
            "<h3>Contraseña incorrecta</h3><a href='/login'>Volver</a>"
        )

    # Autenticación válida
    token = "JWT-FAKE-TOKEN"
    user_id = user[0]
    email_id = user[1]
    udid = user[3]

    # Renderizar el dashboard de reportes
    return templates.TemplateResponse(
        "reportes.html",
        {
            "request": request,
            "username": username,
            "testers": [email_id],   # Solo tu propio dashboard
            "token": token,
            "user_id": user_id,
            "udid": udid  
        }
    )