from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Base de datos en memoria (simulada)
eventos = []

class Evento(BaseModel):
    usuario: str
    evento: str

@app.post("/events")
def registrar_evento(evento: Evento):
    eventos.append(evento.dict())
    return {"mensaje": "Evento registrado", "total_eventos": len(eventos)}

@app.get("/recommend/{usuario}")
def recomendar(usuario: str):
    # Ejemplo sencillo: devuelve los últimos 3 eventos del usuario
    recomendacion = [e["evento"] for e in eventos if e["usuario"] == usuario][-3:]
    return {"usuario": usuario, "recomendacion": recomendacion}
