import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.models.predictor import evaluate_joke
from app.models.loader import list_available_models

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Montar archivos estáticos
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

# Cargar templates HTML
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
# Permitir que el frontend en otro origen pueda llamar a la API (útil en dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class JokeRequest(BaseModel):
    joke: str
    model: str

@app.post("/rate-joke")
def rate_joke(payload: JokeRequest):
    try:
        return evaluate_joke(payload.joke, payload.model)
    except Exception as e:
        return {"error": str(e)}

@app.get("/models")
def get_models():
    return list_available_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
