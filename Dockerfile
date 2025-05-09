# Imagen base
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY app/ ./app/

# Copiar los modelos Hugging Face descargados previamente
COPY models/ ./models/

# Exponer el puerto de la API
EXPOSE 8000

# Comando para iniciar FastAPI con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
