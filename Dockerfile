# Imagen base
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias y DVC con soporte S3
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install dvc[s3]

# Copiar el código fuente
COPY app/ ./app/

# Copiar archivos necesarios de DVC
COPY models.dvc ./
COPY app/models/trained.dvc ./app/models/
COPY .dvc .dvc
COPY .dvcignore ./
COPY .git .git

# Copiar el entrypoint
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Exponer el puerto
EXPOSE 8000

# Ejecutar el entrypoint que hace dvc pull + inicia Uvicorn
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
