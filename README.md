# 🧠 JokeMeter

Let the AI roast your sense of humor 🤡  
JokeMeter es una aplicación web donde puedes escribir un chiste y recibir una puntuación del 1 al 5, generada por un modelo de inteligencia artificial.

---

## ✨ Características

- 🎨 Interfaz moderna, responsiva y clara
- 📤 Entrada de texto para chistes
- 🤖 Selección de modelos IA (RoBERTA, Claude, LLama - Qwen)
- 🔄 Animación de puntuación circular
- 😆 Emoji y feedback dinámico según el resultado
- ⭐ Sección emergente de “Rockstars” (creadores del modelo)
- 🐳 Contenedor Docker listo para producción o desarrollo

## Instrucciones de uso

- Ingresar al enlace de la aplicación.
- Despues de cargada la aplicación, ingresar el texto a clasificar.
- Selecciona con que modelo, o conjunto de modelos se quiere realizar la clasificación.
- Dar click en "Calificar Chiste".

---

## 🚀 Cómo ejecutar localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/jokemeter.git
cd jokemeter
```

### 2. Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el servidor

```bash
uvicorn app.main:app --reload
```

📍 Abre [http://localhost:8000](http://localhost:8000) en tu navegador.

---

## 🐳 Ejecutar con Docker

### 1. Construir la imagen

```bash
docker build -t jokemeter .
```

### 2. Ejecutar el contenedor

```bash
docker run -d -p 8000:8000 --name jokemeter_container jokemeter
```

📍 Abre [http://localhost:8000](http://localhost:8000) en tu navegador.

---

## 📁 Estructura del proyecto

```
jokemeter/
│
├── app/                 # Código fuente principal de la aplicación (FastAPI, scripts, etc.)
├── datos/               # Conjunto de datos utilizados para entrenamiento/pruebas
├── modelos/             # Modelos entrenados y archivos relacionados
├── notebooks/           # Jupyter Notebooks para experimentación y pruebas
│
├── .dvc/                # Archivos y configuración interna de DVC (no editar manualmente)
├── .dvcignore           # Archivos/Directorios ignorados por DVC
├── .gitignore           # Archivos/Directorios ignorados por Git
├── Dockerfile           # Configuración de la imagen Docker
├── entrypoint.sh        # Script de inicio para Docker o despliegue
├── models.dvc           # Archivo de control de DVC para la carpeta de modelos
├── requirements.txt     # Dependencias de Python
├── README.md            # Documentación principal del proyecto
```

---

## ⭐ Equipo

Este proyecto fue posible gracias a:

- **Andrés Vergara** 
- **Andres Lenis**
- **Kevin Castellanos**
- **Nelson Penagos**
- **Nicolas Oviedo**

---

## 📜 Licencia

MIT License © 2025
