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
├── app/
│   ├── main.py
│   ├── models/
│   │   └── dummy_model.py
│   ├── static/
│   │   ├── styles.css
│   │   └── script.js
│   └── templates/
│       └── index.html
├── requirements.txt
├── Dockerfile
└── README.md
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
