import os
import joblib
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Carpeta donde están los modelos .pkl
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained")
_cache = {}

# Asociación entre modelo .pkl y encoder/tokenizer
MODEL_CONFIG = {
    "RoBERTa_mlp_classifier_bin": {
        "encoder_name": "./models/roberta-base-bne/model",
        "tokenizer_path": "./models/roberta-base-bne/tokenizer",
        "type": "encoder"
    },
    "RoBERTa_mlp_classifier_multi":{
        "encoder_name": "./models/roberta-base-bne/model",
        "tokenizer_path": "./models/roberta-base-bne/tokenizer",
        "type": "encoder"
    },
    "Llama_mlp_classifier_bin":{
        "encoder_name": "./models/llama/model",
        "tokenizer_path": "./models/llama/tokenizer",
        "type": "classification" 
    }
}

def load_model(name: str):
    try:
        if name in _cache:
            return _cache[name]

        # Buscar el modelo en subcarpetas de trained/
        for folder in os.listdir(MODEL_DIR):
            folder_path = os.path.join(MODEL_DIR, folder)
            if not os.path.isdir(folder_path):
                continue

            model_path = os.path.join(folder_path, f"{name}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                _cache[name] = model
                return model

        raise FileNotFoundError(f"Modelo '{name}' no encontrado en subcarpetas de {MODEL_DIR}")

    except Exception as e:
        print(f"❌ Error al cargar el modelo '{name}':", e)
        raise

def load_tokenizer_and_encoder(model_name: str):
    """
    Carga tokenizer y modelo, soportando:
    - Modelos base (encoder) para extracción de embeddings
    - Modelos fine-tuned para clasificación directa (ej: Llama, BERT, RoBERTa fine-tuned)
    El tipo se define en MODEL_CONFIG con la clave "type": "encoder" o "classification"
    """
    try:
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Modelo '{model_name}' no está registrado en MODEL_CONFIG")

        config = MODEL_CONFIG[model_name]
        encoder_path = config["encoder_name"]
        tokenizer_path = config["tokenizer_path"]
        model_type = config.get("type", "encoder")  # Por defecto "encoder"

        tokenizer_key = f"tokenizer::{tokenizer_path}"
        encoder_key = f"encoder::{encoder_path}::{model_type}"

        # Tokenizer siempre igual
        if tokenizer_key not in _cache:
            _cache[tokenizer_key] = AutoTokenizer.from_pretrained(tokenizer_path)

        # Modelo depende del tipo
        if encoder_key not in _cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if model_type == "classification":
                _cache[encoder_key] = AutoModelForSequenceClassification.from_pretrained(encoder_path).to(device)
            else:
                _cache[encoder_key] = AutoModel.from_pretrained(encoder_path).to(device)

        return _cache[tokenizer_key], _cache[encoder_key]

    except Exception as e:
        print(f"❌ Error al cargar tokenizer/encoder para '{model_name}':", e)
        raise


def list_available_models():
    return ["roberta → bin + multi", "llama → bin + qwen → multi"]

