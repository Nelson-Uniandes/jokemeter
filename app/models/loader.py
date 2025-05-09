import os
import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# Carpeta donde están los modelos .pkl
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained")
_cache = {}

# Asociación entre modelo .pkl y encoder/tokenizer
MODEL_CONFIG = {
    "modelo_mlp_roberta": {
        "encoder_name": "./models/roberta-base-bne/model",
        "tokenizer_path": "./models/roberta-base-bne/tokenizer"
    }
}

def load_model(name: str):
    try:
        if name in _cache:
            return _cache[name]

        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo '{name}' no encontrado en {MODEL_DIR}")

        model = joblib.load(path)
        _cache[name] = model
        return model

    except Exception as e:
        print(f"❌ Error al cargar el modelo '{name}':", e)
        raise

def load_tokenizer_and_encoder(model_name: str):
    try:
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Modelo '{model_name}' no está registrado en MODEL_CONFIG")

        encoder_path = MODEL_CONFIG[model_name]["encoder_name"]
        tokenizer_path = MODEL_CONFIG[model_name]["tokenizer_path"]

        tokenizer_key = f"tokenizer::{tokenizer_path}"
        encoder_key = f"encoder::{encoder_path}"

        if tokenizer_key not in _cache:
            _cache[tokenizer_key] = AutoTokenizer.from_pretrained(tokenizer_path)

        if encoder_key not in _cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _cache[encoder_key] = AutoModel.from_pretrained(encoder_path).to(device)

        return _cache[tokenizer_key], _cache[encoder_key]

    except Exception as e:
        print(f"❌ Error al cargar tokenizer/encoder para '{model_name}':", e)
        raise


def list_available_models():
    try:
        return [
            f.replace(".pkl", "") for f in os.listdir(MODEL_DIR)
            if f.endswith(".pkl")
        ]
    except Exception as e:
        print("❌ Error al listar modelos:", e)
        return []
