import os
import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# Carpeta donde están los modelos .pkl
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained")
_cache = {}

# Asociación entre modelo .pkl y encoder/tokenizer
MODEL_CONFIG = {
    "RoBERTa_mlp_classifier_bin": {
        "encoder_name": "./models/roberta-base-bne/model",
        "tokenizer_path": "./models/roberta-base-bne/tokenizer"
    },
    "RoBERTa_mlp_classifier_multi":{
        "encoder_name": "./models/roberta-base-bne/model",
        "tokenizer_path": "./models/roberta-base-bne/tokenizer"
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
        result = {}

        for folder_name in os.listdir(MODEL_DIR):
            folder_path = os.path.join(MODEL_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue

            model_types = []
            for file in os.listdir(folder_path):
                if file.endswith(".pkl"):
                    fname = file.lower()
                    if "bin" in fname:
                        model_types.append("bin")
                    elif "multi" in fname or "score" in fname:
                        model_types.append("multi")
                    else:
                        model_types.append("otro")

            if model_types:
                result[folder_name] = sorted(set(model_types))

        return [f"{k} → {' + '.join(v)}" for k, v in result.items()]

    except Exception as e:
        print("❌ Error al listar modelos:", e)
        return []
