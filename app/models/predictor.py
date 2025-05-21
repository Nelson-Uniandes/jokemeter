from app.models.loader import load_model, load_tokenizer_and_encoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NAME_OVERRIDES = {
    "roberta": "RoBERTa",
}

def generate_model_dict(entry: str):
    try:
        base_part, types_part = entry.split("→")
        base = base_part.strip()
        types = [t.strip() for t in types_part.strip().split("+")]

        # Usa estilos personalizados si existen
        base_formatted = NAME_OVERRIDES.get(base.lower(), base.capitalize())

        return {
            t: f"{base_formatted}_mlp_classifier_{t}"
            for t in types
        }

    except Exception as e:
        print("❌ Error procesando entrada:", e)
        return {}

def detect_models(s: str):
    s_lower = s.lower()
    return "llama" in s_lower or "qwen" in s_lower

def evaluate_joke(joke: str, model_name: str):
    model = {
        "llama":"Llama_mlp_classifier_bin"
    }
    print(model_name, flush=True)
    try:
        #model_name = "llama"
        # Detecta si es modelo Llama (ajusta el string si tienes otro nombre para Llama2)
        if detect_models(model_name):
            print("LLAMA o QWEN detectado", flush=True)
            print(model["llama"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer, model = load_tokenizer_and_encoder(model["llama"])
            model.eval()
            inputs = tokenizer(joke, return_tensors="pt", truncation=True, padding=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred].item()
                score = 0
            return {
                "is_funny": bool(pred),
                "confidence": round(confidence, 2),
                "score": score
            }
        
        print("ROBERTA", flush=True)
        # ---- Resto de modelos, pipeline original ----
        model_names = generate_model_dict(model_name)
        clf = load_model(model_names["bin"])
        clf_multi = load_model(model_names["multi"])
        tokenizer, encoder = load_tokenizer_and_encoder(model_names["bin"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(joke, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = encoder(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

        prediction = clf.predict(embedding)[0]
        proba = clf.predict_proba(embedding)[0]
        
        is_funny = bool(prediction)
        confidence = float(proba[prediction])
        score = float(clf_multi.predict(embedding)[0])

        return {
            "is_funny": is_funny,
            "confidence": round(confidence, 2),
            "score": score
        }

    except Exception as e:
        return {"error": str(e)}