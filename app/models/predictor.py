from app.models.loader import load_model, load_tokenizer_and_encoder
import torch
from app.models.qwen_predict import PredictQwen
from app.models.llama_predict import PredictLLama
import anthropic
import os

PredictQwen(device='cpu').loadModel()
PredictLLama(device='cpu').loadModel()

print("OK")
print("CLAUDE_TOKEN", os.getenv('CLAUDE_TOKEN'))

def evaluate_joke_claude(text:str):
    api_key = os.getenv('CLAUDE_TOKEN')
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[
                {"role": "user", "content": f"Clasifica el siguiente texto con los valores 0 o 1, donde 1 indica que es un texto con contenido humorístico y 0 en caso contrario, retorna únicamente el valor 0 o 1:\n{text}"}
            ]
        )
    prediction_binary = response.content[0].text



    response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[
                {"role": "user", "content": f"Asigna un puntaje entre 1 y 5 dependiendo del nivel de gracia causado por el siguiente texto, donde 1 es bajo y 5 es alto, retorna únicamente el puntaje:\n{text}"}
            ]
        )
    prediction_score = response.content[0].text
    return {
        "is_funny": int(prediction_binary),
        "confidence": round(0, 2),
        "score": int(prediction_score)
    }

NAME_OVERRIDES = {
    "roberta": "RoBERTa",
    "xlm-r": "XLM-R",
    "maria": "MarIA"
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


def evaluate_joke(joke: str, model_name: str):
    try:
        if model_name == 'Claude':
            return evaluate_joke_claude(joke)
    
        elif model_name == 'LLama - QWEN':
            qwen_predictor = PredictQwen(device='cpu')
            llama_predictor = PredictLLama(device='cpu')
            score = int(qwen_predictor.predict_score(joke))
            is_joke = llama_predictor.predict_is_joke(joke)
            return {
                'score':score,
                'is_funny':is_joke
                
            }

        clf = load_model(model_name)
        tokenizer, encoder = load_tokenizer_and_encoder(model_name)
        model_names = generate_model_dict(model_name)
        # Modelo binario y multi-score
        clf = load_model(model_names["bin"])
        clf_multi = load_model(model_names["multi"])
        tokenizer, encoder = load_tokenizer_and_encoder(model_names["bin"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(joke, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = encoder(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

        # Clasificación
        prediction = clf.predict(embedding)[0]
        proba = clf.predict_proba(embedding)[0]
        
        is_funny = bool(prediction)
        confidence = float(proba[prediction])

        # Score (multi)
        score = float(clf_multi.predict(embedding)[0])

        return {
            "is_funny": is_funny,
            "confidence": round(confidence, 2),
            "score": score
        }

    except Exception as e:
        return {"error": str(e)}