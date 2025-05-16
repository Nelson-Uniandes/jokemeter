from app.models.loader import load_model, load_tokenizer_and_encoder
import torch
from app.models.qwen_predict import PredictQwen
import anthropic
import os

# PredictQwen(device='cpu').loadModel()

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
        "is_funny": prediction_binary,
        "confidence": round(0, 2),
        "score": prediction_score
    }

def evaluate_joke(joke: str, model_name: str):
    try:
        if model_name == 'Claude':
            return evaluate_joke_claude(joke)
    
        elif model_name == 'QWEN':
            qwen_predictor = PredictQwen(device='cpu')
            score = qwen_predictor.predict_score(joke)
            is_joke = qwen_predictor.predict_is_joke(joke)
            return {
                'score':score,
                'is_funny':is_joke
                
            }

        clf = load_model(model_name)
        tokenizer, encoder = load_tokenizer_and_encoder(model_name)

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
        #score = float(clf.predict_proba(embedding)[0][1])

        return {
            "is_funny": is_funny,
            "confidence": round(confidence, 2),
         #   "score": round(score, 2)
        }

    except Exception as e:
        return {"error": str(e)}