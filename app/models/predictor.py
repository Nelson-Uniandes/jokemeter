from app.models.loader import load_model, load_tokenizer_and_encoder
import torch

def evaluate_joke(joke: str, model_name: str):
    try:
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