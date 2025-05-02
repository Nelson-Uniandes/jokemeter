import random

def score_joke(joke: str, model_name: str):
    score = round(random.uniform(1, 10), 1)
    if score >= 8:
        feedback = "Pretty funny! You might have a future in comedy."
    elif score >= 5:
        feedback = "Meh... it has potential."
    else:
        feedback = "Not your best work... try again!"
    return score, feedback
