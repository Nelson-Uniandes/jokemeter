# Importación de librerías
import io
import time
import numpy as np

from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, pipeline
from transformers import AutoModel

from sklearn.model_selection import train_test_split

# import pandas as pd
# from sklearn.utils import resample
# from torch.utils.data import DataLoader, random_split

# import re
# from datasets import Dataset, DatasetDict

# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, log_loss
# from sklearn.neural_network import MLPClassifier
import time
import argparse


import warnings
warnings.filterwarnings("ignore")

# Configurar semillas para facilitar la reproducibilidad de los resultados
seed = 99
torch.manual_seed(seed)
np.random.seed(seed)


from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda"

# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen1.5-1.8B-Chat",
#     # torch_dtype="auto",
#     torch_dtype=torch.float16,  # aquí indicas float16

#     device_map="auto"
# )

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm




class PredictLLama():
    def __init__(self, device='auto'):
        PredictLLama.model = None
        PredictLLama.model_path  = "./app/models/modelos-llama/"
        

        self.device = device


    def loadModel(self):
        # Selección de dispositivo
        # if self.device == 'cpu':
        #     device = -1
        # elif self.device == 'auto':
        #     device = 0 if torch.cuda.is_available() else -1
        # elif self.device.isdigit():
        #     device = int(self.device)
        # else:
        #     raise ValueError(f"Dispositivo no reconocido: {self.device}")
        if self.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = self.device
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        if PredictLLama.model is None:
            PredictLLama.model = AutoModelForCausalLM.from_pretrained(PredictLLama.model_path, trust_remote_code=True)
            PredictLLama.tokenizer = AutoTokenizer.from_pretrained(PredictLLama.model_path, trust_remote_code=True)
            # PredictLLama.generator = pipeline("text-generation", model=PredictLLama.model, tokenizer=PredictLLama.tokenizer, device_map="auto")
            PredictLLama.classifier = pipeline("text-generation", model=PredictLLama.model, tokenizer=PredictLLama.tokenizer,device=device)

        PredictLLama.model.to(device)
            # generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    
    def predict_is_joke(self,text):
        # predictions = []
        prompt_template = """
    Analiza el siguiente texto y clasifícalo ÚNICAMENTE con una de estas etiquetas:
    [humor], [no humor].

    Texto: "{texto}"

    Respuesta (solo la etiqueta, sin explicaciones):""".strip()
        

        # Generar prompt
        prompt = prompt_template.format(texto=text[:3000])

        # Clasificación
        output = PredictLLama.classifier(
            prompt,
            max_new_tokens=4,  # Suficiente para las etiquetas
            do_sample=False,
            temperature=0.0,  # Para mayor determinismo
            pad_token_id=50256  # Usar el pad token de LLaMA si es necesario
        )[0]["generated_text"]
        
        # Procesamiento robusto de la respuesta
        respuesta = output.replace(prompt, "").strip().lower()
        respuesta = respuesta.split()[0] if respuesta else ""
        #print(respuesta)

        # Validar y normalizar etiqueta
        etiqueta = 1 if "humor" in respuesta else 0

        # prediction = PredictLLama.generator(prompt, max_new_tokens=1, do_sample=True)[0]['generated_text'][-1]
        # predictions.append(prediction)
        return etiqueta
    
    # def predict_is_joke(self,text):
    #     # predictions = []
    #     prompt = f"Clasifica el siguiente texto con los valores 0 o 1, donde 1 indica que es un texto con contenido humorístico y 0 en caso contrario, retorna únicamente el valor 0 o 1:\n{text}"
    #     prediction = PredictLLama.generator(prompt, max_new_tokens=5, do_sample=True)[0]['generated_text']
    #     # predictions.append(prediction)
    #     return prediction
    

if __name__ == "__main":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None, help='cpu, cuda, 0 (GPU index) o auto')
    args = parser.parse_args()

    prediction_qwen = PredictLLama(device=args.device)
    prediction_qwen.loadModel()

    list_texts =  [
        'HOLA QUE MAS',
        'HOLA QUE MAS OTRO TEXTO',
        'HOLA QUE MAS ALGO MAS QUI'
    ]
    for text in list_texts:
        start_time = time.time()
        score = prediction_qwen.predict_score(text)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f'{score=}, tiempo: {elapsed:.4f} segundos')