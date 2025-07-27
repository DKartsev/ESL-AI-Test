# translate_cefr_dataset.py

import torch
from datasets import load_dataset
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os

# Язык → модель перевода
lang_to_model = {
    "es": "Helsinki-NLP/opus-mt-es-eo",
    "en": "Helsinki-NLP/opus-mt-en-eo",
    "de": "Helsinki-NLP/opus-mt-de-eo",
    "fr": "Helsinki-NLP/opus-mt-fr-eo",
    "et": "Helsinki-NLP/opus-mt-et-eo",
}

# Указать нужный датасет
dataset_name = "UniversalCEFR/caes_es"
output_file = "translated_caes_es.csv"

print(f"🔽 Загружаем {dataset_name}...")
dataset = load_dataset(dataset_name, split="train")
df = dataset.to_pandas()

# Оставляем нужные колонки
df = df[["text", "cefr_level", "lang"]].dropna()
df["esperanto"] = ""

# Настройки
device = 0 if torch.cuda.is_available() else -1
current_lang = df["lang"].iloc[0]
model_name = lang_to_model.get(current_lang)

if not model_name:
    raise ValueError(f"Модель для языка {current_lang} не найдена.")

print(f"🌐 Загружаем модель: {model_name}")
translator = pipeline("translation", model=model_name, device=device)

# Перевод
translated_texts = []

print("🔄 Переводим...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        translation = translator(row["text"][:512])[0]["translation_text"]
    except:
        translation = "N/A"
    translated_texts.append(translation)

df["esperanto"] = translated_texts

# Сохранение
df.to_csv(output_file, index=False)
print(f"✅ Переведённый файл сохранён: {output_file}")
