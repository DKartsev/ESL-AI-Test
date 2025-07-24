from datasets import load_dataset, Dataset
from transformers import pipeline
import pandas as pd

# Загрузка датасета UniversalCEFR
dataset = load_dataset("UniversalCEFR/cefr_sp_en", split="train")
print(f"🔹 Загружено примеров: {len(dataset)}")

# Инициализация переводчика (английский → эсперанто)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-eo")

# Перевод всех фраз
translations = translator([x["text"] for x in dataset], max_length=256, batch_size=16)

# Подготовка нового датасета с переводом
translated_data = pd.DataFrame({
    "text_eo": [t["translation_text"] for t in translations],
    "level": [x["label"] for x in dataset]
})

# Сохранение в CSV
translated_data.to_csv("translated_cefr_dataset.csv", index=False)
print("✅ Датасет с эсперанто фразами сохранён в translated_cefr_dataset.csv")
