# download_cefr_dataset.py

from datasets import load_dataset
import pandas as pd

# Загружаем датасет UniversalCEFR/cefr_sp_en с Hugging Face
print("🔄 Загружаем датасет с Hugging Face...")
dataset = load_dataset("UniversalCEFR/cefr_sp_en", split="train")

# Преобразуем в DataFrame
df = pd.DataFrame(dataset)

# Сохраняем в файл
df.to_csv("cefr_data.csv", index=False, encoding="utf-8")
print("✅ Готово! Датасет сохранён в файл cefr_data.csv")
