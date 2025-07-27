# ✅ Установка зависимостей (выполняй ОДИН РАЗ в терминале, не в коде!)
# pip install datasets transformers pandas tqdm

# 📦 Импорты
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline

# 📥 Список датасетов и модели перевода
dataset_names = [
    "UniversalCEFR/caes_es",       # испанский
    "UniversalCEFR/cefr_sp_en",    # испанский ↔ английский
    "UniversalCEFR/cefr_asag_en",  # английский
    "UniversalCEFR/merlin_de",     # немецкий
    "UniversalCEFR/kwiqiz_fr",     # французский
    "UniversalCEFR/elle_et",       # эстонский
]

lang_to_model = {
    "es": "Helsinki-NLP/opus-mt-es-eo",
    "en": "Helsinki-NLP/opus-mt-en-eo",
    "de": "Helsinki-NLP/opus-mt-de-eo",
    "fr": "Helsinki-NLP/opus-mt-fr-eo",
    "et": "Helsinki-NLP/opus-mt-et-eo",
}

# 📥 Загрузка всех датасетов
dfs = []
for dataset_name in dataset_names:
    print(f"🔽 Загружаем {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    df = dataset.to_pandas()
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
print(f"✅ Всего строк: {len(combined_df)}")

# 🎯 Только нужные колонки
df = combined_df[["text", "cefr_level", "lang"]].dropna()

# 🔁 Перевод текста
translated_texts = []
current_lang = None
translator = None

print("🔄 Переводим на эсперанто...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["text"]
    lang = row["lang"]

    if lang != current_lang:
        model_name = lang_to_model.get(lang)
        if model_name is None:
            translated_texts.append("N/A")
            continue
        print(f"\n🌐 Загружается модель: {model_name}")
        translator = pipeline("translation", model=model_name,
                              device=0 if torch.cuda.is_available() else -1)
        current_lang = lang

    try:
        translated = translator(text[:512])[0]["translation_text"]
    except Exception as e:
        print(f"⚠️ Ошибка при переводе: {e}")
        translated = "N/A"

    translated_texts.append(translated)

# 🧪 Добавим колонку
df["esperanto"] = translated_texts

# 💾 Сохраняем файл
output_file = "translated_cefr_dataset.csv"
df.to_csv(output_file, index=False)
print(f"✅ Готово! Файл сохранён: {output_file}")
