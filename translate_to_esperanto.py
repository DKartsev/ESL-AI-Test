import os
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
from functools import partial

# ==== 🔧 Настройки ====
OUTPUT_CSV = "translated_cefr_dataset.csv"
TEXT_COLUMN = "text"
CEFR_COLUMN = "cefr_level"
LANG_COLUMN = "lang"
BATCH_SIZE = 8  # можно увеличить, если хватает VRAM

# Датасеты и модели
dataset_names = {
    "es": "UniversalCEFR/caes_es",
    "en": "UniversalCEFR/cefr_asag_en",
    "de": "UniversalCEFR/merlin_de",
    "fr": "UniversalCEFR/kwiqiz_fr",
    "et": "UniversalCEFR/elle_et",
}

lang_to_model = {
    "es": "Helsinki-NLP/opus-mt-es-eo",
    "en": "Helsinki-NLP/opus-mt-en-eo",
    "de": "Helsinki-NLP/opus-mt-de-eo",
    "fr": "Helsinki-NLP/opus-mt-fr-eo",
    "et": "Helsinki-NLP/opus-mt-et-eo",
}


# ==== 🔁 Переводим батч (для map)
def translate_batch(examples, lang, model_name):
    translator = pipeline("translation", model=model_name,
                          device=0 if torch.cuda.is_available() else -1)
    texts = examples[TEXT_COLUMN]
    translations = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        try:
            outputs = translator(batch, max_length=512)
            batch_translations = [o["translation_text"] for o in outputs]
        except Exception as e:
            print(f"⚠️ Ошибка при переводе: {e}")
            batch_translations = ["N/A"] * len(batch)
        translations.extend(batch_translations)

    return {"esperanto": translations}


# ==== ⬇ Главная точка входа (важно для Windows!)
def main():
    all_datasets = []

    for lang_code, dataset_name in dataset_names.items():
        print(f"\n🔽 Загружается {dataset_name}...")
        ds = load_dataset(dataset_name, split="train")
        ds = ds.filter(lambda ex: ex[TEXT_COLUMN]
                       is not None and ex[CEFR_COLUMN] is not None)

        if LANG_COLUMN not in ds.column_names:
            ds = ds.add_column(LANG_COLUMN, [lang_code] * len(ds))

        model_name = lang_to_model[lang_code]
        print(f"🌐 Перевод {lang_code} через {model_name}...")

        # 🔁 Перевод с map и multiprocessing
        ds = ds.map(
            partial(translate_batch, lang=lang_code, model_name=model_name),
            batched=True,
            batch_size=64,
            num_proc=2,  # Можно уменьшить, если слабый ПК
        )

        all_datasets.append(ds)

    # Объединяем всё
    final_dataset = concatenate_datasets(all_datasets)
    df = final_dataset.to_pandas()
    df = df[[TEXT_COLUMN, CEFR_COLUMN, LANG_COLUMN, "esperanto"]]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Датасет сохранён в {OUTPUT_CSV}")


# ==== ⚙ Запуск
if __name__ == "__main__":
    main()
