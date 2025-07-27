# translate_dataset.py

from datasets import load_dataset
from transformers import pipeline
import pandas as pd
from tqdm import tqdm


def main():
    # Загружаем датасет
    dataset = load_dataset("UniversalCEFR/cefr_sp_en", split="train")

    # Инициализируем pipeline для перевода с английского на эсперанто
    translator = pipeline(
        "translation", model="Helsinki-NLP/opus-mt-en-eo", device=-1)

    # Списки для хранения результатов
    buffer_source = []
    buffer_target = []
    buffer_labels = []

    # Переводим все английские фразы на эсперанто
    for example in tqdm(dataset, desc="Translating"):
        english = example["text"]
        esperanto = translator(english, max_length=128)[0]["translation_text"]

        buffer_source.append(esperanto)
        buffer_target.append(english)
        buffer_labels.append(example["cefr_level"])  # исправили здесь!

    # Сохраняем в DataFrame
    df = pd.DataFrame({
        "esperanto": buffer_source,
        "english": buffer_target,
        "cefr_level": buffer_labels
    })

    # Сохраняем в CSV
    df.to_csv("translated_cefr_dataset.csv", index=False, encoding="utf-8")
    print("✅ Переведённый датасет сохранён в translated_cefr_dataset.csv")


if __name__ == "__main__":
    main()
