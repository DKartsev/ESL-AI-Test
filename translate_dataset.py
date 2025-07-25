"""Translate the UniversalCEFR dataset in a streaming fashion."""

from datasets import load_dataset
from transformers import pipeline
from tqdm.auto import tqdm
import pandas as pd


def main() -> None:
    """Stream the UniversalCEFR dataset and translate it batch by batch."""

    # Загрузка датасета UniversalCEFR в режиме стриминга
    dataset = load_dataset("UniversalCEFR/cefr_sp_en", split="train", streaming=True)

    # Инициализация переводчика (английский → эсперанто)
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-eo")

    batch_size = 16
    buffer_texts = []
    buffer_labels = []
    results = []

    for example in tqdm(dataset, desc="Translating", unit="example"):
        buffer_texts.append(example["text"])
        buffer_labels.append(example["label"])

        if len(buffer_texts) == batch_size:
            outs = translator(buffer_texts, max_length=256)
            for out, label in zip(outs, buffer_labels):
                results.append({"text_eo": out["translation_text"], "level": label})
            buffer_texts = []
            buffer_labels = []

    # Обработка оставшихся примеров
    if buffer_texts:
        outs = translator(buffer_texts, max_length=256)
        for out, label in zip(outs, buffer_labels):
            results.append({"text_eo": out["translation_text"], "level": label})

    # Сохранение результатов в CSV
    pd.DataFrame(results).to_csv("translated_cefr_dataset.csv", index=False)
    print("✅ Датасет сохранён в translated_cefr_dataset.csv")


if __name__ == "__main__":
    main()
