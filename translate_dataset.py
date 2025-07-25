from datasets import load_dataset, Dataset
from transformers import pipeline
from tqdm.auto import tqdm
import pandas as pd


def main() -> None:
    """Translate the UniversalCEFR dataset to Esperanto and save it."""
    # Загрузка датасета UniversalCEFR
    dataset = load_dataset("UniversalCEFR/cefr_sp_en", split="train")
    print(f"🔹 Загружено примеров: {len(dataset)}")

    # Инициализация переводчика (английский → эсперанто)
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-eo")

    # Перевод всех фраз
    texts = [x["text"] for x in dataset]
    translations = []
    for i in tqdm(range(0, len(texts), 16), desc="Translating"):
        batch = texts[i:i+16]
        out = translator(batch, max_length=256)
        translations.extend(out)

    # Подготовка нового датасета с переводом
    translated_data = pd.DataFrame({
        "text_eo": [t["translation_text"] for t in translations],
        "level": [x["label"] for x in dataset]
    })

    # Сохранение в CSV
    translated_data.to_csv("translated_cefr_dataset.csv", index=False)
    print(
        "✅ Датасет с эсперанто фразами сохранён в translated_cefr_dataset.csv"
    )


if __name__ == "__main__":
    main()
