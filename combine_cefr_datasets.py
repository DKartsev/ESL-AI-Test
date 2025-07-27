# combine_cefr_datasets.py

from datasets import load_dataset
import pandas as pd

# Список датасетов: (имя в HF, язык)
datasets_to_load = [
    ("UniversalCEFR/caes_es", "es"),
    ("UniversalCEFR/cefr_sp_en", "es-en"),
    ("UniversalCEFR/cefr_asag_en", "en"),
    ("UniversalCEFR/merlin_de", "de"),
    ("UniversalCEFR/kwiqiz_fr", "fr"),
    ("UniversalCEFR/elle_et", "et"),
]

combined_data = []

for dataset_name, lang_code in datasets_to_load:
    print(f"📥 Загружаем {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split="train")
        df = dataset.to_pandas()

        # Проверим, есть ли нужные колонки
        if not all(col in df.columns for col in ["text", "cefr_level"]):
            print(f"⚠️ Пропущен {dataset_name}: нет нужных колонок.")
            continue

        df = df[["text", "cefr_level"]].copy()
        df["lang"] = lang_code
        df["source_name"] = dataset_name

        combined_data.append(df)
        print(f"✅ Загружено: {len(df)} строк.")
    except Exception as e:
        print(f"❌ Ошибка при загрузке {dataset_name}: {e}")

# Объединение всех данных
if combined_data:
    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.to_csv("universal_cefr_combined.csv", index=False)
    print(f"\n📁 Сохранён объединённый датасет: universal_cefr_combined.csv")
    print(f"📊 Общее количество записей: {len(combined_df)}")
else:
    print("❌ Не удалось собрать ни одного датасета.")
